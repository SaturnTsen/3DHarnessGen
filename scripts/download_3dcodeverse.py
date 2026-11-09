import argparse
import os
import re
import threading
import time

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from fnmatch import fnmatchcase
from functools import wraps
from pathlib import Path
from queue import Empty, Full, Queue

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.utils import parse_ratelimit_headers
from tqdm import tqdm


REPO_ID = "ilabai/3dcodeverse"
REPO_TYPE = "dataset"
LOCAL_DIR = "./3dcodeverse"

MAX_WORKERS = 32

RETRY_DELAY = 2
MAX_RETRY_DELAY = 60

REQUEST_INTERVAL = 0.0
RATE_LIMIT_BUFFER = 10
DEFAULT_RATE_LIMIT_WAIT = 310

SKIP_EXISTING = True

PATTERNS = [
    "dataset_info*.json",
    "3dcodebench/instances_geo_full_llamafactory_img/*",
    "3dcodebench/instances_tex_full_llamafactory_img/*",
    "3dcodebench/pair_renders/instances_geo_full/*",
    "3dcodebench/pair_renders/instances_tex_full/*",
    "bioinspired3d/bpy_structures_geo_llamafactory_img/*",
    "bioinspired3d/bpy_structures_tex_llamafactory_img/*",
    "bioinspired3d/pair_renders/*",
    "deepcad/*_llamafactory_img/*",
    "deepcad/pair_renders/*",
    "articraft/*_llamafactory_img/*",
    "articraft/pair_renders/*",
    "threejs_coding_agent/threejs_coding_agent_llamafactory_img/*",
    "threejs_coding_agent/pair_renders/*",
    "thingiverse/*_llamafactory_img/*",
    "thingiverse/pair_renders/*",
]


class QuietFileProgress(tqdm):
    """Keep concurrent per-file bars from overwriting the overall progress."""

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


class RequestLimiter:
    """Coordinate optional pacing and server-requested cooldown across workers."""

    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_request_time = 0.0
        self.blocked_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                delay = max(self.next_request_time, self.blocked_until) - now
                if delay <= 0:
                    self.next_request_time = now + self.interval
                    return
            time.sleep(min(delay, 1.0))

    def handle_rate_limit(self, error):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        message = str(error)
        if status != 429 and not re.search(
            r"\b429\b|Too Many Requests", message, re.IGNORECASE
        ):
            return False

        delay = DEFAULT_RATE_LIMIT_WAIT
        match = re.search(r"Retry after (\d+) seconds", message, re.IGNORECASE)
        if match:
            delay = int(match.group(1))
        if response is not None:
            rate_limit = parse_ratelimit_headers(response.headers)
            retry_after = response.headers.get("Retry-After", "").strip()
            if rate_limit is not None:
                delay = rate_limit.reset_in_seconds
            elif retry_after.isdigit():
                delay = int(retry_after)

        with self.lock:
            now = time.monotonic()
            was_blocked = self.blocked_until > now
            self.blocked_until = max(self.blocked_until, now + delay + RATE_LIMIT_BUFFER)
            if not was_blocked:
                tqdm.write(f"[429] Global cooldown: {self.blocked_until - now:.0f}s")
        return True

    def retry_forever(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = RETRY_DELAY
            while True:
                self.wait()
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    if self.handle_rate_limit(error):
                        delay = RETRY_DELAY
                        continue
                    tqdm.write(
                        f"[Retry] {func.__name__}: {type(error).__name__}: {error}"
                        f" | retry in {delay}s"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, MAX_RETRY_DELAY)

        return wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download selected 3DCodeVerse files concurrently; skip completed files."
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help="Concurrent file downloads (default: %(default)s)",
    )
    parser.add_argument(
        "--request-interval", type=float, default=REQUEST_INTERVAL,
        help="Minimum seconds between task starts (default: %(default)s)",
    )
    parser.add_argument(
        "--local-dir", default=LOCAL_DIR,
        help="Download directory (default: %(default)s)",
    )
    parser.add_argument(
        "--pattern", action="append", dest="patterns",
        help="Override default file patterns; repeat to select multiple patterns",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 0 <= args.request_interval < float("inf"):
        parser.error("--request-interval must be finite and non-negative")
    return args


def main():
    args = parse_args()
    limiter = RequestLimiter(args.request_interval)
    api = HfApi()

    local_root = Path(args.local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    @limiter.retry_forever
    def get_revision():
        return api.repo_info(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        ).sha

    revision = get_revision()

    print(f"Repository: {REPO_ID}", flush=True)
    print(f"Revision: {revision}", flush=True)
    print(f"Local directory: {local_root.resolve()}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Request interval: {args.request_interval}s", flush=True)

    tree_cache = {}

    @limiter.retry_forever
    def fetch_tree(path):
        return list(
            api.list_repo_tree(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                path_in_repo=path or None,
                recursive=False,
                revision=revision,
            )
        )

    def list_tree(path):
        if path not in tree_cache:
            tree_cache[path] = fetch_tree(path)

        return tree_cache[path]

    def walk(path):
        limiter.wait()

        for item in api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=path,
            recursive=True,
            revision=revision,
        ):
            if isinstance(item, RepoFile):
                yield item.path

    def expand_pattern(pattern):
        parts = pattern.split("/")
        file_pattern = parts[-1]
        directories = [""]

        for component in parts[:-1]:
            next_directories = []

            for parent in directories:
                for item in list_tree(parent):
                    if (
                        isinstance(item, RepoFolder)
                        and fnmatchcase(
                            Path(item.path).name,
                            component,
                        )
                    ):
                        next_directories.append(item.path)

            directories = next_directories

        for directory in directories:
            if file_pattern == "*":
                yield from walk(directory)

            else:
                for item in list_tree(directory):
                    if (
                        isinstance(item, RepoFile)
                        and fnmatchcase(
                            Path(item.path).name,
                            file_pattern,
                        )
                    ):
                        yield item.path

    def download(filename):
        local_path = local_root / filename

        if SKIP_EXISTING and local_path.is_file():
            return "skipped"

        delay = RETRY_DELAY

        while True:
            if SKIP_EXISTING and local_path.is_file():
                return "skipped"

            limiter.wait()

            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    filename=filename,
                    revision=revision,
                    local_dir=local_root,
                    tqdm_class=QuietFileProgress,
                )

                return "downloaded"

            except Exception as e:
                if limiter.handle_rate_limit(e):
                    delay = RETRY_DELAY
                    continue

                tqdm.write(
                    f"[Download Retry] {filename}: "
                    f"{type(e).__name__}: {e}"
                    f" | retry in {delay}s"
                )

                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)

    patterns = args.patterns or PATTERNS
    # Bound scan-ahead and queued download work.
    discoveries = Queue(maxsize=args.workers * 4)
    stop_scanning = threading.Event()

    def publish(kind, value=None):
        while not stop_scanning.is_set():
            try:
                discoveries.put((kind, value), timeout=1)
                return
            except Full:
                continue

    def scan_files():
        for pattern in patterns:
            if stop_scanning.is_set():
                return
            tqdm.write(f"Scanning: {pattern}")
            scan_delay = RETRY_DELAY
            while not stop_scanning.is_set():
                try:
                    for filename in expand_pattern(pattern):
                        if stop_scanning.is_set():
                            return
                        publish("file", filename)
                    break
                except Exception as error:
                    if limiter.handle_rate_limit(error):
                        scan_delay = RETRY_DELAY
                        limiter.wait()
                        continue
                    tqdm.write(
                        f"[Scan Retry] {pattern}: {type(error).__name__}: {error}"
                        f" | retry in {scan_delay}s"
                    )
                    time.sleep(scan_delay)
                    scan_delay = min(scan_delay * 2, MAX_RETRY_DELAY)
            publish("pattern_done")
        publish("done")

    seen = set()
    total_downloaded = 0
    total_skipped = 0
    scanned_patterns = 0
    scan_finished = False
    pending = set()

    with ThreadPoolExecutor(max_workers=args.workers) as executor, tqdm(
        total=0, desc="Download (scanning)", unit="file", dynamic_ncols=True,
    ) as progress:
        def update_status():
            progress.set_postfix(
                downloaded=total_downloaded, skipped=total_skipped, pending=len(pending),
                scan="done" if scan_finished else f"{scanned_patterns}/{len(patterns)}",
                refresh=False,
            )

        def collect_completed(done):
            nonlocal total_downloaded, total_skipped
            for future in done:
                result = future.result()
                if result == "downloaded":
                    total_downloaded += 1
                elif result == "skipped":
                    total_skipped += 1
            update_status()
            progress.update(len(done))

        scanner = threading.Thread(target=scan_files, name="file-scanner", daemon=True)
        scanner.start()
        try:
            while not scan_finished or pending:
                done = {future for future in pending if future.done()}
                pending.difference_update(done)
                collect_completed(done)

                if pending and (scan_finished or len(pending) >= args.workers * 4):
                    done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    collect_completed(done)
                    progress.refresh()
                    continue

                if scan_finished:
                    break
                try:
                    kind, filename = discoveries.get(timeout=1)
                except Empty:
                    progress.refresh()
                    continue

                if kind == "done":
                    scan_finished = True
                    progress.set_description("Download")
                elif kind == "pattern_done":
                    scanned_patterns += 1
                elif filename not in seen:
                    seen.add(filename)
                    progress.total = len(seen)
                    if SKIP_EXISTING and (local_root / filename).is_file():
                        total_skipped += 1
                        progress.update(1)
                    else:
                        pending.add(executor.submit(download, filename))
                update_status()
                progress.update(0)
        finally:
            stop_scanning.set()
        progress.refresh()

    print("\n" + "=" * 60)
    print("Download complete")
    print(f"Total files: {len(seen)}")
    print(f"Downloaded: {total_downloaded}")
    print(f"Skipped: {total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()
