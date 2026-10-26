
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from fnmatch import fnmatchcase
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile, RepoFolder
from tqdm import tqdm

REPO_ID = "ilabai/3dcodeverse"
REPO_TYPE = "dataset"
LOCAL_DIR = "./3dcodeverse"
MAX_WORKERS = 8

PATTERNS = [
    "dataset_info*.json",
    "3dcodebench/instances_geo_llamafactory_img/*",
    "3dcodebench/instances_tex_llamafactory_img/*",
    "3dcodebench/pair_renders/*",
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


def main():
    api = HfApi()
    revision = api.repo_info(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    ).sha

    tree_cache = {}

    def list_tree(path):
        if path not in tree_cache:
            tree_cache[path] = list(
                api.list_repo_tree(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    path_in_repo=path or None,
                    recursive=False,
                    revision=revision,
                )
            )
        return tree_cache[path]

    def walk(path):
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
        return hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            revision=revision,
            local_dir=LOCAL_DIR,
        )

    seen = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pending = set()

        for pattern in PATTERNS:
            print(f"\nScanning: {pattern}", flush=True)
            count = 0

            with tqdm(desc=pattern, unit="file") as progress:
                for filename in expand_pattern(pattern):
                    if filename in seen:
                        continue

                    seen.add(filename)
                    count += 1
                    pending.add(executor.submit(download, filename))

                    if len(pending) >= MAX_WORKERS * 4:
                        done, pending = wait(
                            pending,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in done:
                            future.result()
                            progress.update(1)

                while pending:
                    done, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        future.result()
                        progress.update(1)

            print(f"Matched: {count}", flush=True)

    print(f"\nComplete: {len(seen)} files")


if __name__ == "__main__":
    main()

