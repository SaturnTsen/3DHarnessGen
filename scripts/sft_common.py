"""Shared, portable Qwen multimodal trajectory format and image storage."""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import urllib.parse
import zlib
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DATA_URI_RE = re.compile(
    r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n]+(?:\[REDACTED_[A-Z0-9_]+\][A-Za-z0-9+/=\r\n]*)*)")


class ImageStore:
    def __init__(self, raw_root: Path, output_stage: Path | None):
        self.raw_root = raw_root
        self.output_stage = output_stage
        self.digest_paths: dict[str, str] = {}
        self.warnings: list[str] = []
        self.blobs: dict[str, Path] | None = None

    def _blob_index(self) -> dict[str, Path]:
        if self.blobs is None:
            self.blobs = {}
            for path in self.raw_root.rglob("blobs/*"):
                if path.is_file():
                    self.blobs[path.name] = path
        return self.blobs

    @staticmethod
    def _valid_png(data: bytes) -> bool:
        """Reject truncated/redacted PNGs even when their base64 decodes."""
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return False
        offset = 8
        while offset + 12 <= len(data):
            size = int.from_bytes(data[offset:offset + 4], "big")
            end = offset + 8 + size
            if end + 4 > len(data):
                return False
            if zlib.crc32(data[offset + 4:end]) != int.from_bytes(data[end:end + 4], "big"):
                return False
            if data[offset + 4:offset + 8] == b"IEND":
                return size == 0
            offset = end + 4
        return False

    @staticmethod
    def _extension(data: bytes, mime: str | None) -> str | None:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if data.startswith((b"RIFF",)) and data[8:12] == b"WEBP":
            return ".webp"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if data.startswith(b"BM"):
            return ".bmp"
        if mime:
            ext = mimetypes.guess_extension(mime.split(";", 1)[0])
            if ext in IMAGE_SUFFIXES:
                return ext
        return None

    def _bytes_from_value(self, value: Any, mime: str | None = None) -> tuple[bytes, str | None] | None:
        if isinstance(value, bytes):
            return value, mime
        if isinstance(value, dict):
            mime = value.get("mimeType") or value.get("mime_type") or value.get("media_type") or mime
            for key in ("url", "data", "base64", "path", "file", "blob", "id"):
                if key in value:
                    found = self._bytes_from_value(value[key], mime)
                    if found:
                        return found
            return None
        if not isinstance(value, str):
            return None
        match = DATA_URI_RE.search(value)
        if match:
            if "[REDACTED_" in match.group(2):
                return None
            try:
                return base64.b64decode(match.group(2), validate=False), match.group(1)
            except (ValueError, base64.binascii.Error):
                return None
        # Some providers put raw base64 in a field named data or image.
        compact = value.strip()
        if compact.startswith("blobref:"):
            if ";" not in compact:
                return None
            mime, compact = compact[len("blobref:"):].split(";", 1)
        if compact.startswith("blob:"):
            compact = compact[5:]
        if len(compact) > 256 and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", compact):
            try:
                decoded = base64.b64decode(compact, validate=False)
                if self._extension(decoded, mime):
                    return decoded, mime
            except (ValueError, base64.binascii.Error):
                pass
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme == "file":
            candidate = Path(urllib.parse.unquote(parsed.path))
        else:
            candidate = Path(value)
        candidates = [candidate] if candidate.is_absolute() else [self.raw_root / candidate]
        # Recover paths captured before the run artifacts were relocated.
        for marker in ("/log_artifacts/", "/iteration_renders/", "/session/"):
            if marker in candidate.as_posix():
                candidates.append(self.raw_root / marker.strip("/") / candidate.as_posix().split(marker, 1)[1])
        # Kimi stores image payloads as extensionless content-addressed blobs.
        if re.fullmatch(r"[a-fA-F0-9]{40,80}", compact):
            blob = self._blob_index().get(compact)
            if blob:
                candidates.append(blob)
        for candidate in candidates:
            try:
                data = candidate.read_bytes()
            except (OSError, IsADirectoryError):
                continue
            if self._extension(data, mime):
                return data, mime
        return None

    def save(self, value: Any, mime: str | None = None) -> str | None:
        found = self._bytes_from_value(value, mime)
        if not found:
            self.warnings.append("unreadable image payload")
            return None
        data, detected_mime = found
        extension = self._extension(data, detected_mime)
        if not extension:
            self.warnings.append("unsupported image format")
            return None
        if extension == ".png" and not self._valid_png(data):
            self.warnings.append("corrupt or truncated PNG payload")
            return None
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.digest_paths:
            return self.digest_paths[digest]
        # Content-addressed names stay valid while a new JSONL is published.
        name = f"{digest}{extension}"
        relative = f"images/{name}"
        self.digest_paths[digest] = relative
        if self.output_stage is not None:
            self.output_stage.mkdir(parents=True, exist_ok=True)
            (self.output_stage / name).write_bytes(data)
        return relative


def read_prompt(raw_root: Path, override: Path | None = None) -> tuple[str, str]:
    """The supplied task prompt must be the first content in every example."""
    # Full3DInteraction archives the supplied task skill as SKILL.md instead
    # of prompt.txt; it is source data here, not exporter instructions.
    candidates = [override] if override else [raw_root / "prompt.txt", raw_root.parent / "prompt.txt",
                                             raw_root / "SKILL.md"]
    for path in candidates:
        if path.is_file():
            prompt = path.read_text(encoding="utf-8").rstrip()
            if prompt:
                return prompt, str(path)
    raise ValueError(f"missing or empty task prompt under {raw_root}; supply --prompt-file")


def make_row(messages: list[dict[str, Any]], prompt: str,
             metadata: dict[str, Any]) -> dict[str, Any]:
    """Use Qwen roles, structured content and one ordered image per placeholder.

    Paths in `images` are relative to the JSONL. Repeated image occurrences
    remain in that list, although identical bytes are stored only once.
    """
    system = [{"type": "text", "text": prompt}]
    normalized = []
    image_paths = []
    for original in messages:
        message = dict(original)
        role = message["role"]
        content = message.get("content") or []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if role in {"system", "developer"}:
            # Qwen rejects system messages anywhere except index zero.
            for block in content:
                if block["type"] != "text":
                    raise ValueError("system/developer messages cannot contain images")
                if block.get("text") and block["text"].strip() != prompt.strip():
                    system.append({"type": "text", "text": "\n\n" + block["text"]})
            continue
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role}")
        blocks = []
        for block in content:
            if block["type"] == "image":
                path = block.get("image")
                if not isinstance(path, str) or not path.startswith("images/"):
                    raise ValueError(f"invalid image reference: {path!r}")
                image_paths.append(path)
                blocks.append({"type": "image"})
            elif block["type"] == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})
            else:
                raise ValueError(f"unsupported content block: {block['type']}")
        message["content"] = blocks
        normalized.append(message)
    if not normalized or normalized[0]["role"] != "user":
        raise ValueError("trajectory has no initial user request")
    if not any(m["role"] == "assistant" for m in normalized):
        raise ValueError("trajectory has no assistant training target")
    return {"messages": [{"role": "system", "content": system}, *normalized],
            "images": image_paths, "metadata": metadata}


def write_trajectory(output_dir: Path, row: dict[str, Any], stage: Path) -> None:
    """Publish exactly one row after every referenced image is available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    referenced = set(row["images"])
    for relative in referenced:
        name = Path(relative).name
        source = stage / name
        if not source.is_file():
            raise ValueError(f"missing staged image: {relative}")
    tmp_jsonl = output_dir / ".trajectory.jsonl.tmp"
    tmp_jsonl.write_text(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    for relative in referenced:
        name = Path(relative).name
        os.replace(stage / name, image_dir / name)
    os.replace(tmp_jsonl, output_dir / "trajectory.jsonl")
    # Both old exporters owned this filename and the numbered image files.
    (output_dir / "trajectories.jsonl").unlink(missing_ok=True)
    for old in image_dir.iterdir():
        generated = re.fullmatch(r"(?:\d{6,}|[a-f0-9]{64})\.(png|jpg|jpeg|webp|gif|bmp)", old.name)
        if old.is_file() and generated and f"images/{old.name}" not in referenced:
            old.unlink()
