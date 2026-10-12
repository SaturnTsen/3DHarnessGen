from pathlib import Path


class DataError(ValueError):
    pass


def validate_image(path: Path, decode: bool) -> str | None:
    if not path.is_file():
        return f"missing image: {path}"
    if decode:
        from PIL import Image

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            return f"unreadable image: {path}: {exc}"
    return None
