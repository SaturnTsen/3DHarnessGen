from pathlib import Path
from io import BytesIO

import requests
from PIL import Image


def download_image(
    url: str,
    save_path: str | Path,
) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)

    print(f"Saved: {save_path}")
    print(f"Image size: {image.size}")

    return image


if __name__ == "__main__":
    image = download_image(
        url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
        save_path="images/car.jpg",
    )