

import requests
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoProcessor

MODEL_PATH = Path.home() / "models/Qwen3.8-27B"

config = AutoConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

def create_image() -> Path:
    # Create a deterministic local test image
    image_path = Path("lab_test_image.png").resolve()

    image = Image.new("RGB", (256, 256), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((56, 56, 200, 200), fill=(220, 35, 35))
    image.save(image_path)
    return image_path

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

# Create a local test image for the lab test

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": str(create_image())},
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# Or, alternatively, download an image from the web

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": download_image(
                url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
                save_path="images/car.jpg",
            )},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

print("=== PROCESSOR ===")
print(type(processor).__name__)
print(type(processor.tokenizer).__name__)
print(type(processor.image_processor).__name__)

print("\n=== RAW CHAT TEMPLATE ===")
print(repr(prompt[:500]))

print("\n=== INPUT TENSORS ===")
for key, value in inputs.items():
    print(key, tuple(value.shape), value.dtype)

print("\n=== TOKEN IDs ===")
ids = inputs["input_ids"][0]
print("Sequence length:", len(ids))
print("First 30 IDs:", ids[:30].tolist())

print("\n=== IMAGE PLACEHOLDERS ===")
image_token_id = config.image_token_id
image_token_count = (ids == image_token_id).sum().item()

print("Image token ID:", image_token_id)
print("Image placeholder count:", image_token_count)
print("Image grid:", inputs["image_grid_thw"].tolist())

print("\n=== IMAGE TENSOR ===")
pixels = inputs["pixel_values"]
print("Shape:", tuple(pixels.shape))
print("Dtype:", pixels.dtype)
print("Range:", pixels.min().item(), pixels.max().item())

print("\n=== DECODED TOKEN PREFIX ===")
print(repr(processor.tokenizer.decode(ids[:30].tolist())))
