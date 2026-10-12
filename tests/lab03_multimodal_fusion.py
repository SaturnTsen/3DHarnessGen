# %% Processor and config
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_PATH = Path.home() / "models/Qwen3.8-27B"
IMAGE_PATH = Path("images/car.jpg")

config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
vision_cfg = config.vision_config
text_cfg = config.text_config
image_processor = processor.image_processor

image = Image.open(IMAGE_PATH).convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

print("=== CONFIG ===")
print("architecture:", config.architectures)
print("image_token_id:", config.image_token_id)
print("vision patch size:", vision_cfg.patch_size)
print("vision temporal patch size:", vision_cfg.temporal_patch_size)
print("vision spatial merge size:", vision_cfg.spatial_merge_size)
print("vision hidden size:", vision_cfg.hidden_size)
print("vision output hidden size:", vision_cfg.out_hidden_size)
print("language hidden size:", text_cfg.hidden_size)
print("processor resize settings:", image_processor.size)
print("processor mean/std:", image_processor.image_mean, image_processor.image_std)

print("\n=== PROCESSOR ===")
print("raw image size (W, H):", image.size)
print("raw template image-pad count:", prompt.count("<|image_pad|>"))
print("raw template prefix:", repr(prompt[:250]))
for name, tensor in inputs.items():
    print(name, tuple(tensor.shape), tensor.dtype)

ids = inputs["input_ids"]
image_mask = ids == config.image_token_id
image_positions = torch.where(image_mask[0])[0]
t, h, w = inputs["image_grid_thw"][0].tolist()
patch_count = t * h * w
patch_width = vision_cfg.in_channels * vision_cfg.temporal_patch_size * vision_cfg.patch_size**2
merged_count = patch_count // vision_cfg.spatial_merge_size**2

assert tuple(inputs["pixel_values"].shape) == (patch_count, patch_width)
assert image_positions.numel() == merged_count

print("\n=== PATCH / PLACEHOLDER ALIGNMENT ===")
print("grid (T, H, W):", (t, h, w))
print("patch count:", patch_count)
print("flattened patch width:", patch_width)
print("merged visual token count:", merged_count)
print("image-placeholder count:", image_positions.numel())
print("image-placeholder positions:", (image_positions[0].item(), image_positions[-1].item()))
print("first 8 patch floats:", inputs["pixel_values"][0, :8].tolist())
print("token IDs around image start:", ids[0, image_positions[0] - 2:image_positions[0] + 3].tolist())
modality_ids, modality_counts = torch.unique(inputs["mm_token_type_ids"], return_counts=True)
print("modality counts:", dict(zip(modality_ids.tolist(), modality_counts.tolist())))

# %% Load model and attach read-only hooks
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
).eval()

captured = {}


def patch_hook(module, args, output):
    captured["patch_shape"] = tuple(output.shape)
    captured["patch_sample"] = output[0, :8].detach().float().cpu()


def vision_hook(module, args, output):
    captured["vision_shape"] = tuple(output.last_hidden_state.shape)
    captured["vision_sample"] = output.last_hidden_state[0, :8].detach().float().cpu()
    captured["merged"] = output.pooler_output.detach().clone()


def language_pre_hook(module, args, kwargs):
    captured["joint"] = kwargs["inputs_embeds"].detach().clone()
    captured["position_ids"] = kwargs["position_ids"].detach().cpu()


handles = [
    model.model.visual.patch_embed.register_forward_hook(patch_hook),
    model.model.visual.register_forward_hook(vision_hook),
    model.model.language_model.register_forward_pre_hook(
        language_pre_hook, with_kwargs=True
    ),
]

# %% One actual model forward
model_inputs = inputs.to(model.get_input_embeddings().weight.device)
with torch.inference_mode():
    outputs = model(
        **model_inputs,
        logits_to_keep=1,
        use_cache=False,
        return_dict=True,
    )

for handle in handles:
    handle.remove()

# %% Verify multimodal fusion
joint = captured["joint"]
merged = captured["merged"].to(device=joint.device, dtype=joint.dtype)
mask = image_mask.to(joint.device)

with torch.inference_mode():
    original_token_embeddings = model.get_input_embeddings()(
        model_inputs["input_ids"]
    ).to(joint.device)

assert torch.equal(joint[mask], merged)
assert torch.equal(joint[~mask], original_token_embeddings[~mask])

print("\n=== VISION STAGES ===")
print("patch embedding:", captured["patch_shape"])
print("patch embedding first 8 floats:", captured["patch_sample"].tolist())
print("vision transformer:", captured["vision_shape"])
print("vision output first 8 floats:", captured["vision_sample"].tolist())
print("spatial merger:", tuple(merged.shape))
print("merged feature first 8 floats:", merged[0, :8].float().cpu().tolist())

print("\n=== MULTIMODAL FUSION ===")
print("original token embeddings:", tuple(original_token_embeddings.shape))
print("joint embeddings:", tuple(joint.shape))
print("visual positions match merged features:", torch.equal(joint[mask], merged))
print("text positions unchanged:", torch.equal(joint[~mask], original_token_embeddings[~mask]))
print("visual feature vs placeholder embedding max difference:",
      (joint[mask] - original_token_embeddings[mask]).float().abs().max().item())

print("\n=== POSITION INPUT ===")
position_ids = captured["position_ids"]
print("position_ids:", tuple(position_ids.shape))
print("first 5 visual token positions (columns T/H/W plus text position):")
print(position_ids[:, 0, image_positions[:5]].T.tolist())
print("next-token logits:", tuple(outputs.logits.shape))
