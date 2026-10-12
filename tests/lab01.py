import torch
from pathlib import Path
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_PATH = Path.home() / "models/Qwen3.8-27B"

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is a Blender MCP server? Answer briefly.",
            }
        ],
    },
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

inputs = inputs.to(model.device)

print("\n=== CHAT TEMPLATE ===")
print(repr(prompt))

print("\n=== MODEL INPUT ===")
for name, tensor in inputs.items():
    print(name, tensor.shape, tensor.dtype, tensor.device)

print("\n=== DEVICE MAP ===")
print(model.hf_device_map)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=6400,
        do_sample=False,
    )

input_length = inputs["input_ids"].shape[1]
generated_ids = output_ids[:, input_length:]

print("\n=== OUTPUT SHAPES ===")
print("input_ids:", inputs["input_ids"].shape)
print("output_ids:", output_ids.shape)
print("generated_ids:", generated_ids.shape)

print("\n=== GENERATED TOKENS ===")
print(generated_ids[0].tolist())

print("\n=== DECODED RESPONSE ===")
print(
    processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
)
