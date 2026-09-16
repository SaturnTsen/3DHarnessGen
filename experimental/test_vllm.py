from openai import OpenAI
import os

VLLM_API_KEY = os.getenv("VLLM_API_KEY")  # TODO: Fill in your VLLM_API_KEY.
BASE_URL = "http://localhost:8000/v1"

if not VLLM_API_KEY:
    raise ValueError("Fill in VLLM_API_KEY first.")

client = OpenAI(base_url=BASE_URL, api_key=VLLM_API_KEY)
model = client.models.list().data[0].id
messages = []

print(f"Model: {model} (type /exit to quit)")

while True:
    prompt = input("You: ")
    if prompt == "/exit":
        break
    if not prompt.strip():
        continue

    messages.append({"role": "user", "content": prompt})
    parts = ["<think>\n"]

    print("Assistant: <think>", flush=True)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            print(piece, end="", flush=True)
            parts.append(piece)

    print()
    messages.append({"role": "assistant", "content": "".join(parts)})