"""Serve a post-training LoRA checkpoint using the project's vLLM installation.

Example:
    pixi run python scripts/serve_checkpoint.py \
        --checkpoint-path outputs/post_training/Blender/run/checkpoints/checkpoint-1612 \
        --base-model-path ~/models/Qwen3.5-9B \
        --max-model-len 73728 --port 8000

Arguments not owned by this launcher are passed unchanged to ``vllm serve``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from harness_training.inference import checkpoint_model_name, checkpoint_snapshot, parse_passthrough


def option_value(arguments: list[str], flag: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == flag:
            if index + 1 == len(arguments):
                raise ValueError(f"{flag} requires a value")
            return arguments[index + 1]
        if argument.startswith(flag + "="):
            return argument.split("=", 1)[1]
    return None


def has_option(arguments: list[str], flag: str) -> bool:
    return any(argument == flag or argument.startswith(flag + "=") for argument in arguments)


def read_adapter_config(checkpoint: Path, snapshot: dict) -> dict:
    source = checkpoint / "adapter_config.json"
    if source.is_file():
        return json.loads(source.read_text())
    from peft import LoraConfig, LoraRuntimeConfig

    model = snapshot["model"]
    if model["tuning"] != "lora":
        raise ValueError("Only LoRA/QLoRA checkpoints are supported")
    lora = model["lora"]
    options = dict(lora["backend"])
    options["runtime_config"] = LoraRuntimeConfig(**options["runtime_config"])
    options.update(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                   target_modules=lora["target_modules"], task_type=lora["task_type"], bias=lora["bias"])
    config = LoraConfig(**options).to_dict()
    # PEFT's serializer handles sets and enums; use it when exporting the adapter.
    config["base_model_name_or_path"] = model["path"]
    return config


def normalize_lora_name(name: str) -> str:
    return re.sub(r"(\.lora_[AB])\.[^.]+\.weight$", r"\1.weight", name)


def prepare_tokenizer(checkpoint: Path, base: Path, cache_root: Path) -> Path:
    """Pair the saved tokenizer with a model config required by vLLM probing."""
    sources = [base / "config.json"]
    sources.extend(path for path in sorted(checkpoint.iterdir()) if path.is_file() and (
        path.suffix in (".json", ".jinja", ".model", ".txt")
        and path.name not in {"adapter_config.json", "trainer_state.json", "config.json"}
    ))
    fingerprint = hashlib.sha256()
    for path in sources:
        fingerprint.update(path.name.encode() + path.read_bytes())
    destination = cache_root / ("tokenizer-" + fingerprint.hexdigest())
    if not destination.is_dir():
        cache_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache_root, prefix="tokenizer-") as temporary:
            output = Path(temporary)
            for path in sources:
                shutil.copyfile(path, output / path.name)
            output.rename(destination)
    return destination


def extract_fsdp_adapter(checkpoint: Path, config: dict, cache_root: Path) -> Path:
    import torch
    from peft import LoraConfig
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from torch.distributed.checkpoint.state_dict_loader import load

    source = checkpoint / "pytorch_model_fsdp_0"
    fingerprint = hashlib.sha256(str(checkpoint).encode() + json.dumps(
        config, sort_keys=True, default=lambda value: sorted(value),
    ).encode())
    for path in sorted(source.iterdir()):
        info = path.stat()
        fingerprint.update(f"{path.name}:{info.st_size}:{info.st_mtime_ns}".encode())
    destination = cache_root / fingerprint.hexdigest()
    if (destination / "adapter_model.safetensors").is_file():
        return destination
    metadata = FileSystemReader(source).read_metadata()
    weights = {}
    for key, value in metadata.state_dict_metadata.items():
        if not key.startswith("model."):
            continue
        if not isinstance(value, TensorStorageMetadata):
            raise ValueError(f"Unsupported FSDP metadata: {key}")
        name = normalize_lora_name(key.removeprefix("model."))
        if not name.endswith((".lora_A.weight", ".lora_B.weight")):
            continue
        weights[key.removeprefix("model.")] = torch.empty(value.size, dtype=value.properties.dtype)
    if not weights:
        raise ValueError("FSDP checkpoint contains no LoRA A/B tensors")
    # Allocate and read only adapter tensors, regardless of frozen base/optimizer size.
    load({"model": weights}, checkpoint_id=source, no_dist=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_root, prefix="extract-") as temporary:
        output = Path(temporary)
        LoraConfig(**config).save_pretrained(str(output))
        normalized = {normalize_lora_name(name): tensor.contiguous() for name, tensor in weights.items()}
        if len(normalized) != len(weights):
            raise ValueError("FSDP checkpoint contains multiple adapters; select a single-adapter checkpoint")
        save_file(normalized, output / "adapter_model.safetensors")
        output.rename(destination)
    return destination


def validate_adapter(base: Path, adapter: Path, config: dict) -> int:
    """Compare adapter dimensions to base tensor headers without loading base weights."""
    from safetensors import safe_open
    from transformers import AutoConfig

    architecture = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if architecture.model_type != "qwen3_5":
        raise ValueError(f"Expected qwen3_5 multimodal base model, got {architecture.model_type}")
    if config["peft_type"] != "LORA" or config["bias"] != "none" or config.get("modules_to_save"):
        raise ValueError("This launcher supports LoRA A/B weights with bias=none and no modules_to_save")
    if config.get("use_dora") or config.get("alora_invocation_tokens") or config.get("use_bdlora"):
        raise ValueError("DoRA, aLoRA and BDLora adapters are not supported by this launcher")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("Per-module LoRA rank/alpha patterns are not supported by vLLM's adapter loader")
    index = base / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    maximum_rank = 0
    with ExitStack() as stack:
        shards = {}
        if weight_map is None:
            handle = stack.enter_context(safe_open(base / "model.safetensors", framework="pt", device="cpu"))
            shards["model.safetensors"] = handle
            weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
        weights = stack.enter_context(safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu"))
        names = set(weights.keys())
        if not names:
            raise ValueError("Adapter weight file is empty")
        for name in sorted(names):
            match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", name)
            if match is None:
                raise ValueError(f"Unsupported adapter tensor: {name}")
            module, side = match.groups()
            counterpart = name.replace(f".lora_{side}.weight", f".lora_{'B' if side == 'A' else 'A'}.weight")
            if counterpart not in names:
                raise ValueError(f"Missing LoRA A/B counterpart for {name}")
            base_name = module + ".weight"
            if base_name not in weight_map:
                raise ValueError(f"Adapter module is absent from base model: {base_name}")
            shard_name = weight_map[base_name]
            if shard_name not in shards:
                shards[shard_name] = stack.enter_context(safe_open(base / shard_name, framework="pt", device="cpu"))
            shape = shards[shard_name].get_slice(base_name).get_shape()
            current = weights.get_slice(name).get_shape()
            other = weights.get_slice(counterpart).get_shape()
            if len(shape) != 2 or len(current) != 2 or len(other) != 2:
                raise ValueError(f"Expected two-dimensional LoRA/base tensors: {name}")
            expected = [current[0], shape[1]] if side == "A" else [shape[0], current[1]]
            rank = current[0] if side == "A" else current[1]
            other_rank = other[1] if side == "A" else other[0]
            if current != expected or rank != other_rank:
                raise ValueError(f"Incompatible adapter/base shape for {name}: adapter={current}, base={shape}")
            if rank != config["r"]:
                raise ValueError(f"Adapter tensor rank does not match adapter_config.json: {name}")
            maximum_rank = max(maximum_rank, rank)
    return maximum_rank


def build_command(args: argparse.Namespace, passthrough: list[str]) -> tuple[list[str], str]:
    checkpoint = args.checkpoint_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    snapshot = checkpoint_snapshot(checkpoint)
    config = read_adapter_config(checkpoint, snapshot)
    recorded_base = config.get("base_model_name_or_path") or snapshot.get("model", {}).get("path")
    if args.base_model_path is None and not recorded_base:
        raise ValueError("Cannot determine base model; provide --base-model-path")
    base = (args.base_model_path or Path(recorded_base)).expanduser().resolve()
    if not (base / "config.json").is_file():
        raise ValueError(f"Base config.json is unavailable at {base}; provide --base-model-path with model files")
    if (checkpoint / "pytorch_model_fsdp_0/.metadata").is_file():
        cache_root = Path(__file__).resolve().parents[1] / "temp/vllm_adapters"
        adapter = extract_fsdp_adapter(checkpoint, config, cache_root)
    else:
        adapter = checkpoint
    if not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError("Checkpoint must contain adapter_model.safetensors or FSDP shards")
    rank = validate_adapter(base, adapter, config)
    if any(has_option(passthrough, flag) for flag in ("--lora-modules", "--no-enable-lora", "--model")):
        raise ValueError("The launcher owns --model, --lora-modules and --enable-lora")
    supplied_rank = option_value(passthrough, "--max-lora-rank")
    if supplied_rank is not None and int(supplied_rank) < rank:
        raise ValueError(f"--max-lora-rank must be at least {rank}")
    name = checkpoint_model_name(checkpoint, args.adapter_name)
    command = ["vllm", "serve", str(base), "--enable-lora", "--lora-modules",
               json.dumps({"name": name, "path": str(adapter), "base_model_name": str(base)})]
    if supplied_rank is None:
        supported_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
        selected_rank = next((value for value in supported_ranks if value >= rank), None)
        if selected_rank is None:
            raise ValueError(f"LoRA rank {rank} is unsupported by vLLM 0.30.0")
        command += ["--max-lora-rank", str(selected_rank)]
    if not has_option(passthrough, "--tokenizer") and (checkpoint / "tokenizer_config.json").is_file():
        tokenizer = prepare_tokenizer(checkpoint, base, Path(__file__).resolve().parents[1] / "temp/vllm_adapters")
        command += ["--tokenizer", str(tokenizer)]
    if not has_option(passthrough, "--chat-template") and (checkpoint / "chat_template.jinja").is_file():
        command += ["--chat-template", str(checkpoint / "chat_template.jinja")]
    processor_path = checkpoint / "processor_config.json"
    if not has_option(passthrough, "--mm-processor-kwargs"):
        if processor_path.is_file():
            image_processor = json.loads(processor_path.read_text()).get("image_processor")
            if image_processor is not None:
                command += ["--mm-processor-kwargs", json.dumps({"size": image_processor["size"]})]
        elif "template" in snapshot:
            template = snapshot["template"]
            command += ["--mm-processor-kwargs", json.dumps({
                "min_pixels": template["min_image_pixels"], "max_pixels": template["max_image_pixels"],
            })]
    return command + passthrough, name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--adapter-name", "--model-name", dest="adapter_name",
                        help="Request this name in the API's model field")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print argv without starting vLLM")
    args, passthrough = parse_passthrough(parser)
    try:
        command, name = build_command(args, passthrough)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Client request model: {name}", flush=True)
    if args.dry_run:
        display = list(command)
        redact = False
        for index, argument in enumerate(display):
            if redact and not argument.startswith("--"):
                display[index] = "<redacted>"
                continue
            redact = False
            if argument == "--api-key":
                redact = True
            if argument.startswith("--api-key="):
                display[index] = "--api-key=<redacted>"
        print(json.dumps(display, ensure_ascii=False, indent=2))
        return
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
