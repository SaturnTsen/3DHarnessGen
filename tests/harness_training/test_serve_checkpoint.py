"""Validate checkpoint serving without allocating base model weights."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.serve_checkpoint import (
    build_command, extract_fsdp_adapter, option_value, read_adapter_config, validate_adapter,
)


@pytest.fixture
def serving_checkpoint(tmp_path):
    import torch
    from peft import LoraConfig
    from safetensors.torch import save_file

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 32, "num_hidden_layers": 4},
    }))
    module = "model.language_model.layers.3.self_attn.q_proj"
    save_file({module + ".weight": torch.zeros(64, 32)}, base / "model.safetensors")
    checkpoint = tmp_path / "training/checkpoints/checkpoint-7"
    checkpoint.mkdir(parents=True)
    LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj"],
               base_model_name_or_path=str(base)).save_pretrained(str(checkpoint))
    save_file({
        f"base_model.model.{module}.lora_A.weight": torch.ones(16, 32),
        f"base_model.model.{module}.lora_B.weight": torch.ones(64, 16),
    }, checkpoint / "adapter_model.safetensors")
    (checkpoint / "tokenizer_config.json").write_text('{}')
    (checkpoint / "chat_template.jinja").write_text('{{ messages }}')
    (checkpoint / "processor_config.json").write_text(json.dumps({
        "image_processor": {"size": {"shortest_edge": 65536, "longest_edge": 262144}},
    }))
    return base, checkpoint


def arguments(checkpoint, base=None):
    return argparse.Namespace(checkpoint_path=checkpoint, base_model_path=base, adapter_name=None)


def digest_files(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


def test_standard_adapter_base_override_and_passthrough(serving_checkpoint):
    base, checkpoint = serving_checkpoint
    config_path = checkpoint / 'adapter_config.json'
    config = json.loads(config_path.read_text())
    config['base_model_name_or_path'] = '/unavailable/previous/server/model'
    config_path.write_text(json.dumps(config))
    original = digest_files(checkpoint)
    options = ['--port', '18096', '--max-model-len=73728', '--tensor-parallel-size', '1',
               '--data-parallel-size', '3', '--default-chat-template-kwargs', '{"enable_thinking":false}']
    command, name = build_command(arguments(checkpoint, base), options)
    assert command[:3] == ['vllm', 'serve', str(base)]
    assert command[-len(options):] == options
    assert name == 'training-checkpoint-7'
    modules = option_value(command, '--lora-modules')
    assert modules is not None
    assert json.loads(modules) == {
        'name': name, 'path': str(checkpoint), 'base_model_name': str(base),
    }
    assert option_value(command, '--max-lora-rank') == '16'
    tokenizer_option = option_value(command, '--tokenizer')
    assert tokenizer_option is not None
    tokenizer = Path(tokenizer_option)
    assert (tokenizer / 'config.json').read_bytes() == (base / 'config.json').read_bytes()
    assert (tokenizer / 'tokenizer_config.json').read_bytes() == (checkpoint / 'tokenizer_config.json').read_bytes()
    assert option_value(command, '--chat-template') == str(checkpoint / 'chat_template.jinja')
    processor_option = option_value(command, '--mm-processor-kwargs')
    assert processor_option is not None
    assert json.loads(processor_option)['size']['shortest_edge'] == 65536
    assert digest_files(checkpoint) == original


def test_explicit_tokenizer_template_image_settings_and_rank_take_precedence(serving_checkpoint):
    base, checkpoint = serving_checkpoint
    options = ['--tokenizer=custom-tokenizer', '--chat-template', 'custom.jinja',
               '--mm-processor-kwargs', '{"min_pixels":4096}', '--max-lora-rank', '32']
    command, _ = build_command(arguments(checkpoint), options)
    assert command[-len(options):] == options
    assert '--tokenizer' not in command
    assert command.count('--chat-template') == command.count('--max-lora-rank') == 1
    with pytest.raises(ValueError, match='at least 16'):
        build_command(arguments(checkpoint), ['--max-lora-rank=8'])
    with pytest.raises(ValueError, match='launcher owns'):
        build_command(arguments(checkpoint), ['--lora-modules', 'wrong=somewhere'])


def test_rejects_incompatible_base_dimensions(serving_checkpoint, tmp_path):
    import torch
    from safetensors.torch import save_file

    base, checkpoint = serving_checkpoint
    save_file({'model.language_model.layers.3.self_attn.q_proj.weight': torch.zeros(64, 48)},
              base / 'model.safetensors')
    with pytest.raises(ValueError, match='Incompatible adapter/base shape'):
        build_command(arguments(checkpoint), [])


@pytest.mark.parametrize('named_adapter', [False, True])
def test_fsdp_extracts_only_lora_and_preserves_checkpoint(serving_checkpoint, tmp_path, named_adapter):
    import torch
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict_saver import save

    base, checkpoint = serving_checkpoint
    expected = load_file(checkpoint / 'adapter_model.safetensors')
    stored = {name.replace('.weight', '.custom.weight') if named_adapter else name: value
              for name, value in expected.items()}
    source = checkpoint / 'pytorch_model_fsdp_0'
    save({'model': {**stored, 'frozen_base.weight': torch.ones(1024, 32)}},
         checkpoint_id=source, no_dist=True)
    (checkpoint / 'adapter_model.safetensors').unlink()
    config = json.loads((checkpoint / 'adapter_config.json').read_text())
    original = digest_files(checkpoint)
    output = extract_fsdp_adapter(checkpoint, config, tmp_path / 'cache')
    actual = load_file(output / 'adapter_model.safetensors')
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], value) for name, value in expected.items())
    assert validate_adapter(base, output, config) == 16
    assert digest_files(checkpoint) == original
    assert extract_fsdp_adapter(checkpoint, config, tmp_path / 'cache') == output


def test_fsdp_config_can_come_from_training_snapshot(serving_checkpoint):
    import torch
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict_saver import save
    from harness_training.config import load_config

    base, checkpoint = serving_checkpoint
    config = load_config(Path(__file__).resolve().parents[2] / 'configs/sft_blender_9b_fsdp.yaml')
    config.model.path = base
    snapshot = config.model_dump(mode='json')
    (checkpoint.parent.parent / 'resolved_config.yaml').write_text(json.dumps(snapshot))
    expected = load_file(checkpoint / 'adapter_model.safetensors')
    save({'model': expected}, checkpoint_id=checkpoint / 'pytorch_model_fsdp_0', no_dist=True)
    (checkpoint / 'adapter_config.json').unlink()
    (checkpoint / 'adapter_model.safetensors').unlink()
    reconstructed = read_adapter_config(checkpoint, snapshot)
    assert reconstructed['r'] == config.model.lora.rank
    assert reconstructed['lora_alpha'] == config.model.lora.alpha
    assert reconstructed['base_model_name_or_path'] == str(config.model.path)
    command, _ = build_command(arguments(checkpoint, base), [])
    modules = option_value(command, '--lora-modules')
    assert modules is not None
    exported = load_file(Path(json.loads(modules)['path']) / 'adapter_model.safetensors')
    assert all(torch.equal(exported[name], value) for name, value in expected.items())


def test_cli_dry_run_does_not_expose_api_key(serving_checkpoint):
    _, checkpoint = serving_checkpoint
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([
        sys.executable, str(root / 'scripts/serve_checkpoint.py'),
        '--checkpoint-path', str(checkpoint), '--dry-run', '--api-key', 'dummy-test-key',
        '--port', '18096',
    ], capture_output=True, text=True, cwd=root, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'dummy-test-key' not in result.stdout
    command = json.loads(result.stdout.split('\n', 1)[1])
    assert option_value(command, '--api-key') == '<redacted>'
    assert option_value(command, '--port') == '18096'
