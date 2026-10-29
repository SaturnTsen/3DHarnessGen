"""Keep service names and benchmark results consistent when checkpoints change."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from harness_training.inference import (
    build_harness_command, build_mmsi_command, checkpoint_model_name,
)
from scripts.serve_checkpoint import option_value


@pytest.fixture
def launches(tmp_path):
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    (benchmark / 'pyproject.toml').touch()
    (benchmark / 'pixi.toml').touch()
    checkpoints = []
    for name, step in [('blender_9b', 1612), ('view_27b', 40)]:
        checkpoint = tmp_path / name / 'checkpoints' / f'checkpoint-{step}'
        checkpoint.mkdir(parents=True)
        checkpoints.append(checkpoint)
    return [argparse.Namespace(checkpoint_path=checkpoint, model_name=None, benchmark_root=benchmark,
                               base_url='http://localhost:9000/v1', output_root=None,
                               setting='single-view', context_window=None,
                               max_tokens=8192, dataset='MMSI_Bench')
            for checkpoint in checkpoints]


def test_switch_checkpoint_changes_service_identity_and_both_result_namespaces(launches):
    harness_outputs = []
    mmsi_aliases = []
    for args in launches:
        expected = checkpoint_model_name(args.checkpoint_path)
        _, command = build_harness_command(args, ['--iterations', '0'])
        assert option_value(command, '--model-name') == expected
        assert option_value(command, '--base-url') == args.base_url
        assert command[-2:] == ['--iterations', '0']
        harness_outputs.append(option_value(command, '--output-dir'))
        _, command = build_mmsi_command(args, ['--mode', 'infer', '--api-nproc', '1'])
        config_file = option_value(command, '--config')
        assert config_file is not None
        config = json.loads(Path(config_file).read_text())
        assert config['model'][expected]['model'] == expected
        assert config['model'][expected]['api_base'] == args.base_url
        assert command[-4:] == ['--mode', 'infer', '--api-nproc', '1']
        mmsi_aliases.append(next(iter(config['model'])))
    assert harness_outputs[0] != harness_outputs[1]
    assert mmsi_aliases == ['blender_9b-checkpoint-1612', 'view_27b-checkpoint-40']


def test_fixed_service_alias_still_separates_results_by_checkpoint(launches):
    outputs = []
    aliases = []
    for args in launches:
        args.model_name = 'sft'
        _, command = build_harness_command(args, [])
        assert option_value(command, '--model-name') == 'sft'
        outputs.append(option_value(command, '--output-dir'))
        _, command = build_mmsi_command(args, [])
        path = option_value(command, '--config')
        assert path is not None
        config = json.loads(Path(path).read_text())
        aliases.append(next(iter(config['model'])))
        assert next(iter(config['model'].values()))['model'] == 'sft'
    assert outputs[0] != outputs[1] and aliases[0] != aliases[1]


@pytest.mark.parametrize('setting', ['active-visual', 'full-3d-interaction'])
def test_interaction_budget_is_explicit_and_passed_to_native_runner(launches, setting):
    args = launches[1]
    args.setting = setting
    with pytest.raises(ValueError, match='context-window'):
        build_harness_command(args, [])
    args.context_window = 49152
    _, command = build_harness_command(args, ['--limit', '1'])
    assert setting in command
    assert option_value(command, '--context-window') == '49152'


def test_clients_reject_options_that_could_reuse_another_models_output(launches):
    args = launches[0]
    with pytest.raises(ValueError, match='Launcher-managed'):
        build_harness_command(args, ['--agent=other'])
    with pytest.raises(ValueError, match='Launcher-managed'):
        build_mmsi_command(args, ['--work-dir', 'old-results'])


def test_interaction_can_read_budget_from_saved_snapshot(launches):
    args = launches[1]
    args.setting = 'active-visual'
    snapshot = args.checkpoint_path.parent.parent / 'resolved_config.yaml'
    snapshot.write_text(json.dumps({'template': {'max_length': 49152}}))
    _, command = build_harness_command(args, [])
    assert option_value(command, '--context-window') == '49152'


def test_circular_dataset_and_output_root(launches):
    args = launches[0]
    args.dataset = 'MMSI_Bench_Circular'
    args.output_root = args.benchmark_root / 'custom-output'
    _, command = build_mmsi_command(args, [])
    assert option_value(command, '--work-dir') == str(args.output_root)
    config_file = option_value(command, '--config')
    assert config_file is not None
    config = json.loads(Path(config_file).read_text())
    assert config['data'][args.dataset]['class'] == 'MMSIBenchCircular'


@pytest.mark.parametrize('script', ['run_3dharness_checkpoint.py', 'eval_mmsi_checkpoint.py'])
def test_client_scripts_dry_run_without_starting_benchmarks(launches, script):
    args = launches[1]
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([
        sys.executable, str(root / 'scripts' / script), '--checkpoint-path', str(args.checkpoint_path),
        '--benchmark-root', str(args.benchmark_root), '--dry-run',
        '--limit', '1', '--', '--dry-run',
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'view_27b-checkpoint-40' in result.stdout
    rendered = json.loads(result.stdout.split('\n', 1)[1])
    assert rendered['cwd'] == str(args.benchmark_root)
    assert rendered['argv'][0] == 'pixi'
    assert rendered['argv'][-3:] == ['--limit', '1', '--dry-run']
    assert '--' not in rendered['argv']
