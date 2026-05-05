from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_server_requirements_and_runbook_exist_with_core_commands():
    requirements_path = PROJECT_ROOT / 'requirements_server.txt'
    runbook_path = PROJECT_ROOT / 'deliverables' / 'server_runbook.md'
    if not requirements_path.exists() or not runbook_path.exists():
        pytest.skip('requires generated server deliverables, which are not committed')

    requirements_text = requirements_path.read_text(encoding='utf-8')
    runbook_text = runbook_path.read_text(encoding='utf-8-sig')

    for token in ['torch', 'lightgbm', 'scikit-learn', 'pandas', 'matplotlib']:
        assert token in requirements_text

    assert '# 第2轮服务器运行手册' in runbook_text
    assert '## 1. 环境准备' in runbook_text
    assert 'artifacts/round2/samples' in runbook_text
    assert 'results/round2' in runbook_text
    assert r'F:\battery_runs' not in runbook_text

    for command_snippet in [
        'python scripts/build_dataset_meta.py',
        'python scripts/build_samples.py --mode full',
        'python scripts/run_lightgbm_recognition.py --task identification',
        'python scripts/run_lightgbm_recognition.py --task warning',
        'python scripts/run_threshold_warning_baseline.py',
        'python scripts/run_sequence_baseline.py --architecture lstm',
        'python scripts/run_sequence_baseline.py --architecture transformer',
        'python scripts/run_dual_task_model.py',
        '--ablation no_expert_heads',
        '--ablation no_fault_specific_features',
        '--ablation no_warning_task',
        '--ablation no_label_quality_control',
        'python scripts/export_recognition_figures.py',
        'python scripts/export_warning_figures.py',
        'python scripts/export_ablation_figures.py',
        'python scripts/aggregate_round1_tables.py',
    ]:
        assert command_snippet in runbook_text

    for unexpected_snippet in [
        'python scripts/build_samples.py --mode medium',
        '--max-train-samples',
        '--max-val-samples',
        '--max-test-samples',
    ]:
        assert unexpected_snippet not in runbook_text


def test_local_cpu_runbook_uses_f_drive_paths():
    runbook_path = PROJECT_ROOT / 'deliverables' / 'local_cpu_runbook.md'
    if not runbook_path.exists():
        pytest.skip('requires generated local runbook, which is not committed')

    runbook_text = runbook_path.read_text(encoding='utf-8-sig')
    assert '# 本地 CPU 第1轮运行清单' in runbook_text
    assert r'F:\Data_set' in runbook_text
    assert r'F:\RAW_DATA\data' in runbook_text
    assert r'F:\battery_runs\artifacts\samples' in runbook_text
    assert r'F:\battery_runs\results' in runbook_text


def test_server_short_runbook_exists():
    runbook_path = PROJECT_ROOT / 'deliverables' / 'server_short_commands.md'
    if not runbook_path.exists():
        pytest.skip('requires generated server short runbook, which is not committed')

    runbook_text = runbook_path.read_text(encoding='utf-8-sig')
    assert '# 第2轮服务器最短命令' in runbook_text
    assert 'artifacts/round2/samples' in runbook_text
    assert 'results/round2' in runbook_text
    assert 'python scripts/build_samples.py --mode full' in runbook_text
    assert 'python scripts/run_dual_task_model.py' in runbook_text
    assert 'python scripts/export_recognition_figures.py' in runbook_text
    assert '--max-train-samples' not in runbook_text
