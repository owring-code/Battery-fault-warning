from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@pytest.mark.parametrize(
    ('script_path', 'expected_flags'),
    [
        (
            'scripts/build_samples.py',
            ['--mode', '--train-target', '--val-target', '--test-target', '--include-raw', '--vehicle-ids', '--structured-root', '--raw-root'],
        ),
        (
            'scripts/run_lightgbm_recognition.py',
            ['--task', '--samples-root', '--train-samples-root', '--eval-samples-root', '--results-root', '--max-train-samples', '--max-val-samples', '--max-test-samples', '--eval-split'],
        ),
        (
            'scripts/run_threshold_warning_baseline.py',
            ['--samples-root', '--train-samples-root', '--eval-samples-root', '--results-root', '--max-val-samples', '--max-test-samples', '--eval-split'],
        ),
        (
            'scripts/run_sequence_baseline.py',
            ['--samples-root', '--train-samples-root', '--eval-samples-root', '--results-root', '--device', '--max-val-samples', '--eval-split'],
        ),
        (
            'scripts/run_dual_task_model.py',
            ['--samples-root', '--train-samples-root', '--eval-samples-root', '--results-root', '--device', '--ablation', '--max-val-samples', '--eval-split'],
        ),
        (
            'scripts/export_recognition_figures.py',
            ['--results-root', '--figures-root'],
        ),
        (
            'scripts/export_warning_figures.py',
            ['--results-root', '--figures-root'],
        ),
        (
            'scripts/export_ablation_figures.py',
            ['--results-root', '--figures-root'],
        ),
        (
            'scripts/aggregate_round1_tables.py',
            ['--results-root'],
        ),
        (
            'scripts/build_dataset_meta.py',
            ['--structured-root', '--raw-root', '--skip-raw'],
        ),
        (
            'scripts/build_labels.py',
            ['--structured-root', '--raw-root'],
        ),
        (
            'scripts/build_external_validation_samples.py',
            ['--samples-root', '--raw-root', '--vehicle-ids', '--label-kind', '--ignore-quality-flag'],
        ),
    ],
)
def test_script_help_exposes_formal_round_one_cli(script_path: str, expected_flags: list[str]):
    result = subprocess.run(
        [PYTHON, script_path, '--help'],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    help_text = result.stdout
    for flag in expected_flags:
        assert flag in help_text
