from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _write_prediction_bundle(bundle_dir: Path, model_name: str, task_type: str) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for sample_id, vehicle_id, fault_type, y_true, y_pred, y_score, threshold in [
        ('s1', 'v1', 'sd', 1, 1, 0.9, 0.5),
        ('s1', 'v1', 'samp', 0, 0, 0.1, 0.5),
        ('s1', 'v1', 'ins', 0, 0, 0.1, 0.5),
        ('s1', 'v1', 'isc', 1, 1, 0.8, 0.5),
        ('s1', 'v1', 'conn', 0, 0, 0.1, 0.5),
        ('s2', 'v1', 'sd', 0, 0, 0.2, 0.5),
        ('s2', 'v1', 'samp', 1, 0, 0.2, 0.5),
        ('s2', 'v1', 'ins', 0, 0, 0.1, 0.5),
        ('s2', 'v1', 'isc', 0, 0, 0.2, 0.5),
        ('s2', 'v1', 'conn', 0, 0, 0.1, 0.5),
        ('s3', 'v2', 'sd', 0, 0, 0.1, 0.5),
        ('s3', 'v2', 'samp', 0, 0, 0.1, 0.5),
        ('s3', 'v2', 'ins', 1, 0, 0.4, 0.6),
        ('s3', 'v2', 'isc', 0, 0, 0.1, 0.5),
        ('s3', 'v2', 'conn', 0, 0, 0.1, 0.5),
    ]:
        predictions.append(
            {
                'sample_id': sample_id,
                'vehicle_id': vehicle_id,
                'source_dataset': 'structured_dataset',
                'model_name': model_name,
                'task_type': task_type,
                'fault_type': fault_type,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_score': y_score,
                'threshold': threshold,
            }
        )
    pd.DataFrame(predictions).to_csv(bundle_dir / 'predictions.csv', index=False, encoding='utf-8-sig')


def test_export_plan_supplements_generates_tables_and_figures(tmp_path: Path):
    results_root = tmp_path / 'results'
    samples_root = tmp_path / 'samples'

    samples_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {'sample_id': 's1', 'vehicle_id': 'v1', 'end_time': 1, 'future_first_fault': '', 'lead_time_sec': ''},
            {'sample_id': 's2', 'vehicle_id': 'v1', 'end_time': 2, 'future_first_fault': 'samp', 'lead_time_sec': 30},
            {'sample_id': 's3', 'vehicle_id': 'v2', 'end_time': 3, 'future_first_fault': 'ins', 'lead_time_sec': 20},
        ]
    ).to_csv(samples_root / 'samples_master.csv', index=False, encoding='utf-8-sig')

    recognition_metrics = {
        'lightgbm': (0.10, 0.20),
        'lstm': (0.30, 0.40),
        'transformer': (0.20, 0.30),
        'main_dual_task': (0.25, 0.35),
    }
    for model_dir, (f1_value, recall_value) in recognition_metrics.items():
        _write_prediction_bundle(results_root / 'recognition' / model_dir, 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'identification')
        pd.DataFrame(
            [
                {'experiment_name': 'exp1', 'model_name': 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'task_type': 'identification', 'fault_type': 'sd', 'split': 'test', 'f1': f1_value, 'recall': recall_value},
                {'experiment_name': 'exp1', 'model_name': 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'task_type': 'identification', 'fault_type': 'samp', 'split': 'test', 'f1': f1_value / 2, 'recall': recall_value / 2},
                {'experiment_name': 'exp1', 'model_name': 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'task_type': 'identification', 'fault_type': 'ins', 'split': 'test', 'f1': 0.0, 'recall': 0.0},
                {'experiment_name': 'exp1', 'model_name': 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'task_type': 'identification', 'fault_type': 'isc', 'split': 'test', 'f1': 0.1, 'recall': 0.1},
            ]
        ).to_csv(results_root / 'recognition' / model_dir / 'summary.csv', index=False, encoding='utf-8-sig')

    warning_metrics = {
        'threshold_trend': (0.20, 0.30),
        'lightgbm': (0.15, 0.25),
        'lstm': (0.10, 0.20),
        'transformer': (0.11, 0.21),
        'main_dual_task': (0.12, 0.22),
    }
    for model_dir, (f1_value, recall_value) in warning_metrics.items():
        _write_prediction_bundle(results_root / 'warning' / model_dir, 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'warning')
        pd.DataFrame(
            [
                {'experiment_name': 'exp3', 'model_name': 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'task_type': 'warning', 'fault_type': 'sd', 'split': 'test', 'warning_f1': f1_value, 'warning_recall': recall_value},
                {'experiment_name': 'exp3', 'model_name': 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'task_type': 'warning', 'fault_type': 'samp', 'split': 'test', 'warning_f1': f1_value / 2, 'warning_recall': recall_value / 2},
                {'experiment_name': 'exp3', 'model_name': 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir, 'task_type': 'warning', 'fault_type': 'ins', 'split': 'test', 'warning_f1': 0.0, 'warning_recall': 0.0},
            ]
        ).to_csv(results_root / 'warning' / model_dir / 'summary.csv', index=False, encoding='utf-8-sig')

    result = subprocess.run(
        [
            PYTHON,
            'scripts/export_plan_supplements.py',
            '--results-root',
            str(results_root),
            '--samples-root',
            str(samples_root),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (results_root / 'tables' / 'recognition_multifault_summary.csv').exists()
    assert (results_root / 'tables' / 'recognition_model_comparison_core.csv').exists()
    assert (results_root / 'tables' / 'warning_model_comparison_core.csv').exists()
    assert (results_root / 'tables' / 'case_analysis_summary.csv').exists()
    assert (results_root / 'tables' / 'plan_alignment_notes.md').exists()
    assert (results_root / 'figures' / 'comparison' / 'fig6_recognition_model_comparison.png').exists()
    assert (results_root / 'figures' / 'comparison' / 'fig7_warning_model_comparison.png').exists()
