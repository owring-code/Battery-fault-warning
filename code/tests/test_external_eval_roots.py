from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
FAULTS = ['sd', 'isc', 'conn', 'samp', 'ins']
FEATURE_COLUMNS = np.array(['shared_base', 'sd_signal', 'isc_signal', 'conn_signal', 'samp_signal', 'ins_signal'])
SEQUENCE_COLUMNS = np.array(['f1', 'f2'])


def _fault_row(sample_id: str, vehicle_id: str, split: str, signal: float, source_dataset: str = 'structured_dataset') -> dict[str, object]:
    row: dict[str, object] = {
        'sample_id': sample_id,
        'vehicle_id': vehicle_id,
        'source_dataset': source_dataset,
        'split': split,
        'lead_time_sec': 30 if 0.45 <= signal < 0.75 else None,
    }
    for fault in FAULTS:
        row[f'y_id_{fault}'] = 1 if fault == 'sd' and signal >= 0.75 else 0
        row[f'y_warn_{fault}'] = 1 if fault == 'sd' and 0.45 <= signal < 0.75 else 0
    return row


def _write_tensor_bundle(samples_root: Path, split_signals: list[tuple[str, float]], source_dataset: str = 'structured_dataset') -> pd.DataFrame:
    samples_root.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    x_seq_rows = []
    x_feat_rows = []
    y_id_rows = []
    y_warn_rows = []
    sample_ids = []
    feature_rows = []

    for index, (split, signal) in enumerate(split_signals):
        sample_id = f'{source_dataset[:2]}_{index:03d}'
        vehicle_id = f'{source_dataset[:2]}v_{index // 2:02d}'
        row = _fault_row(sample_id, vehicle_id, split, float(signal), source_dataset=source_dataset)
        sample_rows.append(row)
        x_seq_rows.append(np.asarray([[signal, signal / 2], [signal + 0.1, signal / 2 + 0.1]], dtype=np.float32))
        x_feat_rows.append(np.asarray([signal, signal, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        y_id_rows.append([row[f'y_id_{fault}'] for fault in FAULTS])
        y_warn_rows.append([row[f'y_warn_{fault}'] for fault in FAULTS])
        sample_ids.append(sample_id)
        feature_rows.append(
            {
                'sample_id': sample_id,
                'shared_base': signal,
                'sd_cell_dev_min': -signal,
                'sd_rest_drop_rate': -signal / 2,
                'sd_weak_cell_repeat_ratio': signal,
                'isc_module_vdrop_max': signal,
                'isc_module_tmax': signal,
                'isc_module_trise_max': signal,
                'conn_dvdt_max': signal,
                'conn_neighbor_response_gap': signal,
                'conn_abnormal_point_ratio': signal,
                'samp_local_median_residual': signal,
                'samp_sensor_jump_count': signal,
                'samp_abnormal_count_ratio': signal,
                'ins_rins_min': 1000 - signal * 100,
                'ins_threshold_margin': 1000 - signal * 100,
                'ins_low_rins_persistence': signal,
            }
        )

    samples_df = pd.DataFrame(sample_rows)
    samples_df.to_csv(samples_root / 'samples_master.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(feature_rows).to_csv(samples_root / 'features_all.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(
        samples_root / 'dataset_pack.npz',
        X_seq=np.stack(x_seq_rows).astype(np.float32),
        X_feat=np.stack(x_feat_rows).astype(np.float32),
        y_id=np.asarray(y_id_rows, dtype=np.int64),
        y_warn=np.asarray(y_warn_rows, dtype=np.int64),
        sample_id=np.asarray(sample_ids, dtype=str),
        feature_columns=FEATURE_COLUMNS,
        sequence_feature_columns=SEQUENCE_COLUMNS,
    )
    return samples_df


def test_run_sequence_baseline_supports_separate_train_and_eval_roots(tmp_path: Path):
    train_root = tmp_path / 'train_samples'
    eval_root = tmp_path / 'eval_samples'
    results_root = tmp_path / 'results'

    _write_tensor_bundle(
        train_root,
        [('train', 0.1), ('train', 0.9), ('train', 0.2), ('train', 0.8), ('val', 0.15), ('val', 0.85)],
    )
    eval_df = _write_tensor_bundle(
        eval_root,
        [('external_test', 0.25), ('external_test', 0.75), ('external_test', 0.95)],
        source_dataset='raw_dataset',
    )

    result = subprocess.run(
        [
            PYTHON,
            'scripts/run_sequence_baseline.py',
            '--architecture',
            'lstm',
            '--task',
            'identification',
            '--train-samples-root',
            str(train_root),
            '--eval-samples-root',
            str(eval_root),
            '--results-root',
            str(results_root),
            '--device',
            'cpu',
            '--epochs',
            '1',
            '--batch-size',
            '2',
            '--eval-split',
            'external_test',
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(results_root / 'external_validation' / 'recognition' / 'lstm' / 'summary.csv')
    predictions = pd.read_csv(results_root / 'external_validation' / 'recognition' / 'lstm' / 'predictions.csv')
    assert summary['split'].tolist() == ['external_test'] * 5
    assert set(predictions['sample_id']) == set(eval_df['sample_id'])


def test_run_dual_task_model_supports_separate_train_and_eval_roots(tmp_path: Path):
    train_root = tmp_path / 'train_samples'
    eval_root = tmp_path / 'eval_samples'
    results_root = tmp_path / 'results'

    _write_tensor_bundle(
        train_root,
        [('train', 0.1), ('train', 0.9), ('train', 0.2), ('train', 0.8), ('val', 0.15), ('val', 0.85)],
    )
    eval_df = _write_tensor_bundle(
        eval_root,
        [('external_test', 0.25), ('external_test', 0.75), ('external_test', 0.95)],
        source_dataset='raw_dataset',
    )

    result = subprocess.run(
        [
            PYTHON,
            'scripts/run_dual_task_model.py',
            '--train-samples-root',
            str(train_root),
            '--eval-samples-root',
            str(eval_root),
            '--results-root',
            str(results_root),
            '--device',
            'cpu',
            '--epochs-stage1',
            '1',
            '--epochs-joint',
            '1',
            '--hidden-dim',
            '8',
            '--num-layers',
            '1',
            '--batch-size',
            '2',
            '--eval-split',
            'external_test',
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    recognition = pd.read_csv(results_root / 'external_validation' / 'recognition' / 'main_dual_task' / 'summary.csv')
    warning = pd.read_csv(results_root / 'external_validation' / 'warning' / 'main_dual_task' / 'summary.csv')
    predictions = pd.read_csv(results_root / 'external_validation' / 'recognition' / 'main_dual_task' / 'predictions.csv')
    assert recognition['split'].tolist() == ['external_test'] * 5
    assert warning['split'].tolist() == ['external_test'] * 5
    assert set(predictions['sample_id']) == set(eval_df['sample_id'])


def test_run_lightgbm_recognition_supports_separate_train_and_eval_roots(tmp_path: Path):
    train_root = tmp_path / 'train_samples'
    eval_root = tmp_path / 'eval_samples'
    results_root = tmp_path / 'results'

    _write_tensor_bundle(
        train_root,
        [('train', 0.05), ('train', 0.95), ('train', 0.15), ('train', 0.85), ('val', 0.20), ('val', 0.80)],
    )
    eval_df = _write_tensor_bundle(
        eval_root,
        [('external_test', 0.25), ('external_test', 0.75), ('external_test', 0.90)],
        source_dataset='raw_dataset',
    )

    result = subprocess.run(
        [
            PYTHON,
            'scripts/run_lightgbm_recognition.py',
            '--task',
            'warning',
            '--fault',
            'sd',
            '--train-samples-root',
            str(train_root),
            '--eval-samples-root',
            str(eval_root),
            '--results-root',
            str(results_root),
            '--eval-split',
            'external_test',
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(results_root / 'external_validation' / 'warning' / 'lightgbm' / 'summary.csv')
    predictions = pd.read_csv(results_root / 'external_validation' / 'warning' / 'lightgbm' / 'predictions.csv')
    assert summary['split'].tolist() == ['external_test']
    assert set(predictions['sample_id']) == set(eval_df['sample_id'])


def test_run_threshold_warning_baseline_supports_separate_train_and_eval_roots(tmp_path: Path):
    train_root = tmp_path / 'train_samples'
    eval_root = tmp_path / 'eval_samples'
    results_root = tmp_path / 'results'

    _write_tensor_bundle(
        train_root,
        [('train', 0.05), ('train', 0.95), ('train', 0.15), ('train', 0.85), ('val', 0.20), ('val', 0.80)],
    )
    eval_df = _write_tensor_bundle(
        eval_root,
        [('external_test', 0.25), ('external_test', 0.75), ('external_test', 0.90)],
        source_dataset='raw_dataset',
    )

    result = subprocess.run(
        [
            PYTHON,
            'scripts/run_threshold_warning_baseline.py',
            '--train-samples-root',
            str(train_root),
            '--eval-samples-root',
            str(eval_root),
            '--results-root',
            str(results_root),
            '--eval-split',
            'external_test',
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(results_root / 'external_validation' / 'warning' / 'threshold_trend' / 'summary.csv')
    predictions = pd.read_csv(results_root / 'external_validation' / 'warning' / 'threshold_trend' / 'predictions.csv')
    assert summary['split'].tolist() == ['external_test'] * 5
    assert set(predictions['sample_id']) == set(eval_df['sample_id'])
