from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _build_fault_rows(sample_id: str, vehicle_id: str, split: str, signal: float) -> dict[str, object]:
    row: dict[str, object] = {
        'sample_id': sample_id,
        'vehicle_id': vehicle_id,
        'source_dataset': 'structured_dataset',
        'split': split,
        'lead_time_sec': 30 if 0.45 <= signal < 0.75 else None,
    }
    for fault in ['sd', 'isc', 'conn', 'samp', 'ins']:
        row[f'y_id_{fault}'] = 1 if fault == 'sd' and signal >= 0.75 else 0
        row[f'y_warn_{fault}'] = 1 if fault == 'sd' and 0.45 <= signal < 0.75 else 0
    return row


def test_run_lightgbm_recognition_supports_warning_thresholds_and_index_subsets(tmp_path: Path):
    samples_root = tmp_path / 'samples'
    results_root = tmp_path / 'results'
    samples_root.mkdir()
    results_root.mkdir()

    samples_rows = []
    feature_rows = []
    signals = (
        [('train', value) for value in np.linspace(0.05, 0.95, 30)]
        + [('val', value) for value in np.linspace(0.10, 0.90, 8)]
        + [('test', value) for value in np.linspace(0.15, 0.85, 6)]
    )
    for index, (split, signal) in enumerate(signals):
        sample_id = f's_{index:03d}'
        vehicle_id = f'v_{index // 4:02d}'
        samples_rows.append(_build_fault_rows(sample_id, vehicle_id, split, float(signal)))
        feature_rows.append(
            {
                'sample_id': sample_id,
                'shared_soc': 40 + signal * 10,
                'sd_cell_dev_min': -signal,
                'samp_local_median_residual': signal / 2,
                'ins_threshold_margin': 1000 - signal * 100,
            }
        )

    samples_df = pd.DataFrame(samples_rows)
    features_df = pd.DataFrame(feature_rows)
    samples_df.to_csv(samples_root / 'samples_master.csv', index=False, encoding='utf-8-sig')
    features_df.to_csv(samples_root / 'features_all.csv', index=False, encoding='utf-8-sig')

    seed = 20260407
    result = subprocess.run(
        [
            PYTHON,
            'scripts/run_lightgbm_recognition.py',
            '--task',
            'warning',
            '--fault',
            'sd',
            '--samples-root',
            str(samples_root),
            '--results-root',
            str(results_root),
            '--max-test-samples',
            '3',
            '--seed',
            str(seed),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    predictions_path = results_root / 'warning' / 'lightgbm' / 'predictions.csv'
    summary_path = results_root / 'warning' / 'lightgbm' / 'summary.csv'
    predictions = pd.read_csv(predictions_path)
    summary = pd.read_csv(summary_path)

    test_indices = np.flatnonzero(samples_df['split'].to_numpy() == 'test')
    expected_indices = np.sort(np.random.default_rng(seed).choice(test_indices, size=3, replace=False))
    expected_sample_ids = samples_df.iloc[expected_indices]['sample_id'].tolist()

    assert summary['task_type'].tolist() == ['warning']
    assert 'threshold' in predictions.columns
    assert predictions['sample_id'].tolist() == expected_sample_ids
    assert predictions['threshold'].nunique() == 1



def test_fit_predict_pair_trains_lightgbm_only_once(monkeypatch):
    import scripts.run_lightgbm_recognition as runner

    class DummyClassifier:
        fit_calls = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x_train, y_train):
            DummyClassifier.fit_calls += 1
            self.bias = float(pd.Series(y_train).mean())
            return self

        def predict_proba(self, x_eval):
            scores = np.clip(np.linspace(self.bias, self.bias + 0.2, len(x_eval)), 0.0, 1.0)
            return np.column_stack([1.0 - scores, scores])

    monkeypatch.setattr(runner, 'LGBMClassifier', DummyClassifier)

    x_train = pd.DataFrame({'f1': [0.1, 0.2, 0.8, 0.9]})
    y_train = pd.Series([0, 0, 1, 1])
    x_val = pd.DataFrame({'f1': [0.3, 0.7]})
    x_test = pd.DataFrame({'f1': [0.4, 0.6, 0.95]})

    val_scores, test_scores = runner._fit_predict_pair(x_train, y_train, x_val, x_test, seed=20260407)

    assert DummyClassifier.fit_calls == 1
    assert len(val_scores) == 2
    assert len(test_scores) == 3


def test_build_prediction_frame_keeps_sample_order_and_threshold():
    import scripts.run_lightgbm_recognition as runner

    test_df = pd.DataFrame(
        {
            'sample_id': ['s1', 's2'],
            'vehicle_id': ['v1', 'v2'],
            'source_dataset': ['structured_dataset', 'structured_dataset'],
            'y_warn_sd': [0, 1],
        }
    )

    frame = runner._build_prediction_frame(
        test_df=test_df,
        y_column='y_warn_sd',
        fault='sd',
        task_type='warning',
        scores=np.asarray([0.2, 0.8], dtype=float),
        threshold=0.5,
    )

    assert frame['sample_id'].tolist() == ['s1', 's2']
    assert frame['y_pred'].tolist() == [0, 1]
    assert frame['threshold'].tolist() == [0.5, 0.5]



def test_run_lightgbm_recognition_supports_external_test_eval_split(tmp_path: Path):
    samples_root = tmp_path / 'samples'
    results_root = tmp_path / 'results'
    samples_root.mkdir()
    results_root.mkdir()

    samples_rows = []
    feature_rows = []
    signals = (
        [('train', value) for value in np.linspace(0.05, 0.95, 20)]
        + [('val', value) for value in np.linspace(0.10, 0.90, 6)]
        + [('external_test', value) for value in np.linspace(0.15, 0.85, 4)]
    )
    for index, (split, signal) in enumerate(signals):
        sample_id = f'se_{index:03d}'
        vehicle_id = f've_{index // 4:02d}'
        samples_rows.append(_build_fault_rows(sample_id, vehicle_id, split, float(signal)))
        feature_rows.append(
            {
                'sample_id': sample_id,
                'shared_soc': 40 + signal * 10,
                'sd_cell_dev_min': -signal,
                'samp_local_median_residual': signal / 2,
                'ins_threshold_margin': 1000 - signal * 100,
            }
        )

    samples_df = pd.DataFrame(samples_rows)
    features_df = pd.DataFrame(feature_rows)
    samples_df.to_csv(samples_root / 'samples_master.csv', index=False, encoding='utf-8-sig')
    features_df.to_csv(samples_root / 'features_all.csv', index=False, encoding='utf-8-sig')

    result = subprocess.run(
        [
            PYTHON,
            'scripts/run_lightgbm_recognition.py',
            '--task',
            'warning',
            '--fault',
            'sd',
            '--samples-root',
            str(samples_root),
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

    predictions_path = results_root / 'external_validation' / 'warning' / 'lightgbm' / 'predictions.csv'
    summary_path = results_root / 'external_validation' / 'warning' / 'lightgbm' / 'summary.csv'
    predictions = pd.read_csv(predictions_path)
    summary = pd.read_csv(summary_path)

    assert summary['split'].tolist() == ['external_test']
    assert set(predictions['sample_id']) == set(samples_df.loc[samples_df['split'] == 'external_test', 'sample_id'])
