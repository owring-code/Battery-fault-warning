from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import FAULT_ORDER, RESULTS_ROOT, SAMPLES_ROOT
from battery_thesis.metrics import compute_warning_metrics, find_best_threshold
from battery_thesis.results import build_summary_row, write_result_bundle


FAULT_FEATURE_RULES: dict[str, list[tuple[str, float]]] = {
    'sd': [('sd_cell_dev_min', -1.0), ('sd_rest_drop_rate', -1.0), ('sd_weak_cell_repeat_ratio', 1.0)],
    'isc': [('isc_module_vdrop_max', -1.0), ('isc_module_tmax', 1.0), ('isc_module_trise_max', 1.0)],
    'conn': [('conn_dvdt_max', 1.0), ('conn_neighbor_response_gap', 1.0), ('conn_abnormal_point_ratio', 1.0)],
    'samp': [('samp_local_median_residual', 1.0), ('samp_sensor_jump_count', 1.0), ('samp_abnormal_count_ratio', 1.0)],
    'ins': [('ins_rins_min', -1.0), ('ins_threshold_margin', -1.0), ('ins_low_rins_persistence', 1.0)],
}


def _subset_indices(indices: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return np.asarray(indices, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.asarray(indices, dtype=int), size=max_samples, replace=False))


def _resolve_samples_roots(samples_root: str, train_samples_root: str | None, eval_samples_root: str | None) -> tuple[Path, Path]:
    train_root = Path(train_samples_root or samples_root)
    eval_root = Path(eval_samples_root or train_samples_root or samples_root)
    return train_root, eval_root


def _load_merged_frame(samples_root: Path) -> pd.DataFrame:
    samples_df = pd.read_csv(samples_root / 'samples_master.csv')
    features_df = pd.read_csv(samples_root / 'features_all.csv')
    return samples_df.merge(features_df, on='sample_id', how='inner')


def _ensure_rule_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required_features = sorted({feature for rules in FAULT_FEATURE_RULES.values() for feature, _ in rules})
    aligned = frame.copy()
    for feature in required_features:
        if feature not in aligned.columns:
            aligned[feature] = 0.0
    return aligned


def _score_feature(train_values: pd.Series, eval_values: pd.Series, direction: float) -> np.ndarray:
    train_array = pd.to_numeric(train_values, errors='coerce').fillna(0.0).to_numpy(dtype=float) * direction
    eval_array = pd.to_numeric(eval_values, errors='coerce').fillna(0.0).to_numpy(dtype=float) * direction
    train_min = float(np.min(train_array)) if train_array.size else 0.0
    train_max = float(np.max(train_array)) if train_array.size else 0.0
    span = train_max - train_min
    if span <= 0:
        return np.zeros(len(eval_array), dtype=float)
    normalized = (eval_array - train_min) / span
    return np.clip(normalized, 0.0, 1.0)


def _pick_best_rule(train_df: pd.DataFrame, val_df: pd.DataFrame, fault: str) -> tuple[str, float, float, dict[str, float | None]]:
    y_val = val_df[f'y_warn_{fault}'].to_numpy(dtype=int)
    lead_times = val_df['lead_time_sec'].to_numpy(dtype=object) if 'lead_time_sec' in val_df.columns else None
    best_choice: tuple[str, float, float, dict[str, float | None]] | None = None
    best_key = (-1.0, -1.0, 1.0)

    for feature_name, direction in FAULT_FEATURE_RULES[fault]:
        val_scores = _score_feature(train_df[feature_name], val_df[feature_name], direction)
        threshold = find_best_threshold(y_val, val_scores) if len(val_df) else 0.5
        metrics = compute_warning_metrics(y_val, val_scores, lead_times=lead_times, threshold=threshold)
        key = (
            float(metrics['warning_f1'] or 0.0),
            float(metrics['warning_recall'] or 0.0),
            -float(metrics['false_alarm_rate'] or 0.0),
        )
        if key > best_key:
            best_key = key
            best_choice = (feature_name, direction, float(threshold), metrics)

    if best_choice is None:
        return FAULT_FEATURE_RULES[fault][0][0], FAULT_FEATURE_RULES[fault][0][1], 0.5, compute_warning_metrics([], [], threshold=0.5)
    return best_choice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run threshold-trend warning baseline on features_all.csv.')
    parser.add_argument('--fault', default='all', choices=['all', *FAULT_ORDER])
    parser.add_argument('--samples-root', default=str(SAMPLES_ROOT))
    parser.add_argument('--train-samples-root', default=None)
    parser.add_argument('--eval-samples-root', default=None)
    parser.add_argument('--results-root', default=str(RESULTS_ROOT))
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    parser.add_argument('--max-test-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=20260407)
    parser.add_argument('--eval-split', choices=['test', 'external_test'], default='test')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_samples_root, eval_samples_root = _resolve_samples_roots(args.samples_root, args.train_samples_root, args.eval_samples_root)
    results_root = Path(args.results_root)

    train_merged = _ensure_rule_columns(_load_merged_frame(train_samples_root))
    eval_merged = train_merged if eval_samples_root == train_samples_root else _ensure_rule_columns(_load_merged_frame(eval_samples_root))

    train_idx = _subset_indices(np.flatnonzero(train_merged['split'].to_numpy() == 'train'), args.max_train_samples, args.seed)
    val_idx = _subset_indices(np.flatnonzero(train_merged['split'].to_numpy() == 'val'), args.max_val_samples, args.seed)
    test_idx = _subset_indices(np.flatnonzero(eval_merged['split'].to_numpy() == args.eval_split), args.max_test_samples, args.seed)

    train_df = train_merged.iloc[train_idx].reset_index(drop=True)
    val_df = train_merged.iloc[val_idx].reset_index(drop=True)
    test_df = eval_merged.iloc[test_idx].reset_index(drop=True)

    faults = FAULT_ORDER if args.fault == 'all' else [args.fault]
    summary_rows = []
    prediction_rows = []
    data_point_rows = []
    notes: list[str] = [
        f'Threshold-trend warning baseline with validation feature/threshold selection and eval_split={args.eval_split}.',
        f'train_samples_root={train_samples_root}',
        f'eval_samples_root={eval_samples_root}',
    ]

    for fault in faults:
        feature_name, direction, threshold, _ = _pick_best_rule(train_df, val_df, fault)
        test_scores = _score_feature(train_df[feature_name], test_df[feature_name], direction)
        y_test = test_df[f'y_warn_{fault}'].to_numpy(dtype=int)
        lead_times = test_df['lead_time_sec'].to_numpy(dtype=object) if 'lead_time_sec' in test_df.columns else None
        metrics = compute_warning_metrics(y_test, test_scores, lead_times=lead_times, threshold=threshold)
        notes.append(f'{fault}: feature={feature_name}, direction={direction:+.0f}, threshold={threshold:.3f}')

        summary_rows.append(
            build_summary_row(
                experiment_name='threshold_warning_baseline',
                model_name='threshold_trend',
                task_type='warning',
                fault_type=fault,
                split=args.eval_split,
                metrics=metrics,
            )
        )
        data_point_rows.extend(
            [
                {'plot_type': 'bar_metric', 'category': fault, 'series': 'warning_f1', 'value': metrics['warning_f1']},
                {'plot_type': 'bar_metric', 'category': fault, 'series': 'warning_recall', 'value': metrics['warning_recall']},
                {'plot_type': 'threshold', 'category': fault, 'series': 'threshold', 'value': threshold},
            ]
        )

        for row_index, row in test_df[['sample_id', 'vehicle_id', 'source_dataset']].iterrows():
            score = float(test_scores[row_index]) if len(test_scores) else 0.0
            prediction_rows.append(
                {
                    'sample_id': row['sample_id'],
                    'vehicle_id': row['vehicle_id'],
                    'source_dataset': row['source_dataset'],
                    'model_name': 'threshold_trend',
                    'task_type': 'warning',
                    'fault_type': fault,
                    'y_true': int(test_df.loc[row_index, f'y_warn_{fault}']),
                    'y_pred': int(score >= threshold),
                    'y_score': score,
                    'threshold': threshold,
                    'rule_feature': feature_name,
                    'rule_direction': int(direction),
                }
            )

    if args.eval_split == 'test':
        output_dir = results_root / 'warning' / 'threshold_trend'
    else:
        output_dir = results_root / 'external_validation' / 'warning' / 'threshold_trend'
    write_result_bundle(
        output_dir=output_dir,
        summary=pd.DataFrame(summary_rows),
        predictions=pd.DataFrame(prediction_rows),
        data_points=pd.DataFrame(data_point_rows),
        notes='\n'.join(notes),
    )


if __name__ == '__main__':
    main()
