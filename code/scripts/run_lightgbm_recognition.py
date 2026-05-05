from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import FAULT_ORDER, RESULTS_ROOT, SAMPLES_ROOT
from battery_thesis.metrics import compute_binary_metrics, compute_warning_metrics, find_best_threshold
from battery_thesis.results import build_summary_row, write_result_bundle


def _subset_indices(indices: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return np.asarray(indices, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.asarray(indices, dtype=int), size=max_samples, replace=False))


def _resolve_samples_roots(samples_root: str, train_samples_root: str | None, eval_samples_root: str | None) -> tuple[Path, Path]:
    train_root = Path(train_samples_root or samples_root)
    eval_root = Path(eval_samples_root or train_samples_root or samples_root)
    return train_root, eval_root


def _load_merged_frame(samples_path: Path, features_path: Path, task_type: str, faults: list[str]) -> tuple[pd.DataFrame, list[str]]:
    label_prefix = 'y_id_' if task_type == 'identification' else 'y_warn_'
    sample_columns = ['sample_id', 'vehicle_id', 'source_dataset', 'split']
    if task_type == 'warning':
        sample_columns.append('lead_time_sec')
    sample_columns.extend(f'{label_prefix}{fault}' for fault in faults)

    samples_df = pd.read_csv(samples_path, usecols=sample_columns)
    features_df = pd.read_csv(features_path)
    feature_columns = [column for column in features_df.columns if column != 'sample_id']

    if len(samples_df) == len(features_df) and samples_df['sample_id'].equals(features_df['sample_id']):
        merged = pd.concat(
            [samples_df.reset_index(drop=True), features_df[feature_columns].reset_index(drop=True)],
            axis=1,
        )
    else:
        merged = samples_df.merge(features_df, on='sample_id', how='inner')
    return merged, feature_columns


def _align_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = frame.reindex(columns=feature_columns, fill_value=0.0).copy()
    for column in aligned.columns:
        aligned[column] = pd.to_numeric(aligned[column], errors='coerce').fillna(0.0)
    return aligned


def _predict_scores(model: LGBMClassifier, x_eval: pd.DataFrame) -> np.ndarray:
    if x_eval.empty:
        return np.zeros(0, dtype=float)
    return np.asarray(model.predict_proba(x_eval)[:, 1], dtype=float)


def _fit_predict_pair(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if y_train.nunique() < 2:
        positive_rate = float(y_train.iloc[0]) if not y_train.empty else 0.0
        return (
            np.full(len(x_val), positive_rate, dtype=float),
            np.full(len(x_test), positive_rate, dtype=float),
        )

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=1,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return _predict_scores(model, x_val), _predict_scores(model, x_test)


def _build_prediction_frame(
    test_df: pd.DataFrame,
    y_column: str,
    fault: str,
    task_type: str,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    score_array = np.asarray(scores, dtype=float)
    prediction_frame = test_df[['sample_id', 'vehicle_id', 'source_dataset']].copy()
    prediction_frame['model_name'] = 'lightgbm'
    prediction_frame['task_type'] = task_type
    prediction_frame['fault_type'] = fault
    prediction_frame['y_true'] = pd.to_numeric(test_df[y_column], errors='coerce').fillna(0).astype(int).to_numpy()
    prediction_frame['y_pred'] = (score_array >= float(threshold)).astype(int)
    prediction_frame['y_score'] = score_array
    prediction_frame['threshold'] = float(threshold)
    return prediction_frame


def _write_progress(
    output_dir: Path,
    task_type: str,
    completed_faults: list[str],
    threshold_notes: list[str],
    split_sizes: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        '# LightGBM Progress',
        '',
        f'- task: {task_type}',
        f"- completed_faults: {', '.join(completed_faults) if completed_faults else 'none'}",
        f"- train_samples: {split_sizes.get('train', 0)}",
        f"- val_samples: {split_sizes.get('val', 0)}",
        f"- test_samples: {split_sizes.get('test', 0)}",
        f"- thresholds: {', '.join(threshold_notes) if threshold_notes else 'pending'}",
    ]
    (output_dir / 'progress.md').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run LightGBM baselines on features_all.csv.')
    parser.add_argument('--task', default='identification', choices=['identification', 'warning'])
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
    run_start = time.perf_counter()

    train_samples_root, eval_samples_root = _resolve_samples_roots(args.samples_root, args.train_samples_root, args.eval_samples_root)
    results_root = Path(args.results_root)
    if args.eval_split == 'test':
        output_dir = results_root / ('recognition' if args.task == 'identification' else 'warning') / 'lightgbm'
    else:
        output_dir = results_root / 'external_validation' / ('recognition' if args.task == 'identification' else 'warning') / 'lightgbm'

    faults = FAULT_ORDER if args.fault == 'all' else [args.fault]
    print(f'[lightgbm] loading task={args.task} faults={faults} train_root={train_samples_root} eval_root={eval_samples_root}', flush=True)
    train_merged, feature_columns = _load_merged_frame(train_samples_root / 'samples_master.csv', train_samples_root / 'features_all.csv', args.task, faults)
    eval_merged, _ = (train_merged, feature_columns) if eval_samples_root == train_samples_root else _load_merged_frame(eval_samples_root / 'samples_master.csv', eval_samples_root / 'features_all.csv', args.task, faults)

    train_idx = _subset_indices(np.flatnonzero(train_merged['split'].to_numpy() == 'train'), args.max_train_samples, args.seed)
    val_idx = _subset_indices(np.flatnonzero(train_merged['split'].to_numpy() == 'val'), args.max_val_samples, args.seed)
    test_idx = _subset_indices(np.flatnonzero(eval_merged['split'].to_numpy() == args.eval_split), args.max_test_samples, args.seed)

    train_df = train_merged.iloc[train_idx].reset_index(drop=True)
    val_df = train_merged.iloc[val_idx].reset_index(drop=True)
    test_df = eval_merged.iloc[test_idx].reset_index(drop=True)
    split_sizes = {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)}
    print(
        f"[lightgbm] prepared splits train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} features={len(feature_columns)}",
        flush=True,
    )

    x_train = _align_feature_frame(train_df, feature_columns)
    x_val = _align_feature_frame(val_df, feature_columns)
    x_test = _align_feature_frame(test_df, feature_columns)

    summary_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    data_point_rows: list[dict[str, object]] = []
    threshold_notes: list[str] = []
    completed_faults: list[str] = []
    _write_progress(output_dir, args.task, completed_faults, threshold_notes, split_sizes)

    for fault_index, fault in enumerate(faults, start=1):
        fault_start = time.perf_counter()
        y_column = f'y_id_{fault}' if args.task == 'identification' else f'y_warn_{fault}'
        y_train = train_df[y_column]
        y_val = val_df[y_column]
        y_test = test_df[y_column]

        print(f'[lightgbm] ({fault_index}/{len(faults)}) training fault={fault}', flush=True)
        val_scores, test_scores = _fit_predict_pair(x_train, y_train, x_val, x_test, args.seed)
        threshold = find_best_threshold(y_val.to_numpy(), val_scores) if len(val_df) else 0.5
        threshold_notes.append(f'{fault}:{threshold:.3f}')

        if args.task == 'identification':
            metrics = compute_binary_metrics(y_test.to_numpy(), test_scores, threshold=threshold)
            summary_rows.append(
                build_summary_row(
                    experiment_name='lightgbm_baseline',
                    model_name='lightgbm',
                    task_type='identification',
                    fault_type=fault,
                    split=args.eval_split,
                    metrics=metrics,
                )
            )
            data_point_rows.extend(
                [
                    {'plot_type': 'bar_metric', 'category': fault, 'series': 'f1', 'value': metrics['f1']},
                    {'plot_type': 'bar_metric', 'category': fault, 'series': 'recall', 'value': metrics['recall']},
                    {'plot_type': 'threshold', 'category': fault, 'series': 'threshold', 'value': threshold},
                ]
            )
        else:
            lead_times = test_df['lead_time_sec'].to_numpy(dtype=object) if 'lead_time_sec' in test_df.columns else None
            metrics = compute_warning_metrics(y_test.to_numpy(), test_scores, lead_times=lead_times, threshold=threshold)
            summary_rows.append(
                build_summary_row(
                    experiment_name='lightgbm_baseline',
                    model_name='lightgbm',
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

        prediction_frames.append(
            _build_prediction_frame(
                test_df=test_df,
                y_column=y_column,
                fault=fault,
                task_type=args.task,
                scores=test_scores,
                threshold=threshold,
            )
        )
        completed_faults.append(fault)
        _write_progress(output_dir, args.task, completed_faults, threshold_notes, split_sizes)
        print(
            f'[lightgbm] ({fault_index}/{len(faults)}) done fault={fault} threshold={threshold:.3f} elapsed={time.perf_counter() - fault_start:.1f}s',
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    data_points_df = pd.DataFrame(data_point_rows)
    write_result_bundle(
        output_dir=output_dir,
        summary=summary_df,
        predictions=predictions_df,
        data_points=data_points_df,
        notes='LightGBM baseline with validation thresholds\nthresholds=' + ', '.join(threshold_notes) + f'\ntrain_samples_root={train_samples_root}\neval_samples_root={eval_samples_root}',
    )
    print(f'[lightgbm] finished output_dir={output_dir} total_elapsed={time.perf_counter() - run_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
