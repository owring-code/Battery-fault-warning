from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import FAULT_ORDER, RESULTS_ROOT, SAMPLES_ROOT
from battery_thesis.metrics import compute_binary_metrics, compute_warning_metrics, find_best_threshold
from battery_thesis.models import SequenceClassificationBaseline
from battery_thesis.results import build_summary_row, write_result_bundle
from battery_thesis.training import (
    align_bundle_to_reference_columns,
    build_split_indices,
    compute_multilabel_pos_weight,
    load_tensor_bundle,
    normalize_bundle_with_reference_stats,
    normalize_bundle_with_train_stats,
)


def _ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f'Non-finite values detected in {name}.')


def _subset_indices(indices: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


def _resolve_samples_roots(samples_root: str, train_samples_root: str | None, eval_samples_root: str | None) -> tuple[Path, Path]:
    train_root = Path(train_samples_root or samples_root)
    eval_root = Path(eval_samples_root or train_samples_root or samples_root)
    return train_root, eval_root


def _resolve_device(device_name: str) -> torch.device:
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    return torch.device(device_name)


def _prepare_pos_weight(pos_weight: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    if pos_weight is None:
        return None
    return torch.as_tensor(np.asarray(pos_weight, dtype=np.float32), dtype=torch.float32, device=device)


def _run_epoch(model, optimizer, x_seq, targets, batch_size, device, train: bool, pos_weight: np.ndarray | None) -> float:
    order = np.arange(len(x_seq))
    if train:
        np.random.shuffle(order)
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    batch_count = 0
    pos_weight_tensor = _prepare_pos_weight(pos_weight, device)
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        batch_x = torch.as_tensor(x_seq[batch_idx], dtype=torch.float32, device=device)
        batch_y = torch.as_tensor(targets[batch_idx], dtype=torch.float32, device=device)
        _ensure_finite_tensor('x_seq', batch_x)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        _ensure_finite_tensor('logits', logits)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight_tensor)
        _ensure_finite_tensor('loss', loss)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss.detach().item())
        batch_count += 1
    return total_loss / max(batch_count, 1)


def _predict(model, x_seq, batch_size, device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_seq), batch_size):
            batch_x = torch.as_tensor(x_seq[start : start + batch_size], dtype=torch.float32, device=device)
            _ensure_finite_tensor('x_seq', batch_x)
            logits = model(batch_x)
            _ensure_finite_tensor('logits', logits)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 5), dtype=np.float32)


def _select_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    if len(y_true) == 0:
        return np.full(y_score.shape[1], 0.5, dtype=np.float32)
    return np.asarray(
        [find_best_threshold(y_true[:, fault_idx], y_score[:, fault_idx]) for fault_idx in range(y_score.shape[1])],
        dtype=np.float32,
    )


def _build_prediction_frame(
    test_df: pd.DataFrame,
    y_column: str,
    fault: str,
    task_type: str,
    model_name: str,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    score_array = np.asarray(scores, dtype=float)
    frame = test_df[['sample_id', 'vehicle_id', 'source_dataset']].copy()
    frame['model_name'] = model_name
    frame['task_type'] = task_type
    frame['fault_type'] = fault
    frame['y_true'] = pd.to_numeric(test_df[y_column], errors='coerce').fillna(0).astype(int).to_numpy()
    frame['y_pred'] = (score_array >= float(threshold)).astype(int)
    frame['y_score'] = score_array
    frame['threshold'] = float(threshold)
    return frame


def _write_progress(
    progress_path: Path,
    architecture: str,
    task_type: str,
    epoch: int,
    total_epochs: int,
    train_loss: float | None,
    val_loss: float | None,
    split_sizes: dict[str, int],
    status: str,
    thresholds: np.ndarray | None = None,
) -> None:
    lines = [
        '# Sequence Baseline Progress',
        '',
        f'- architecture: {architecture}',
        f'- task: {task_type}',
        f'- status: {status}',
        f'- epoch: {epoch}/{total_epochs}',
        f"- train_samples: {split_sizes.get('train', 0)}",
        f"- val_samples: {split_sizes.get('val', 0)}",
        f"- test_samples: {split_sizes.get('test', 0)}",
        f"- train_loss: {'' if train_loss is None else f'{train_loss:.6f}'}",
        f"- val_loss: {'' if val_loss is None else f'{val_loss:.6f}'}",
    ]
    if thresholds is not None:
        threshold_text = ', '.join(f'{fault}:{float(thresholds[idx]):.3f}' for idx, fault in enumerate(FAULT_ORDER))
        lines.append(f'- thresholds: {threshold_text}')
    progress_path.write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run LSTM/Transformer sequence baselines.')
    parser.add_argument('--architecture', required=True, choices=['lstm', 'transformer'])
    parser.add_argument('--task', required=True, choices=['identification', 'warning'])
    parser.add_argument('--samples-root', default=str(SAMPLES_ROOT))
    parser.add_argument('--train-samples-root', default=None)
    parser.add_argument('--eval-samples-root', default=None)
    parser.add_argument('--results-root', default=str(RESULTS_ROOT))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_samples_root, eval_samples_root = _resolve_samples_roots(args.samples_root, args.train_samples_root, args.eval_samples_root)
    results_root = Path(args.results_root)
    if args.eval_split == 'test':
        result_root = results_root / ('recognition' if args.task == 'identification' else 'warning') / args.architecture
    else:
        result_root = results_root / 'external_validation' / ('recognition' if args.task == 'identification' else 'warning') / args.architecture
    result_root.mkdir(parents=True, exist_ok=True)
    progress_path = result_root / 'progress.md'

    print(f'[sequence] loading architecture={args.architecture} task={args.task} train_root={train_samples_root} eval_root={eval_samples_root}', flush=True)
    train_bundle_raw = load_tensor_bundle(train_samples_root / 'samples_master.csv', train_samples_root / 'dataset_pack.npz')
    train_idx = _subset_indices(build_split_indices(train_bundle_raw.samples_master, 'train'), args.max_train_samples, args.seed)
    val_idx = _subset_indices(build_split_indices(train_bundle_raw.samples_master, 'val'), args.max_val_samples, args.seed)
    train_bundle = normalize_bundle_with_train_stats(train_bundle_raw, train_idx)

    if eval_samples_root == train_samples_root:
        eval_bundle = train_bundle
    else:
        eval_bundle_raw = load_tensor_bundle(eval_samples_root / 'samples_master.csv', eval_samples_root / 'dataset_pack.npz')
        eval_bundle_raw = align_bundle_to_reference_columns(
            eval_bundle_raw,
            train_bundle_raw.feature_columns,
            train_bundle_raw.sequence_feature_columns,
        )
        eval_bundle = normalize_bundle_with_reference_stats(eval_bundle_raw, train_bundle_raw, train_idx)

    test_idx = _subset_indices(build_split_indices(eval_bundle.samples_master, args.eval_split), args.max_test_samples, args.seed)

    train_target_matrix = train_bundle.y_id if args.task == 'identification' else train_bundle.y_warn
    eval_target_matrix = eval_bundle.y_id if args.task == 'identification' else eval_bundle.y_warn
    x_train = train_bundle.x_seq[train_idx]
    y_train = train_target_matrix[train_idx]
    x_val = train_bundle.x_seq[val_idx]
    y_val = train_target_matrix[val_idx]
    x_test = eval_bundle.x_seq[test_idx]
    y_test = eval_target_matrix[test_idx]
    split_sizes = {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)}
    print(
        f"[sequence] prepared splits train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}",
        flush=True,
    )
    _write_progress(progress_path, args.architecture, args.task, 0, args.epochs, None, None, split_sizes, 'running')

    model = SequenceClassificationBaseline(
        architecture=args.architecture,
        sequence_input_dim=train_bundle.x_seq.shape[2],
        hidden_dim=args.hidden_dim,
        output_dim=5,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=train_bundle.x_seq.shape[1],
    )
    device = _resolve_device(args.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pos_weight = compute_multilabel_pos_weight(y_train)

    best_state = None
    best_val_loss = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _run_epoch(model, optimizer, x_train, y_train, args.batch_size, device, train=True, pos_weight=pos_weight)
        val_loss = _run_epoch(model, optimizer, x_val, y_val, args.batch_size, device, train=False, pos_weight=pos_weight) if len(x_val) else train_loss
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        _write_progress(progress_path, args.architecture, args.task, epoch, args.epochs, train_loss, val_loss, split_sizes, 'running')
        print(
            f'[sequence] epoch {epoch}/{args.epochs} architecture={args.architecture} task={args.task} train_loss={train_loss:.6f} val_loss={val_loss:.6f} elapsed={time.perf_counter() - epoch_start:.1f}s',
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'[sequence] best validation checkpoint restored architecture={args.architecture} task={args.task}', flush=True)

    _write_progress(progress_path, args.architecture, args.task, args.epochs, args.epochs, history[-1]['train_loss'] if history else None, history[-1]['val_loss'] if history else None, split_sizes, 'scoring_validation')
    print(f'[sequence] scoring validation set for threshold selection architecture={args.architecture} task={args.task}', flush=True)
    val_scores = _predict(model, x_val, args.batch_size, device) if len(x_val) else np.zeros((0, 5), dtype=np.float32)

    _write_progress(progress_path, args.architecture, args.task, args.epochs, args.epochs, history[-1]['train_loss'] if history else None, history[-1]['val_loss'] if history else None, split_sizes, 'selecting_thresholds')
    print(f'[sequence] selecting validation thresholds architecture={args.architecture} task={args.task}', flush=True)
    thresholds = _select_thresholds(y_val, val_scores) if len(x_val) else np.full(5, 0.5, dtype=np.float32)

    _write_progress(progress_path, args.architecture, args.task, args.epochs, args.epochs, history[-1]['train_loss'] if history else None, history[-1]['val_loss'] if history else None, split_sizes, 'scoring_test', thresholds=thresholds)
    print(f'[sequence] scoring test set architecture={args.architecture} task={args.task}', flush=True)
    scores = _predict(model, x_test, args.batch_size, device)
    summary_rows = []
    prediction_frames = []
    data_point_rows = []

    for row in history:
        data_point_rows.append({'plot_type': 'loss_curve', 'series': 'train_loss', 'x': row['epoch'], 'y': row['train_loss']})
        data_point_rows.append({'plot_type': 'loss_curve', 'series': 'val_loss', 'x': row['epoch'], 'y': row['val_loss']})

    test_samples = eval_bundle.samples_master.iloc[test_idx].reset_index(drop=True)
    for fault_index, fault in enumerate(FAULT_ORDER):
        threshold = float(thresholds[fault_index])
        y_column = f"y_id_{fault}" if args.task == 'identification' else f"y_warn_{fault}"
        if args.task == 'identification':
            metrics = compute_binary_metrics(y_test[:, fault_index], scores[:, fault_index], threshold=threshold)
            summary_rows.append(
                build_summary_row(
                    experiment_name='sequence_baseline',
                    model_name=args.architecture,
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
            lead_times = test_samples['lead_time_sec'].to_numpy(dtype=object)
            metrics = compute_warning_metrics(y_test[:, fault_index], scores[:, fault_index], lead_times=lead_times, threshold=threshold)
            summary_rows.append(
                build_summary_row(
                    experiment_name='sequence_baseline',
                    model_name=args.architecture,
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
                test_df=test_samples,
                y_column=y_column,
                fault=fault,
                task_type=args.task,
                model_name=args.architecture,
                scores=scores[:, fault_index],
                threshold=threshold,
            )
        )

    torch.save(model.state_dict(), result_root / 'best_model.pt')
    _write_progress(progress_path, args.architecture, args.task, args.epochs, args.epochs, history[-1]['train_loss'] if history else None, history[-1]['val_loss'] if history else None, split_sizes, 'writing_results', thresholds=thresholds)
    print(f'[sequence] writing result bundle architecture={args.architecture} task={args.task}', flush=True)
    write_result_bundle(
        output_dir=result_root,
        summary=pd.DataFrame(summary_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        data_points=pd.DataFrame(data_point_rows),
        notes=f'{args.architecture} sequence baseline for {args.task} with train pos_weight, validation thresholds, eval_split={args.eval_split}, train_samples_root={train_samples_root}, eval_samples_root={eval_samples_root}.',
    )
    _write_progress(progress_path, args.architecture, args.task, args.epochs, args.epochs, history[-1]['train_loss'] if history else None, history[-1]['val_loss'] if history else None, split_sizes, 'completed', thresholds=thresholds)
    print(f'[sequence] finished output_dir={result_root} total_elapsed={time.perf_counter() - run_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()


