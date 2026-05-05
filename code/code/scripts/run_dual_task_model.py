from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import FAULT_ORDER, RESULTS_ROOT, SAMPLES_ROOT
from battery_thesis.metrics import compute_binary_metrics, compute_warning_metrics, find_best_threshold
from battery_thesis.models import MultiFaultDualTaskModel
from battery_thesis.results import build_summary_row, write_result_bundle
from battery_thesis.training import (
    align_bundle_to_reference_columns,
    build_split_indices,
    compute_joint_loss,
    compute_multilabel_pos_weight,
    create_dataloader,
    load_tensor_bundle,
    normalize_bundle_with_reference_stats,
    normalize_bundle_with_train_stats,
    predict_scores,
    train_one_epoch,
)


ABLATION_CHOICES = [
    'none',
    'no_fault_specific_features',
    'no_expert_heads',
    'no_warning_task',
    'no_label_quality_control',
]


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


def _evaluate_joint_loss(model, dataloader, device, lambda_warn: float, id_pos_weight, warn_pos_weight) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_id_loss = 0.0
    total_warn_loss = 0.0
    batch_count = 0
    id_pos_weight_tensor = None if id_pos_weight is None else torch.as_tensor(id_pos_weight, dtype=torch.float32, device=device)
    warn_pos_weight_tensor = None if warn_pos_weight is None else torch.as_tensor(warn_pos_weight, dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in dataloader:
            x_seq = batch['x_seq'].to(device, non_blocking=True)
            x_feat = batch['x_feat'].to(device, non_blocking=True)
            if not torch.isfinite(x_seq).all():
                raise RuntimeError('Non-finite values detected in x_seq during evaluation.')
            if not torch.isfinite(x_feat).all():
                raise RuntimeError('Non-finite values detected in x_feat during evaluation.')
            outputs = model(x_seq, x_feat)
            if not torch.isfinite(outputs['id_logits']).all() or not torch.isfinite(outputs['warn_logits']).all():
                raise RuntimeError('Non-finite logits detected during evaluation.')
            losses = compute_joint_loss(
                outputs['id_logits'],
                outputs['warn_logits'],
                batch['y_id'].to(device, non_blocking=True),
                batch['y_warn'].to(device, non_blocking=True),
                lambda_warn=lambda_warn,
                id_pos_weight=id_pos_weight_tensor,
                warn_pos_weight=warn_pos_weight_tensor,
            )
            if not torch.isfinite(losses['loss']):
                raise RuntimeError('Non-finite loss detected during evaluation.')
            total_loss += float(losses['loss'].detach().item())
            total_id_loss += float(losses['id_loss'])
            total_warn_loss += float(losses['warn_loss'])
            batch_count += 1
    divisor = max(batch_count, 1)
    return {
        'loss': total_loss / divisor,
        'id_loss': total_id_loss / divisor,
        'warn_loss': total_warn_loss / divisor,
    }


def _build_prediction_frame(
    samples_df: pd.DataFrame,
    y_column: str,
    fault: str,
    task_type: str,
    model_name: str,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    score_array = np.asarray(scores, dtype=float)
    frame = samples_df[['sample_id', 'vehicle_id', 'source_dataset']].copy()
    frame['model_name'] = model_name
    frame['task_type'] = task_type
    frame['fault_type'] = fault
    frame['y_true'] = pd.to_numeric(samples_df[y_column], errors='coerce').fillna(0).astype(int).to_numpy()
    frame['y_pred'] = (score_array >= float(threshold)).astype(int)
    frame['y_score'] = score_array
    frame['threshold'] = float(threshold)
    return frame


def _select_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    if len(y_true) == 0:
        return np.full(y_score.shape[1], 0.5, dtype=np.float32)
    return np.asarray(
        [find_best_threshold(y_true[:, fault_idx], y_score[:, fault_idx]) for fault_idx in range(y_score.shape[1])],
        dtype=np.float32,
    )


def _write_progress(
    progress_paths: list[Path],
    ablation: str,
    stage: str,
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, float] | None,
    val_metrics: dict[str, float] | None,
    split_sizes: dict[str, int],
    status: str,
    thresholds_note: str | None = None,
) -> None:
    lines = [
        '# Dual Task Progress',
        '',
        f'- ablation: {ablation}',
        f'- status: {status}',
        f'- stage: {stage}',
        f'- epoch: {epoch}/{total_epochs}',
        f"- train_samples: {split_sizes.get('train', 0)}",
        f"- val_samples: {split_sizes.get('val', 0)}",
        f"- test_samples: {split_sizes.get('test', 0)}",
    ]
    if train_metrics is not None:
        lines.extend(
            [
                f"- train_loss: {train_metrics.get('loss', 0.0):.6f}",
                f"- train_id_loss: {train_metrics.get('id_loss', 0.0):.6f}",
                f"- train_warn_loss: {train_metrics.get('warn_loss', 0.0):.6f}",
            ]
        )
    if val_metrics is not None:
        lines.extend(
            [
                f"- val_loss: {val_metrics.get('loss', 0.0):.6f}",
                f"- val_id_loss: {val_metrics.get('id_loss', 0.0):.6f}",
                f"- val_warn_loss: {val_metrics.get('warn_loss', 0.0):.6f}",
            ]
        )
    if thresholds_note:
        lines.append(f'- thresholds: {thresholds_note}')
    content = '\n'.join(lines) + '\n'
    for path in progress_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8-sig')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the shared-encoder dual-task model.')
    parser.add_argument('--samples-root', default=str(SAMPLES_ROOT))
    parser.add_argument('--train-samples-root', default=None)
    parser.add_argument('--eval-samples-root', default=None)
    parser.add_argument('--results-root', default=str(RESULTS_ROOT))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--ablation', choices=ABLATION_CHOICES, default='none')
    parser.add_argument('--epochs-stage1', type=int, default=5)
    parser.add_argument('--epochs-joint', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--lambda-warn', type=float, default=0.3)
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
    print(f'[dual-task] loading ablation={args.ablation} train_root={train_samples_root} eval_root={eval_samples_root}', flush=True)
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
    split_sizes = {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)}
    print(
        f"[dual-task] prepared splits train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}",
        flush=True,
    )

    device = _resolve_device(args.device)
    pin_memory = device.type == 'cuda'
    train_loader = create_dataloader(train_bundle, train_idx, args.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = create_dataloader(train_bundle, val_idx, args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = create_dataloader(eval_bundle, test_idx, args.batch_size, shuffle=False, pin_memory=pin_memory)

    use_fault_specific_features = args.ablation != 'no_fault_specific_features'
    use_expert_heads = args.ablation != 'no_expert_heads'
    effective_lambda_warn = 0.0 if args.ablation == 'no_warning_task' else args.lambda_warn
    effective_joint_epochs = 0 if args.ablation == 'no_warning_task' else args.epochs_joint

    model = MultiFaultDualTaskModel(
        sequence_input_dim=train_bundle.x_seq.shape[2],
        feature_columns=train_bundle.feature_columns,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=train_bundle.x_seq.shape[1],
        use_fault_specific_features=use_fault_specific_features,
        use_expert_heads=use_expert_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    id_pos_weight = compute_multilabel_pos_weight(train_bundle.y_id[train_idx]) if len(train_idx) else np.ones(5, dtype=np.float32)
    warn_pos_weight = compute_multilabel_pos_weight(train_bundle.y_warn[train_idx]) if len(train_idx) else np.ones(5, dtype=np.float32)

    if args.ablation == 'none':
        if args.eval_split == 'test':
            recognition_dir = results_root / 'recognition' / 'main_dual_task'
            warning_dir = results_root / 'warning' / 'main_dual_task'
        else:
            recognition_dir = results_root / 'external_validation' / 'recognition' / 'main_dual_task'
            warning_dir = results_root / 'external_validation' / 'warning' / 'main_dual_task'
        recognition_dir.mkdir(parents=True, exist_ok=True)
        warning_dir.mkdir(parents=True, exist_ok=True)
        progress_paths = [recognition_dir / 'progress.md', warning_dir / 'progress.md']
    else:
        if args.eval_split == 'test':
            ablation_dir = results_root / 'ablation' / args.ablation
        else:
            ablation_dir = results_root / 'external_validation' / 'ablation' / args.ablation
        ablation_dir.mkdir(parents=True, exist_ok=True)
        progress_paths = [ablation_dir / 'progress.md']
    total_epochs = args.epochs_stage1 + effective_joint_epochs
    _write_progress(progress_paths, args.ablation, 'stage1', 0, total_epochs, None, None, split_sizes, 'running')

    stage1_history = []
    best_stage1_loss = float('inf')
    best_stage1_state = None
    completed_epochs = 0
    for epoch in range(1, args.epochs_stage1 + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_warn=0.0,
            id_pos_weight=id_pos_weight,
            warn_pos_weight=warn_pos_weight,
        )
        val_metrics = _evaluate_joint_loss(
            model,
            val_loader,
            device,
            lambda_warn=0.0,
            id_pos_weight=id_pos_weight,
            warn_pos_weight=warn_pos_weight,
        ) if len(val_idx) else train_metrics
        stage1_history.append({'stage': 'stage1', 'epoch': epoch, **train_metrics, 'val_loss': val_metrics['loss']})
        if val_metrics['loss'] < best_stage1_loss:
            best_stage1_loss = val_metrics['loss']
            best_stage1_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        completed_epochs += 1
        _write_progress(progress_paths, args.ablation, 'stage1', completed_epochs, total_epochs, train_metrics, val_metrics, split_sizes, 'running')
        print(
            f"[dual-task] stage1 epoch {epoch}/{args.epochs_stage1} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} elapsed={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

    if best_stage1_state is not None:
        model.load_state_dict(best_stage1_state)

    joint_history = []
    best_joint_loss = float('inf')
    best_joint_state = None
    for epoch in range(1, effective_joint_epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_warn=effective_lambda_warn,
            id_pos_weight=id_pos_weight,
            warn_pos_weight=warn_pos_weight,
        )
        val_metrics = _evaluate_joint_loss(
            model,
            val_loader,
            device,
            lambda_warn=effective_lambda_warn,
            id_pos_weight=id_pos_weight,
            warn_pos_weight=warn_pos_weight,
        ) if len(val_idx) else train_metrics
        joint_history.append({'stage': 'joint', 'epoch': epoch, **train_metrics, 'val_loss': val_metrics['loss']})
        if val_metrics['loss'] < best_joint_loss:
            best_joint_loss = val_metrics['loss']
            best_joint_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        completed_epochs += 1
        _write_progress(progress_paths, args.ablation, 'joint', completed_epochs, total_epochs, train_metrics, val_metrics, split_sizes, 'running')
        print(
            f"[dual-task] joint epoch {epoch}/{effective_joint_epochs} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} elapsed={time.perf_counter() - epoch_start:.1f}s",
            flush=True,
        )

    if best_joint_state is None and best_stage1_state is not None:
        best_joint_state = best_stage1_state
    if best_joint_state is not None:
        model.load_state_dict(best_joint_state)
    print(f'[dual-task] best checkpoint restored ablation={args.ablation}', flush=True)

    _write_progress(progress_paths, args.ablation, 'post_training', total_epochs, total_epochs, joint_history[-1] if joint_history else (stage1_history[-1] if stage1_history else None), None, split_sizes, 'scoring_validation')
    print(f'[dual-task] scoring validation set for threshold selection ablation={args.ablation}', flush=True)
    val_predictions = predict_scores(model, val_loader, device) if len(val_idx) else {
        'id_scores': np.zeros((0, 5), dtype=np.float32),
        'warn_scores': np.zeros((0, 5), dtype=np.float32),
        'y_id': np.zeros((0, 5), dtype=np.float32),
        'y_warn': np.zeros((0, 5), dtype=np.float32),
        'sample_id': np.asarray([], dtype=str),
    }
    _write_progress(progress_paths, args.ablation, 'post_training', total_epochs, total_epochs, joint_history[-1] if joint_history else (stage1_history[-1] if stage1_history else None), None, split_sizes, 'selecting_thresholds')
    print(f'[dual-task] selecting validation thresholds ablation={args.ablation}', flush=True)
    id_thresholds = _select_thresholds(val_predictions['y_id'], val_predictions['id_scores']) if len(val_idx) else np.full(5, 0.5, dtype=np.float32)
    warn_thresholds = _select_thresholds(val_predictions['y_warn'], val_predictions['warn_scores']) if len(val_idx) else np.full(5, 0.5, dtype=np.float32)

    _write_progress(progress_paths, args.ablation, 'post_training', total_epochs, total_epochs, joint_history[-1] if joint_history else (stage1_history[-1] if stage1_history else None), None, split_sizes, 'scoring_test')
    print(f'[dual-task] scoring test set ablation={args.ablation}', flush=True)
    predictions = predict_scores(model, test_loader, device)
    test_samples = eval_bundle.samples_master.iloc[test_idx].reset_index(drop=True)
    recognition_rows = []
    warning_rows = []
    recognition_points = []
    warning_points = []

    for row in stage1_history + joint_history:
        recognition_points.append({'plot_type': 'loss_curve', 'series': f"{row['stage']}_train_loss", 'x': row['epoch'], 'y': row['loss']})
        recognition_points.append({'plot_type': 'loss_curve', 'series': f"{row['stage']}_val_loss", 'x': row['epoch'], 'y': row['val_loss']})
        warning_points.append({'plot_type': 'loss_curve', 'series': f"{row['stage']}_warn_loss", 'x': row['epoch'], 'y': row['warn_loss']})

    lead_times = test_samples['lead_time_sec'].to_numpy(dtype=object)
    model_name = 'shared_encoder_expert_heads' if args.ablation == 'none' else f'shared_encoder_{args.ablation}'
    recognition_prediction_frames = []
    warning_prediction_frames = []
    for fault_index, fault in enumerate(FAULT_ORDER):
        id_threshold = float(id_thresholds[fault_index])
        warn_threshold = float(warn_thresholds[fault_index])
        id_metrics = compute_binary_metrics(predictions['y_id'][:, fault_index], predictions['id_scores'][:, fault_index], threshold=id_threshold)
        warn_metrics = compute_warning_metrics(
            predictions['y_warn'][:, fault_index],
            predictions['warn_scores'][:, fault_index],
            lead_times=lead_times,
            threshold=warn_threshold,
        )
        recognition_rows.append(
            build_summary_row(
                experiment_name='dual_task_main' if args.ablation == 'none' else 'dual_task_ablation',
                model_name=model_name,
                task_type='identification',
                fault_type=fault,
                split=args.eval_split,
                metrics=id_metrics,
            )
        )
        warning_rows.append(
            build_summary_row(
                experiment_name='dual_task_main' if args.ablation == 'none' else 'dual_task_ablation',
                model_name=model_name,
                task_type='warning',
                fault_type=fault,
                split=args.eval_split,
                metrics=warn_metrics,
            )
        )
        recognition_points.extend(
            [
                {'plot_type': 'bar_metric', 'category': fault, 'series': 'f1', 'value': id_metrics['f1']},
                {'plot_type': 'bar_metric', 'category': fault, 'series': 'recall', 'value': id_metrics['recall']},
                {'plot_type': 'threshold', 'category': fault, 'series': 'threshold', 'value': id_threshold},
            ]
        )
        warning_points.extend(
            [
                {'plot_type': 'bar_metric', 'category': fault, 'series': 'warning_f1', 'value': warn_metrics['warning_f1']},
                {'plot_type': 'bar_metric', 'category': fault, 'series': 'warning_recall', 'value': warn_metrics['warning_recall']},
                {'plot_type': 'threshold', 'category': fault, 'series': 'threshold', 'value': warn_threshold},
            ]
        )
        recognition_prediction_frames.append(
            _build_prediction_frame(
                samples_df=test_samples,
                y_column=f'y_id_{fault}',
                fault=fault,
                task_type='identification',
                model_name=model_name,
                scores=predictions['id_scores'][:, fault_index],
                threshold=id_threshold,
            )
        )
        warning_prediction_frames.append(
            _build_prediction_frame(
                samples_df=test_samples,
                y_column=f'y_warn_{fault}',
                fault=fault,
                task_type='warning',
                model_name=model_name,
                scores=predictions['warn_scores'][:, fault_index],
                threshold=warn_threshold,
            )
        )

    recognition_predictions = pd.concat(recognition_prediction_frames, ignore_index=True)
    warning_predictions = pd.concat(warning_prediction_frames, ignore_index=True)

    threshold_note = 'validation thresholds=' + ', '.join(f'{fault}:{id_thresholds[idx]:.3f}/{warn_thresholds[idx]:.3f}' for idx, fault in enumerate(FAULT_ORDER))

    _write_progress(progress_paths, args.ablation, 'post_training', total_epochs, total_epochs, joint_history[-1] if joint_history else (stage1_history[-1] if stage1_history else None), None, split_sizes, 'writing_results', thresholds_note=threshold_note)
    print(f'[dual-task] writing result bundle ablation={args.ablation}', flush=True)

    if args.ablation == 'none':
        if best_stage1_state is not None:
            torch.save(best_stage1_state, recognition_dir / 'best_id_model.pt')
        if best_joint_state is not None:
            torch.save(best_joint_state, warning_dir / 'best_joint_model.pt')
        write_result_bundle(
            output_dir=recognition_dir,
            summary=pd.DataFrame(recognition_rows),
            predictions=recognition_predictions,
            data_points=pd.DataFrame(recognition_points),
            notes='Stage 1 recognition training plus joint fine-tuning recognition outputs.\n' + threshold_note + f'\ntrain_samples_root={train_samples_root}\neval_samples_root={eval_samples_root}',
        )
        write_result_bundle(
            output_dir=warning_dir,
            summary=pd.DataFrame(warning_rows),
            predictions=warning_predictions,
            data_points=pd.DataFrame(warning_points),
            notes='Joint fine-tuning warning outputs with lambda_warn weighting.\n' + threshold_note + f'\ntrain_samples_root={train_samples_root}\neval_samples_root={eval_samples_root}',
        )
    else:
        if best_stage1_state is not None:
            torch.save(best_stage1_state, ablation_dir / 'best_id_model.pt')
        if best_joint_state is not None:
            torch.save(best_joint_state, ablation_dir / 'best_joint_model.pt')
        ablation_summary = pd.concat([pd.DataFrame(recognition_rows), pd.DataFrame(warning_rows)], ignore_index=True)
        ablation_predictions = pd.concat([recognition_predictions, warning_predictions], ignore_index=True)
        ablation_points = pd.concat([pd.DataFrame(recognition_points), pd.DataFrame(warning_points)], ignore_index=True)
        notes = '\n'.join(
            [
                f'Ablation run: {args.ablation}.',
                f'effective_lambda_warn={effective_lambda_warn}',
                f'use_fault_specific_features={use_fault_specific_features}',
                f'use_expert_heads={use_expert_heads}',
                threshold_note,
                'For no_label_quality_control, use a sample bundle built with raw labels and ignored quality flags.',
                f'train_samples_root={train_samples_root}',
                f'eval_samples_root={eval_samples_root}',
            ]
        )
        write_result_bundle(
            output_dir=ablation_dir,
            summary=ablation_summary,
            predictions=ablation_predictions,
            data_points=ablation_points,
            notes=notes,
        )

    _write_progress(progress_paths, args.ablation, 'completed', total_epochs, total_epochs, joint_history[-1] if joint_history else (stage1_history[-1] if stage1_history else None), None, split_sizes, 'completed', thresholds_note=threshold_note)
    print(f'[dual-task] finished ablation={args.ablation} total_elapsed={time.perf_counter() - run_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()

