from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _sanitize_float_array(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f'Non-finite values detected in {name}.')


def compute_multilabel_pos_weight(
    targets: np.ndarray,
    min_weight: float = 1.0,
    max_weight: float = 100.0,
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.int64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    total = values.shape[0]
    weights = []
    for column_idx in range(values.shape[1]):
        positives = int(values[:, column_idx].sum())
        negatives = total - positives
        if positives <= 0 or negatives <= 0:
            weights.append(1.0)
            continue
        ratio = negatives / positives
        weights.append(float(np.clip(ratio, min_weight, max_weight)))
    return np.asarray(weights, dtype=np.float32)


def _prepare_pos_weight(
    pos_weight: torch.Tensor | np.ndarray | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if pos_weight is None:
        return None
    if isinstance(pos_weight, torch.Tensor):
        return pos_weight.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(np.asarray(pos_weight, dtype=np.float32), device=reference.device, dtype=reference.dtype)


def _bce_with_optional_pos_weight(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    prepared = _prepare_pos_weight(pos_weight, logits)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=prepared)


def compute_joint_loss(
    id_logits: torch.Tensor,
    warn_logits: torch.Tensor,
    y_id: torch.Tensor,
    y_warn: torch.Tensor,
    lambda_warn: float = 0.3,
    id_pos_weight: torch.Tensor | np.ndarray | None = None,
    warn_pos_weight: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float | torch.Tensor]:
    id_loss_tensor = _bce_with_optional_pos_weight(id_logits, y_id, pos_weight=id_pos_weight)
    warn_loss_tensor = _bce_with_optional_pos_weight(warn_logits, y_warn, pos_weight=warn_pos_weight)
    total_loss = id_loss_tensor + lambda_warn * warn_loss_tensor
    return {
        'loss': total_loss,
        'id_loss': float(id_loss_tensor.detach().item()),
        'warn_loss': float(warn_loss_tensor.detach().item()),
    }


class TensorBundleDataset(Dataset):
    def __init__(
        self,
        x_seq: np.ndarray,
        x_feat: np.ndarray,
        y_id: np.ndarray,
        y_warn: np.ndarray,
        sample_ids: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.x_seq = torch.from_numpy(_sanitize_float_array(x_seq))
        self.x_feat = torch.from_numpy(_sanitize_float_array(x_feat))
        self.y_id = torch.from_numpy(np.asarray(y_id, dtype=np.float32))
        self.y_warn = torch.from_numpy(np.asarray(y_warn, dtype=np.float32))
        self.sample_ids = np.asarray(sample_ids, dtype=str)
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        index = int(self.indices[item])
        return {
            'x_seq': self.x_seq[index],
            'x_feat': self.x_feat[index],
            'y_id': self.y_id[index],
            'y_warn': self.y_warn[index],
            'sample_id': str(self.sample_ids[index]),
        }


@dataclass
class LoadedTensorBundle:
    samples_master: pd.DataFrame
    x_seq: np.ndarray
    x_feat: np.ndarray
    y_id: np.ndarray
    y_warn: np.ndarray
    sample_ids: np.ndarray
    feature_columns: np.ndarray
    sequence_feature_columns: np.ndarray


def load_tensor_bundle(samples_path: Path, dataset_pack_path: Path) -> LoadedTensorBundle:
    samples_master = pd.read_csv(samples_path)
    bundle = np.load(dataset_pack_path, allow_pickle=True)
    sample_ids = np.asarray(bundle['sample_id']).astype(str)
    expected_sample_ids = samples_master['sample_id'].astype(str).to_numpy()
    if sample_ids.shape[0] != expected_sample_ids.shape[0]:
        raise ValueError(
            f'sample_id length mismatch between samples_master ({expected_sample_ids.shape[0]}) and dataset_pack ({sample_ids.shape[0]}).'
        )
    if not np.array_equal(expected_sample_ids, sample_ids):
        raise ValueError('sample_id order mismatch between samples_master.csv and dataset_pack.npz; rebuild the sample bundle.')
    return LoadedTensorBundle(
        samples_master=samples_master,
        x_seq=_sanitize_float_array(bundle['X_seq']),
        x_feat=_sanitize_float_array(bundle['X_feat']),
        y_id=np.asarray(bundle['y_id'], dtype=np.int64),
        y_warn=np.asarray(bundle['y_warn'], dtype=np.int64),
        sample_ids=sample_ids,
        feature_columns=bundle['feature_columns'],
        sequence_feature_columns=bundle['sequence_feature_columns'],
    )


def _normalize_with_reference_stats(values: np.ndarray, reference: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    sanitized_values = _sanitize_float_array(values)
    if sanitized_values.size == 0:
        return sanitized_values.astype(np.float32, copy=False)
    sanitized_reference = _sanitize_float_array(reference)
    if sanitized_reference.size == 0:
        return sanitized_values.astype(np.float32, copy=False)
    mean = np.asarray(sanitized_reference.mean(axis=axes, keepdims=True, dtype=np.float64), dtype=np.float32)
    std = np.asarray(sanitized_reference.std(axis=axes, keepdims=True, dtype=np.float64), dtype=np.float32)
    mean = _sanitize_float_array(mean)
    std = _sanitize_float_array(std)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    return ((sanitized_values - mean) / std).astype(np.float32, copy=False)


def normalize_bundle_with_train_stats(bundle: LoadedTensorBundle, train_indices: np.ndarray) -> LoadedTensorBundle:
    train_indices = np.asarray(train_indices, dtype=int)
    normalized_x_seq = _sanitize_float_array(bundle.x_seq)
    normalized_x_feat = _sanitize_float_array(bundle.x_feat)
    if train_indices.size == 0:
        return replace(bundle, x_seq=normalized_x_seq, x_feat=normalized_x_feat)

    normalized_x_seq = _normalize_with_reference_stats(normalized_x_seq, normalized_x_seq[train_indices], axes=(0, 1))
    normalized_x_feat = _normalize_with_reference_stats(normalized_x_feat, normalized_x_feat[train_indices], axes=(0,))
    return replace(bundle, x_seq=normalized_x_seq, x_feat=normalized_x_feat)


def normalize_bundle_with_reference_stats(
    bundle: LoadedTensorBundle,
    reference_bundle: LoadedTensorBundle,
    reference_train_indices: np.ndarray,
) -> LoadedTensorBundle:
    reference_train_indices = np.asarray(reference_train_indices, dtype=int)
    normalized_x_seq = _sanitize_float_array(bundle.x_seq)
    normalized_x_feat = _sanitize_float_array(bundle.x_feat)
    if reference_train_indices.size == 0:
        return replace(bundle, x_seq=normalized_x_seq, x_feat=normalized_x_feat)

    normalized_x_seq = _normalize_with_reference_stats(
        normalized_x_seq,
        reference_bundle.x_seq[reference_train_indices],
        axes=(0, 1),
    )
    normalized_x_feat = _normalize_with_reference_stats(
        normalized_x_feat,
        reference_bundle.x_feat[reference_train_indices],
        axes=(0,),
    )
    return replace(bundle, x_seq=normalized_x_seq, x_feat=normalized_x_feat)


def _align_2d_columns(values: np.ndarray, current_columns: np.ndarray, reference_columns: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((values.shape[0], len(reference_columns)), dtype=np.float32)
    current_map = {str(column): index for index, column in enumerate(np.asarray(current_columns, dtype=str))}
    aligned = np.zeros((values.shape[0], len(reference_columns)), dtype=np.float32)
    for target_index, column in enumerate(np.asarray(reference_columns, dtype=str)):
        source_index = current_map.get(str(column))
        if source_index is not None:
            aligned[:, target_index] = values[:, source_index]
    return _sanitize_float_array(aligned)


def _align_3d_columns(values: np.ndarray, current_columns: np.ndarray, reference_columns: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((values.shape[0], values.shape[1], len(reference_columns)), dtype=np.float32)
    current_map = {str(column): index for index, column in enumerate(np.asarray(current_columns, dtype=str))}
    aligned = np.zeros((values.shape[0], values.shape[1], len(reference_columns)), dtype=np.float32)
    for target_index, column in enumerate(np.asarray(reference_columns, dtype=str)):
        source_index = current_map.get(str(column))
        if source_index is not None:
            aligned[:, :, target_index] = values[:, :, source_index]
    return _sanitize_float_array(aligned)


def align_bundle_to_reference_columns(
    bundle: LoadedTensorBundle,
    reference_feature_columns: np.ndarray,
    reference_sequence_feature_columns: np.ndarray,
) -> LoadedTensorBundle:
    feature_columns = np.asarray(reference_feature_columns, dtype=str)
    sequence_feature_columns = np.asarray(reference_sequence_feature_columns, dtype=str)
    aligned_x_feat = _align_2d_columns(bundle.x_feat, bundle.feature_columns, feature_columns)
    aligned_x_seq = _align_3d_columns(bundle.x_seq, bundle.sequence_feature_columns, sequence_feature_columns)
    return replace(
        bundle,
        x_seq=aligned_x_seq,
        x_feat=aligned_x_feat,
        feature_columns=feature_columns,
        sequence_feature_columns=sequence_feature_columns,
    )


def build_split_indices(samples_master: pd.DataFrame, split_name: str) -> np.ndarray:
    return samples_master.index[samples_master['split'] == split_name].to_numpy(dtype=int)


def create_dataloader(
    bundle: LoadedTensorBundle,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = TensorBundleDataset(
        x_seq=bundle.x_seq,
        x_feat=bundle.x_feat,
        y_id=bundle.y_id,
        y_warn=bundle.y_warn,
        sample_ids=bundle.sample_ids,
        indices=indices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_warn: float,
    id_pos_weight: torch.Tensor | np.ndarray | None = None,
    warn_pos_weight: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_id_loss = 0.0
    total_warn_loss = 0.0
    batch_count = 0
    id_pos_weight_tensor = None if id_pos_weight is None else torch.as_tensor(id_pos_weight, dtype=torch.float32, device=device)
    warn_pos_weight_tensor = None if warn_pos_weight is None else torch.as_tensor(warn_pos_weight, dtype=torch.float32, device=device)
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        x_seq = batch['x_seq'].to(device, non_blocking=True)
        x_feat = batch['x_feat'].to(device, non_blocking=True)
        y_id = batch['y_id'].to(device, non_blocking=True)
        y_warn = batch['y_warn'].to(device, non_blocking=True)
        _ensure_finite_tensor('x_seq', x_seq)
        _ensure_finite_tensor('x_feat', x_feat)
        outputs = model(x_seq, x_feat)
        _ensure_finite_tensor('id_logits', outputs['id_logits'])
        _ensure_finite_tensor('warn_logits', outputs['warn_logits'])
        losses = compute_joint_loss(
            outputs['id_logits'],
            outputs['warn_logits'],
            y_id,
            y_warn,
            lambda_warn=lambda_warn,
            id_pos_weight=id_pos_weight_tensor,
            warn_pos_weight=warn_pos_weight_tensor,
        )
        _ensure_finite_tensor('loss', losses['loss'])
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
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


def predict_scores(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    id_scores = []
    warn_scores = []
    y_id = []
    y_warn = []
    sample_ids = []
    with torch.no_grad():
        for batch in dataloader:
            x_seq = batch['x_seq'].to(device, non_blocking=True)
            x_feat = batch['x_feat'].to(device, non_blocking=True)
            _ensure_finite_tensor('x_seq', x_seq)
            _ensure_finite_tensor('x_feat', x_feat)
            outputs = model(x_seq, x_feat)
            _ensure_finite_tensor('id_logits', outputs['id_logits'])
            _ensure_finite_tensor('warn_logits', outputs['warn_logits'])
            id_scores.append(torch.sigmoid(outputs['id_logits']).cpu().numpy())
            warn_scores.append(torch.sigmoid(outputs['warn_logits']).cpu().numpy())
            y_id.append(batch['y_id'].cpu().numpy())
            y_warn.append(batch['y_warn'].cpu().numpy())
            sample_ids.extend(batch['sample_id'])
    return {
        'id_scores': np.concatenate(id_scores, axis=0) if id_scores else np.zeros((0, 5), dtype=np.float32),
        'warn_scores': np.concatenate(warn_scores, axis=0) if warn_scores else np.zeros((0, 5), dtype=np.float32),
        'y_id': np.concatenate(y_id, axis=0) if y_id else np.zeros((0, 5), dtype=np.float32),
        'y_warn': np.concatenate(y_warn, axis=0) if y_warn else np.zeros((0, 5), dtype=np.float32),
        'sample_id': np.asarray(sample_ids, dtype=str),
    }
