from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score



def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'pr_auc': _safe_metric(average_precision_score, y_true, y_score),
        'roc_auc': _safe_metric(roc_auc_score, y_true, y_score),
    }
    return metrics



def compute_multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
    }



def compute_warning_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    lead_times: Iterable[object] | None = None,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    base = compute_binary_metrics(y_true, y_score, threshold=threshold)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    negatives = max(int((y_true == 0).sum()), 1)
    false_alarms = int(((y_pred == 1) & (y_true == 0)).sum())
    avg_lead_time = None
    if lead_times is not None:
        lead_array = np.asarray(list(lead_times), dtype=object)
        valid = []
        for index, value in enumerate(lead_array):
            if y_pred[index] == 1 and y_true[index] == 1 and value is not None:
                valid.append(float(value))
        if valid:
            avg_lead_time = float(np.mean(valid))
    return {
        'warning_f1': base['f1'],
        'warning_recall': base['recall'],
        'false_alarm_rate': float(false_alarms / negatives),
        'avg_lead_time': avg_lead_time,
    }



def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    candidates: Sequence[float] | None = None,
    default: float = 0.5,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    finite_mask = np.isfinite(y_score)
    if finite_mask.sum() == 0:
        return float(default)

    y_true = y_true[finite_mask]
    y_score = y_score[finite_mask]
    if np.unique(y_true).size < 2:
        return float(default)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size:
        curve_thresholds = thresholds.astype(float, copy=False)
        curve_precision = precision[:-1].astype(float, copy=False)
        curve_recall = recall[:-1].astype(float, copy=False)
        curve_f1 = np.divide(
            2.0 * curve_precision * curve_recall,
            curve_precision + curve_recall,
            out=np.zeros_like(curve_precision, dtype=float),
            where=(curve_precision + curve_recall) > 0,
        )
    else:
        curve_thresholds = np.asarray([], dtype=float)
        curve_precision = np.asarray([], dtype=float)
        curve_recall = np.asarray([], dtype=float)
        curve_f1 = np.asarray([], dtype=float)

    extra_thresholds: list[float] = []
    if 0.0 < float(default) < 1.0:
        extra_thresholds.append(float(default))
    if candidates is not None:
        for candidate in candidates:
            candidate_value = float(candidate)
            if 0.0 < candidate_value < 1.0:
                extra_thresholds.append(candidate_value)

    if extra_thresholds:
        extras = np.asarray(sorted(set(extra_thresholds)), dtype=float)
        extras_precision, extras_recall = _precision_recall_at_thresholds(y_true, y_score, extras)
        extras_f1 = np.divide(
            2.0 * extras_precision * extras_recall,
            extras_precision + extras_recall,
            out=np.zeros_like(extras_precision, dtype=float),
            where=(extras_precision + extras_recall) > 0,
        )
        all_thresholds = np.concatenate([curve_thresholds, extras])
        all_precision = np.concatenate([curve_precision, extras_precision])
        all_recall = np.concatenate([curve_recall, extras_recall])
        all_f1 = np.concatenate([curve_f1, extras_f1])
    else:
        all_thresholds = curve_thresholds
        all_precision = curve_precision
        all_recall = curve_recall
        all_f1 = curve_f1

    if all_thresholds.size == 0:
        return float(default)

    order = np.lexsort((all_thresholds, -all_precision, -all_recall, -all_f1))
    return float(all_thresholds[int(order[0])])



def _precision_recall_at_thresholds(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if thresholds.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    order = np.argsort(y_score, kind='mergesort')
    sorted_scores = y_score[order]
    sorted_true = y_true[order].astype(np.int64, copy=False)
    total_positives = int(sorted_true.sum())
    if total_positives <= 0:
        return np.zeros(thresholds.shape[0], dtype=float), np.zeros(thresholds.shape[0], dtype=float)

    suffix_tp = np.cumsum(sorted_true[::-1], dtype=np.int64)[::-1]
    start_idx = np.searchsorted(sorted_scores, thresholds, side='left')
    positives_pred = sorted_true.shape[0] - start_idx

    tp = np.zeros_like(start_idx, dtype=np.int64)
    valid_mask = start_idx < sorted_true.shape[0]
    if np.any(valid_mask):
        tp[valid_mask] = suffix_tp[start_idx[valid_mask]]
    fp = positives_pred - tp
    fn = total_positives - tp

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(thresholds, dtype=float),
        where=(tp + fp) > 0,
    )
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(thresholds, dtype=float),
        where=(tp + fn) > 0,
    )
    return precision, recall



def _safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None

