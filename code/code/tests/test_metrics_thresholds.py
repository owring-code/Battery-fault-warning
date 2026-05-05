import numpy as np
import pytest

import battery_thesis.metrics as metrics_module


def test_find_best_threshold_avoids_python_metric_loop_on_large_validation_set(monkeypatch: pytest.MonkeyPatch):
    y_true = np.tile(np.asarray([0, 1], dtype=int), 600)
    y_score = np.linspace(0.001, 0.999, y_true.shape[0], dtype=float)

    original = metrics_module.compute_binary_metrics
    call_count = 0

    def counted_compute_binary_metrics(y_true_arg, y_score_arg, threshold=0.5):
        nonlocal call_count
        call_count += 1
        return original(y_true_arg, y_score_arg, threshold=threshold)

    monkeypatch.setattr(metrics_module, 'compute_binary_metrics', counted_compute_binary_metrics)

    best_threshold = metrics_module.find_best_threshold(y_true, y_score)

    assert 0.0 < best_threshold < 1.0
    assert call_count <= 3


def test_find_best_threshold_handles_default_threshold_above_all_scores():
    y_true = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
    y_score = np.asarray([0.02, 0.07, 0.11, 0.15, 0.19, 0.23], dtype=float)

    best_threshold = metrics_module.find_best_threshold(y_true, y_score, default=0.5)

    assert 0.0 < best_threshold < 1.0
