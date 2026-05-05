import numpy as np
import torch

from battery_thesis.feature_catalog import FEATURE_ROWS
from battery_thesis.metrics import compute_binary_metrics, compute_multilabel_metrics, compute_warning_metrics, find_best_threshold
from battery_thesis.models import MultiFaultDualTaskModel, SequenceClassificationBaseline


def test_multifault_dual_task_model_emits_two_5d_heads():
    feature_columns = np.asarray([row[0] for row in FEATURE_ROWS], dtype=str)
    model = MultiFaultDualTaskModel(
        sequence_input_dim=8,
        feature_columns=feature_columns,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.1,
    )
    x_seq = torch.randn(4, 30, 8)
    x_feat = torch.randn(4, len(feature_columns))

    outputs = model(x_seq, x_feat)

    assert outputs['id_logits'].shape == (4, 5)
    assert outputs['warn_logits'].shape == (4, 5)


def test_multifault_model_supports_shared_heads_ablation():
    feature_columns = np.asarray([row[0] for row in FEATURE_ROWS], dtype=str)
    model = MultiFaultDualTaskModel(
        sequence_input_dim=8,
        feature_columns=feature_columns,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.1,
        use_expert_heads=False,
    )
    x_seq = torch.randn(2, 30, 8)
    x_feat = torch.randn(2, len(feature_columns))

    outputs = model(x_seq, x_feat)

    assert outputs['id_logits'].shape == (2, 5)
    assert outputs['warn_logits'].shape == (2, 5)


def test_metric_helpers_cover_binary_multilabel_and_warning_views():
    y_true = np.asarray([0, 1, 1, 0])
    y_score = np.asarray([0.1, 0.9, 0.8, 0.2])
    binary = compute_binary_metrics(y_true, y_score)
    warning = compute_warning_metrics(y_true, y_score, lead_times=np.asarray([None, 20, 30, None], dtype=object))

    multilabel_true = np.asarray([[1, 0], [0, 1], [1, 1]])
    multilabel_score = np.asarray([[0.9, 0.2], [0.1, 0.8], [0.7, 0.7]])
    multilabel = compute_multilabel_metrics(multilabel_true, multilabel_score)

    assert binary['f1'] == 1.0
    assert binary['recall'] == 1.0
    assert warning['warning_f1'] == 1.0
    assert warning['avg_lead_time'] == 25.0
    assert multilabel['macro_f1'] == 1.0
    assert multilabel['micro_f1'] == 1.0


def test_find_best_threshold_uses_validation_scores_to_improve_f1():
    y_true = np.asarray([0, 0, 1, 1])
    y_score = np.asarray([0.1, 0.6, 0.65, 0.9])

    best_threshold = find_best_threshold(y_true, y_score)
    metrics_at_best = compute_binary_metrics(y_true, y_score, threshold=best_threshold)
    metrics_at_default = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert 0.0 < best_threshold < 1.0
    assert metrics_at_best['f1'] >= metrics_at_default['f1']


def test_sequence_baseline_emits_multifault_logits():
    model = SequenceClassificationBaseline(
        architecture='lstm',
        sequence_input_dim=8,
        hidden_dim=16,
        output_dim=5,
    )
    x_seq = torch.randn(3, 30, 8)

    logits = model(x_seq)

    assert logits.shape == (3, 5)
