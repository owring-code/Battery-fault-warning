import numpy as np
import pandas as pd
import torch

from battery_thesis.training import (
    LoadedTensorBundle,
    compute_joint_loss,
    compute_multilabel_pos_weight,
    load_tensor_bundle,
    normalize_bundle_with_reference_stats,
    normalize_bundle_with_train_stats,
)


def test_compute_joint_loss_respects_warning_weight():
    id_logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    warn_logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    y_id = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    y_warn = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    outputs = compute_joint_loss(id_logits, warn_logits, y_id, y_warn, lambda_warn=0.3)

    assert outputs['loss'].item() > outputs['id_loss']
    assert outputs['warn_loss'] > 0
    assert round(outputs['loss'].item(), 6) == round(outputs['id_loss'] + 0.3 * outputs['warn_loss'], 6)


def test_compute_joint_loss_supports_positive_class_weighting():
    id_logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    warn_logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    y_id = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    y_warn = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    unweighted = compute_joint_loss(id_logits, warn_logits, y_id, y_warn, lambda_warn=0.0)
    weighted = compute_joint_loss(
        id_logits,
        warn_logits,
        y_id,
        y_warn,
        lambda_warn=0.0,
        id_pos_weight=torch.tensor([5.0, 1.0], dtype=torch.float32),
    )

    assert weighted['id_loss'] > unweighted['id_loss']
    assert weighted['loss'].item() > unweighted['loss'].item()


def test_compute_multilabel_pos_weight_clamps_sparse_labels():
    targets = np.asarray(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )

    weights = compute_multilabel_pos_weight(targets, max_weight=10.0)

    assert weights.shape == (3,)
    assert weights[0] > 1.0
    assert weights[1] > 1.0
    assert weights[2] == 1.0


def test_load_tensor_bundle_sanitizes_non_finite_arrays(tmp_path):
    samples_path = tmp_path / 'samples_master.csv'
    dataset_pack_path = tmp_path / 'dataset_pack.npz'

    pd.DataFrame([{'sample_id': 's1', 'split': 'train'}]).to_csv(samples_path, index=False, encoding='utf-8-sig')
    np.savez_compressed(
        dataset_pack_path,
        X_seq=np.array([[[1.0, np.nan], [np.inf, 2.0]]], dtype=np.float32),
        X_feat=np.array([[np.nan, np.inf, -np.inf]], dtype=np.float32),
        y_id=np.zeros((1, 5), dtype=np.int64),
        y_warn=np.zeros((1, 5), dtype=np.int64),
        sample_id=np.array(['s1']),
        feature_columns=np.array(['shared_a', 'sd_b', 'ins_c']),
        sequence_feature_columns=np.array(['f1', 'f2']),
    )

    bundle = load_tensor_bundle(samples_path, dataset_pack_path)

    assert not np.isnan(bundle.x_seq).any()
    assert not np.isinf(bundle.x_seq).any()
    assert not np.isnan(bundle.x_feat).any()
    assert not np.isinf(bundle.x_feat).any()


def test_normalize_bundle_with_train_stats_uses_train_split_only():
    bundle = LoadedTensorBundle(
        samples_master=pd.DataFrame(
            [
                {'sample_id': 's1', 'split': 'train'},
                {'sample_id': 's2', 'split': 'train'},
                {'sample_id': 's3', 'split': 'val'},
            ]
        ),
        x_seq=np.array(
            [
                [[1.0, 10.0], [3.0, 10.0]],
                [[5.0, 10.0], [7.0, 10.0]],
                [[9.0, np.inf], [11.0, np.nan]],
            ],
            dtype=np.float32,
        ),
        x_feat=np.array(
            [
                [2.0, 100.0],
                [4.0, 100.0],
                [6.0, np.nan],
            ],
            dtype=np.float32,
        ),
        y_id=np.zeros((3, 5), dtype=np.int64),
        y_warn=np.zeros((3, 5), dtype=np.int64),
        sample_ids=np.array(['s1', 's2', 's3']),
        feature_columns=np.array(['shared_a', 'shared_b']),
        sequence_feature_columns=np.array(['f1', 'f2']),
    )

    normalized = normalize_bundle_with_train_stats(bundle, np.array([0, 1], dtype=int))

    train_seq_feature = normalized.x_seq[:2, :, 0]
    train_feat_feature = normalized.x_feat[:2, 0]
    expected_val_seq = (np.array([9.0, 11.0], dtype=np.float32) - 4.0) / np.std(
        np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    )

    assert normalized.x_seq.dtype == np.float32
    assert normalized.x_feat.dtype == np.float32
    assert np.isfinite(normalized.x_seq).all()
    assert np.isfinite(normalized.x_feat).all()
    assert np.isclose(train_seq_feature.mean(), 0.0)
    assert np.isclose(train_seq_feature.std(), 1.0)
    assert np.isclose(train_feat_feature.mean(), 0.0)
    assert np.isclose(train_feat_feature.std(), 1.0)
    assert np.allclose(normalized.x_seq[2, :, 0], expected_val_seq, atol=1e-6)
    assert np.allclose(normalized.x_seq[:2, :, 1], 0.0, atol=1e-6)
    assert np.allclose(normalized.x_feat[:2, 1], 0.0, atol=1e-6)



def test_load_tensor_bundle_rejects_sample_id_order_mismatch(tmp_path):
    samples_path = tmp_path / 'samples_master.csv'
    dataset_pack_path = tmp_path / 'dataset_pack.npz'

    pd.DataFrame(
        [
            {'sample_id': 's1', 'split': 'train'},
            {'sample_id': 's2', 'split': 'train'},
        ]
    ).to_csv(samples_path, index=False, encoding='utf-8-sig')
    np.savez_compressed(
        dataset_pack_path,
        X_seq=np.zeros((2, 1, 1), dtype=np.float32),
        X_feat=np.zeros((2, 1), dtype=np.float32),
        y_id=np.zeros((2, 5), dtype=np.int64),
        y_warn=np.zeros((2, 5), dtype=np.int64),
        sample_id=np.array(['s2', 's1']),
        feature_columns=np.array(['shared_a']),
        sequence_feature_columns=np.array(['f1']),
    )

    try:
        load_tensor_bundle(samples_path, dataset_pack_path)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError('Expected load_tensor_bundle to reject mismatched sample id order.')

    assert 'sample_id order mismatch' in message



def test_normalize_bundle_with_reference_stats_uses_reference_train_distribution():
    reference_bundle = LoadedTensorBundle(
        samples_master=pd.DataFrame(
            [
                {'sample_id': 't1', 'split': 'train'},
                {'sample_id': 't2', 'split': 'train'},
                {'sample_id': 'v1', 'split': 'val'},
            ]
        ),
        x_seq=np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ],
            dtype=np.float32,
        ),
        x_feat=np.array(
            [
                [2.0, 20.0],
                [4.0, 40.0],
                [6.0, 60.0],
            ],
            dtype=np.float32,
        ),
        y_id=np.zeros((3, 5), dtype=np.int64),
        y_warn=np.zeros((3, 5), dtype=np.int64),
        sample_ids=np.array(['t1', 't2', 'v1']),
        feature_columns=np.array(['shared_a', 'sd_a']),
        sequence_feature_columns=np.array(['f1', 'f2']),
    )
    eval_bundle = LoadedTensorBundle(
        samples_master=pd.DataFrame(
            [
                {'sample_id': 'e1', 'split': 'external_test'},
                {'sample_id': 'e2', 'split': 'external_test'},
            ]
        ),
        x_seq=np.array(
            [
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ],
            dtype=np.float32,
        ),
        x_feat=np.array(
            [
                [6.0, 60.0],
                [8.0, 80.0],
            ],
            dtype=np.float32,
        ),
        y_id=np.zeros((2, 5), dtype=np.int64),
        y_warn=np.zeros((2, 5), dtype=np.int64),
        sample_ids=np.array(['e1', 'e2']),
        feature_columns=np.array(['shared_a', 'sd_a']),
        sequence_feature_columns=np.array(['f1', 'f2']),
    )

    normalized = normalize_bundle_with_reference_stats(eval_bundle, reference_bundle, np.array([0, 1], dtype=int))

    expected_seq = (eval_bundle.x_seq - np.array([[[4.0, 5.0]]], dtype=np.float32)) / np.array([[[2.236068, 2.236068]]], dtype=np.float32)
    expected_feat = (eval_bundle.x_feat - np.array([[3.0, 30.0]], dtype=np.float32)) / np.array([[1.0, 10.0]], dtype=np.float32)

    assert np.allclose(normalized.x_seq, expected_seq, atol=1e-5)
    assert np.allclose(normalized.x_feat, expected_feat, atol=1e-5)
