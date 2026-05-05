from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.run_dual_task_model as dual_runner
import scripts.run_sequence_baseline as sequence_runner


def test_sequence_build_prediction_frame_preserves_order_and_threshold():
    test_samples = pd.DataFrame(
        {
            'sample_id': ['s1', 's2'],
            'vehicle_id': ['v1', 'v2'],
            'source_dataset': ['structured_dataset', 'structured_dataset'],
            'y_id_sd': [0, 1],
        }
    )

    frame = sequence_runner._build_prediction_frame(
        test_df=test_samples,
        y_column='y_id_sd',
        fault='sd',
        task_type='identification',
        model_name='lstm',
        scores=np.asarray([0.2, 0.8], dtype=float),
        threshold=0.5,
    )

    assert frame['sample_id'].tolist() == ['s1', 's2']
    assert frame['model_name'].tolist() == ['lstm', 'lstm']
    assert frame['y_pred'].tolist() == [0, 1]
    assert frame['threshold'].tolist() == [0.5, 0.5]


def test_dual_task_build_prediction_frame_preserves_order_and_threshold():
    test_samples = pd.DataFrame(
        {
            'sample_id': ['s1', 's2'],
            'vehicle_id': ['v1', 'v2'],
            'source_dataset': ['structured_dataset', 'structured_dataset'],
            'y_warn_sd': [0, 1],
        }
    )

    frame = dual_runner._build_prediction_frame(
        samples_df=test_samples,
        y_column='y_warn_sd',
        fault='sd',
        task_type='warning',
        model_name='shared_encoder_expert_heads',
        scores=np.asarray([0.1, 0.9], dtype=float),
        threshold=0.6,
    )

    assert frame['sample_id'].tolist() == ['s1', 's2']
    assert frame['fault_type'].tolist() == ['sd', 'sd']
    assert frame['y_pred'].tolist() == [0, 1]
    assert frame['threshold'].tolist() == [0.6, 0.6]
