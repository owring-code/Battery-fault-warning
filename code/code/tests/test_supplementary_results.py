from __future__ import annotations

import pandas as pd

from battery_thesis.supplementary_results import (
    build_metric_bar_frame,
    case_rows_to_frame,
    compute_multifault_summary_from_predictions,
    load_warning_prediction_frames,
    select_case_examples,
)


def test_compute_multifault_summary_from_predictions_reports_all_and_core_scopes():
    predictions = pd.DataFrame(
        [
            {'sample_id': 's1', 'fault_type': 'sd', 'y_true': 1, 'y_pred': 1},
            {'sample_id': 's1', 'fault_type': 'samp', 'y_true': 0, 'y_pred': 0},
            {'sample_id': 's1', 'fault_type': 'ins', 'y_true': 0, 'y_pred': 0},
            {'sample_id': 's2', 'fault_type': 'sd', 'y_true': 0, 'y_pred': 0},
            {'sample_id': 's2', 'fault_type': 'samp', 'y_true': 1, 'y_pred': 1},
            {'sample_id': 's2', 'fault_type': 'ins', 'y_true': 0, 'y_pred': 0},
        ]
    )

    summary = compute_multifault_summary_from_predictions({'lstm': predictions})

    assert set(summary['fault_scope']) == {'all_faults', 'core_faults'}
    assert set(summary['model_name']) == {'LSTM'}
    assert summary['macro_f1'].between(0.0, 1.0).all()
    assert summary['micro_f1'].between(0.0, 1.0).all()


def test_select_case_examples_and_case_rows_to_frame_build_case_artifacts():
    samples = pd.DataFrame(
        [
            {'sample_id': 's1', 'vehicle_id': 'v1', 'end_time': 1, 'future_first_fault': '', 'lead_time_sec': ''},
            {'sample_id': 's2', 'vehicle_id': 'v1', 'end_time': 2, 'future_first_fault': '', 'lead_time_sec': ''},
            {'sample_id': 's3', 'vehicle_id': 'v1', 'end_time': 3, 'future_first_fault': '', 'lead_time_sec': ''},
            {'sample_id': 's4', 'vehicle_id': 'v1', 'end_time': 4, 'future_first_fault': '', 'lead_time_sec': ''},
        ]
    )
    predictions = pd.DataFrame(
        [
            {'sample_id': 's1', 'vehicle_id': 'v1', 'fault_type': 'isc', 'y_true': 0, 'y_pred': 0, 'y_score': 0.1, 'threshold': 0.5},
            {'sample_id': 's2', 'vehicle_id': 'v1', 'fault_type': 'isc', 'y_true': 1, 'y_pred': 1, 'y_score': 0.9, 'threshold': 0.5},
            {'sample_id': 's3', 'vehicle_id': 'v1', 'fault_type': 'samp', 'y_true': 1, 'y_pred': 0, 'y_score': 0.1, 'threshold': 0.6},
            {'sample_id': 's4', 'vehicle_id': 'v1', 'fault_type': 'samp', 'y_true': 0, 'y_pred': 0, 'y_score': 0.2, 'threshold': 0.6},
        ]
    )

    cases = select_case_examples(samples, predictions, success_fault='isc', failure_fault='samp', context_radius=1)
    summary = case_rows_to_frame(cases)

    assert {case.case_name for case in cases} == {'isc_true_positive', 'samp_false_negative'}
    assert set(summary['case_name']) == {'isc_true_positive', 'samp_false_negative'}
    assert all({'score', 'threshold', 'y_true'} <= set(case.plot_frame['series']) for case in cases)


def test_load_warning_prediction_frames_normalizes_model_names(tmp_path):
    results_root = tmp_path / 'results'
    for model_dir in ['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'main_dual_task']:
        target_dir = results_root / 'warning' / model_dir
        target_dir.mkdir(parents=True)
        pd.DataFrame(
            [{'sample_id': 's1', 'vehicle_id': 'v1', 'fault_type': 'sd', 'y_true': 0, 'y_pred': 0, 'y_score': 0.1, 'threshold': 0.5}]
        ).to_csv(target_dir / 'predictions.csv', index=False)

    frames = load_warning_prediction_frames(results_root)

    assert set(frames) == {'threshold_trend', 'lightgbm', 'lstm', 'transformer', 'shared_encoder_expert_heads'}


def test_build_metric_bar_frame_preserves_categories_and_metric_names():
    comparison = pd.DataFrame(
        [
            {'model_name': 'LSTM', 'mean_core_f1': 0.3, 'mean_core_recall': 0.5},
            {'model_name': 'Proposed Method', 'mean_core_f1': 0.4, 'mean_core_recall': 0.6},
        ]
    )

    bar_frame = build_metric_bar_frame(comparison, category_column='model_name', metric_columns=['mean_core_f1', 'mean_core_recall'])

    assert set(bar_frame['category']) == {'LSTM', 'Proposed Method'}
    assert set(bar_frame['series']) == {'mean_core_f1', 'mean_core_recall'}
