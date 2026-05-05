from pathlib import Path

import pandas as pd

from scripts.build_samples import load_raw_feature_frame
from battery_thesis.samples import (
    assemble_sample_artifacts,
    build_samples_master,
    compute_window_features,
    downsample_split_samples,
    save_sample_artifacts,
    validate_full_split_consistency,
    validate_medium_split_coverage,
)


def _demo_frame_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'timestamp': [10, 20, 30, 40, 50, 60],
            'frame_index': [0, 1, 2, 3, 4, 5],
            'segment_id': [1, 1, 1, 1, 1, 1],
            'quality_flag': ['ok'] * 6,
            'charge_status': [3, 3, 3, 3, 3, 3],
            'soc': [50, 49, 48, 47, 46, 45],
            'speed': [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            'sum_voltage': [350.0, 349.5, 349.0, 348.5, 348.0, 347.5],
            'sum_current': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'insulation_resistance': [5000, 4950, 4900, 4850, 4800, 4750],
            'voltages': [
                [3.50, 3.40],
                [3.49, 3.39],
                [3.48, 3.38],
                [3.47, 3.37],
                [3.46, 3.36],
                [3.45, 3.35],
            ],
            'temperatures': [
                [28, 29],
                [29, 30],
                [30, 31],
                [31, 32],
                [32, 33],
                [33, 34],
            ],
            'label_final_sd': [0, 0, 1, 1, 1, 1],
            'label_final_isc': [0, 0, 0, 0, 0, 0],
            'label_final_conn': [0, 0, 0, 0, 0, 0],
            'label_final_samp': [0, 0, 0, 0, 0, 0],
            'label_final_ins': [0, 0, 0, 0, 0, 0],
        }
    )


def test_build_samples_master_generates_identification_and_warning_labels():
    frame_df = _demo_frame_df()[[
        'timestamp', 'frame_index', 'segment_id', 'quality_flag',
        'label_final_sd', 'label_final_isc', 'label_final_conn', 'label_final_samp', 'label_final_ins',
    ]]

    samples = build_samples_master(
        frame_labels=frame_df,
        vehicle_id='v1',
        source_dataset='structured_dataset',
        split='train',
        window_size=3,
        step_size=1,
        forecast_horizon_frames=2,
        frame_interval_sec=10,
    )

    assert list(samples['sample_id']) == ['v1_000000', 'v1_000001']
    assert list(samples['y_id_sd']) == [0, 1]
    assert list(samples['y_warn_sd']) == [1, 0]


def test_compute_window_features_prefixes_shared_and_fault_specific_columns():
    window = _demo_frame_df().iloc[:3].copy()

    features = compute_window_features(window)

    assert features['shared_soc'] == 48
    assert features['shared_v_range'] == 0.1
    assert 'sd_cell_dev_min' in features
    assert 'conn_dvdt_max' in features
    assert 'ins_rins_min' in features


def test_assemble_and_save_sample_artifacts(tmp_path: Path):
    frame_df = _demo_frame_df()

    samples_master, features_all, tensors = assemble_sample_artifacts(
        frame_df=frame_df,
        vehicle_id='v1',
        source_dataset='structured_dataset',
        split='train',
        window_size=3,
        step_size=1,
        forecast_horizon_frames=2,
        frame_interval_sec=10,
    )

    assert len(samples_master) == 2
    assert list(features_all['sample_id']) == ['v1_000000', 'v1_000001']
    assert tensors['X_seq'].shape == (2, 3, 8)
    assert tensors['X_feat'].shape[0] == 2
    assert tensors['y_id'].shape == (2, 5)
    assert tensors['y_warn'].shape == (2, 5)

    paths = save_sample_artifacts(tmp_path, samples_master, features_all, tensors)

    assert paths['samples_master'].exists()
    assert paths['features_all'].exists()
    assert paths['dataset_pack'].exists()


def test_assemble_sample_artifacts_sanitizes_nan_and_inf_values():
    frame_df = _demo_frame_df().copy()
    frame_df.at[1, 'voltages'] = [float('nan'), 3.39]
    frame_df.at[2, 'temperatures'] = [30.0, float('inf')]
    frame_df.at[0, 'insulation_resistance'] = float('nan')
    frame_df.at[2, 'sum_voltage'] = float('inf')

    samples_master, features_all, tensors = assemble_sample_artifacts(
        frame_df=frame_df,
        vehicle_id='v1',
        source_dataset='structured_dataset',
        split='train',
        window_size=3,
        step_size=1,
        forecast_horizon_frames=2,
        frame_interval_sec=10,
    )

    assert len(samples_master) == 2
    assert not features_all.drop(columns=['sample_id']).isna().any().any()
    assert not pd.isna(features_all['shared_sum_voltage']).any()
    assert not pd.isna(features_all['shared_insulation_resistance']).any()
    assert not __import__('numpy').isnan(tensors['X_seq']).any()
    assert not __import__('numpy').isinf(tensors['X_seq']).any()
    assert not __import__('numpy').isnan(tensors['X_feat']).any()
    assert not __import__('numpy').isinf(tensors['X_feat']).any()


def test_downsample_split_samples_preserves_required_fault_coverage():
    rows = []
    sample_counter = 0
    for fault in ['sd', 'isc', 'conn', 'samp', 'ins']:
        row = {
            'sample_id': f's_{sample_counter}',
            'vehicle_id': 'v1',
            'split': 'test',
            'future_first_fault': fault,
            'lead_time_sec': 20,
        }
        for current_fault in ['sd', 'isc', 'conn', 'samp', 'ins']:
            row[f'y_id_{current_fault}'] = 1 if current_fault == fault else 0
            row[f'y_warn_{current_fault}'] = 1 if current_fault == fault and current_fault in {'sd', 'samp', 'ins'} else 0
        rows.append(row)
        sample_counter += 1

    for _ in range(20):
        row = {
            'sample_id': f's_{sample_counter}',
            'vehicle_id': 'v1',
            'split': 'test',
            'future_first_fault': None,
            'lead_time_sec': None,
        }
        for current_fault in ['sd', 'isc', 'conn', 'samp', 'ins']:
            row[f'y_id_{current_fault}'] = 0
            row[f'y_warn_{current_fault}'] = 0
        rows.append(row)
        sample_counter += 1

    samples = pd.DataFrame(rows)
    selected = downsample_split_samples(
        samples,
        split='test',
        target_count=8,
        seed=20260408,
        required_identification_faults=['sd', 'isc', 'conn', 'samp', 'ins'],
        required_warning_faults=['sd', 'samp', 'ins'],
    )

    assert len(selected) >= 5
    for fault in ['sd', 'isc', 'conn', 'samp', 'ins']:
        assert int(selected[f'y_id_{fault}'].sum()) >= 1
    for fault in ['sd', 'samp', 'ins']:
        assert int(selected[f'y_warn_{fault}'].sum()) >= 1



def test_validate_medium_split_coverage_accepts_complete_summary():
    summary = pd.DataFrame(
        [
            {
                'split': 'train',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 1, 'y_id_conn': 1, 'y_id_samp': 1, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 1, 'y_warn_ins': 1,
            },
            {
                'split': 'val',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 1, 'y_id_conn': 1, 'y_id_samp': 1, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 1, 'y_warn_ins': 1,
            },
            {
                'split': 'test',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 1, 'y_id_conn': 1, 'y_id_samp': 1, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 1, 'y_warn_ins': 1,
            },
        ]
    )

    report = validate_medium_split_coverage(summary)

    assert report['blocking_errors'] == []
    assert report['rare_fault_notes'] == []



def test_validate_medium_split_coverage_rejects_missing_core_faults():
    summary = pd.DataFrame(
        [
            {
                'split': 'train',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 1, 'y_id_conn': 1, 'y_id_samp': 1, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 1, 'y_warn_ins': 1,
            },
            {
                'split': 'val',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 0, 'y_id_conn': 1, 'y_id_samp': 0, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 0, 'y_warn_ins': 1,
            },
            {
                'split': 'test',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 1, 'y_id_conn': 1, 'y_id_samp': 1, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 1, 'y_warn_ins': 1,
            },
        ]
    )

    try:
        validate_medium_split_coverage(summary)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError('Expected validate_medium_split_coverage to fail for incomplete core coverage.')

    assert 'val:y_id_samp' in message
    assert 'val:y_warn_samp' in message



def test_validate_medium_split_coverage_keeps_rare_faults_as_notes():
    summary = pd.DataFrame(
        [
            {
                'split': 'train',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 0, 'y_id_conn': 0, 'y_id_samp': 1, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 1, 'y_warn_ins': 1,
            },
            {
                'split': 'val',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 0, 'y_id_conn': 0, 'y_id_samp': 1, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 1, 'y_warn_ins': 1,
            },
            {
                'split': 'test',
                'sample_count': 10,
                'y_id_sd': 1, 'y_id_isc': 1, 'y_id_conn': 1, 'y_id_samp': 1, 'y_id_ins': 1,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 1, 'y_warn_ins': 1,
            },
        ]
    )

    report = validate_medium_split_coverage(summary)

    assert report['blocking_errors'] == []
    assert any('isc' in note for note in report['rare_fault_notes'])
    assert all('conn' not in note for note in report['rare_fault_notes'])


def test_validate_full_split_consistency_accepts_vehicle_isolation_and_matching_split_map():
    samples_master = pd.DataFrame(
        [
            {'sample_id': 'v1_0', 'vehicle_id': 'v1', 'split': 'train'},
            {'sample_id': 'v1_1', 'vehicle_id': 'v1', 'split': 'train'},
            {'sample_id': 'v2_0', 'vehicle_id': 'v2', 'split': 'val'},
            {'sample_id': 'v3_0', 'vehicle_id': 'v3', 'split': 'test'},
        ]
    )
    split_mapping = pd.DataFrame(
        [
            {'vehicle_id': 'v1', 'split': 'train'},
            {'vehicle_id': 'v2', 'split': 'val'},
            {'vehicle_id': 'v3', 'split': 'test'},
        ]
    )

    report = validate_full_split_consistency(samples_master, split_mapping, raise_on_blocking=False)

    assert report['blocking_errors'] == []
    assert report['split_vehicle_counts'] == {'train': 1, 'val': 1, 'test': 1}


def test_validate_full_split_consistency_rejects_cross_split_or_mapping_drift():
    samples_master = pd.DataFrame(
        [
            {'sample_id': 'v1_0', 'vehicle_id': 'v1', 'split': 'train'},
            {'sample_id': 'v1_1', 'vehicle_id': 'v1', 'split': 'test'},
            {'sample_id': 'v9_0', 'vehicle_id': 'v9', 'split': 'val'},
        ]
    )
    split_mapping = pd.DataFrame(
        [
            {'vehicle_id': 'v1', 'split': 'train'},
            {'vehicle_id': 'v2', 'split': 'val'},
        ]
    )

    report = validate_full_split_consistency(samples_master, split_mapping, raise_on_blocking=False)

    assert any('v1' in item for item in report['blocking_errors'])
    assert any('v9' in item for item in report['blocking_errors'])



def test_materialize_selected_samples_aligns_tensor_order_with_samples_master(monkeypatch):
    import scripts.build_samples as build_samples

    selected_samples = pd.DataFrame(
        [
            {
                'sample_id': 'v2_000001',
                'vehicle_id': 'v2',
                'source_dataset': 'structured_dataset',
                'split': 'test',
                'start_frame': 1,
                'end_frame': 3,
                'y_id_sd': 1, 'y_id_isc': 0, 'y_id_conn': 0, 'y_id_samp': 0, 'y_id_ins': 0,
                'y_warn_sd': 0, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 0, 'y_warn_ins': 0,
            },
            {
                'sample_id': 'v1_000000',
                'vehicle_id': 'v1',
                'source_dataset': 'structured_dataset',
                'split': 'train',
                'start_frame': 0,
                'end_frame': 2,
                'y_id_sd': 0, 'y_id_isc': 0, 'y_id_conn': 0, 'y_id_samp': 1, 'y_id_ins': 0,
                'y_warn_sd': 1, 'y_warn_isc': 0, 'y_warn_conn': 0, 'y_warn_samp': 0, 'y_warn_ins': 0,
            },
        ]
    )

    def fake_resolve_source_csv(vehicle_id, source_dataset, structured_root, raw_root):
        return Path(f'{vehicle_id}.csv')

    def fake_load_label_frame(vehicle_id, source_dataset, label_kind, ignore_quality_flag):
        return pd.DataFrame()

    def fake_load_structured_feature_frame(csv_path):
        frame_df = _demo_frame_df().copy()
        frame_df['segment_id'] = 1
        frame_df['quality_flag'] = 'ok'
        return frame_df

    monkeypatch.setattr(build_samples, 'resolve_source_csv', fake_resolve_source_csv)
    monkeypatch.setattr(build_samples, 'load_label_frame', fake_load_label_frame)
    monkeypatch.setattr(build_samples, 'load_structured_feature_frame', fake_load_structured_feature_frame)
    monkeypatch.setattr(build_samples, 'merge_feature_and_label_frames', lambda feature_df, label_df: feature_df)

    samples_master, features_all, tensors = build_samples.materialize_selected_samples(
        selected_samples=selected_samples,
        label_kind='final',
        ignore_quality_flag=False,
        structured_root=Path('.'),
        raw_root=Path('.'),
    )

    expected_ids = samples_master['sample_id'].tolist()
    actual_ids = tensors['sample_id'].astype(str).tolist()

    assert expected_ids == ['v1_000000', 'v2_000001']
    assert actual_ids == expected_ids
    assert features_all['sample_id'].tolist() == expected_ids



def test_load_raw_feature_frame_normalizes_sensor_lengths_and_keeps_insulation(tmp_path: Path):
    csv_path = tmp_path / 'vin1.csv'
    pd.DataFrame(
        [
            {
                'terminaltime': 1,
                'soc': 20,
                'speed': 0.0,
                'chargestatus': 3,
                'totalvoltage': 10.8,
                'totalcurrent': 0.1,
                'batteryvoltage': '3.6~3.6~3.6',
                'probetemperatures': '30~31',
                'insulationresistance': 500.0,
            },
            {
                'terminaltime': 11,
                'soc': 20,
                'speed': 0.0,
                'chargestatus': 3,
                'totalvoltage': 10.7,
                'totalcurrent': 0.1,
                'batteryvoltage': '3.5~3.5',
                'probetemperatures': '30',
                'insulationresistance': 480.0,
            },
        ]
    ).to_csv(csv_path, index=False, encoding='utf-8')

    frame_df = load_raw_feature_frame(csv_path)

    assert frame_df['insulation_resistance'].tolist() == [500.0, 480.0]
    assert len(frame_df.loc[0, 'voltages']) == len(frame_df.loc[1, 'voltages'])
    assert len(frame_df.loc[0, 'temperatures']) == len(frame_df.loc[1, 'temperatures'])
