from pathlib import Path

import pandas as pd
import pytest

from battery_thesis.splitting import assign_vehicle_splits


REQUIRED_WARNING_FAULTS = ['sd', 'samp', 'ins']
CORE_ID_FAULTS = ['sd', 'samp', 'ins']


def test_assign_vehicle_splits_prioritizes_fault_coverage_for_test_split():
    vehicle_meta = pd.DataFrame(
        [
            {'vehicle_id': 'v1', 'window_count': 100, 'fault_vector': '1|0|0|0|0'},
            {'vehicle_id': 'v2', 'window_count': 100, 'fault_vector': '0|1|0|0|0'},
            {'vehicle_id': 'v3', 'window_count': 100, 'fault_vector': '0|0|1|0|0'},
            {'vehicle_id': 'v4', 'window_count': 100, 'fault_vector': '0|0|0|1|0'},
            {'vehicle_id': 'v5', 'window_count': 100, 'fault_vector': '0|0|0|0|1'},
            {'vehicle_id': 'v6', 'window_count': 100, 'fault_vector': '1|1|1|1|1'},
        ]
    )

    assigned = assign_vehicle_splits(
        vehicle_meta,
        train_count=2,
        val_count=2,
        test_count=2,
        seed=20260407,
        prioritize_fault_coverage=True,
    )

    test_rows = assigned[assigned['split'] == 'test']
    fault_vectors = test_rows['fault_vector'].tolist()

    assert len(test_rows) == 2
    assert '1|1|1|1|1' in fault_vectors


def test_assign_vehicle_splits_preserves_requested_counts():
    vehicle_meta = pd.DataFrame(
        [
            {'vehicle_id': f'v{i}', 'window_count': 10 + i, 'fault_vector': '0|0|0|0|0'}
            for i in range(1, 11)
        ]
    )

    assigned = assign_vehicle_splits(
        vehicle_meta,
        train_count=6,
        val_count=2,
        test_count=2,
        seed=20260407,
        prioritize_fault_coverage=False,
    )

    assert (assigned['split'] == 'train').sum() == 6
    assert (assigned['split'] == 'val').sum() == 2
    assert (assigned['split'] == 'test').sum() == 2


def test_assign_vehicle_splits_can_cover_val_and_test_when_possible():
    vehicle_meta = pd.DataFrame(
        [
            {'vehicle_id': 'v1', 'window_count': 100, 'fault_vector': '1|1|0|0|0'},
            {'vehicle_id': 'v2', 'window_count': 100, 'fault_vector': '0|0|1|1|0'},
            {'vehicle_id': 'v3', 'window_count': 100, 'fault_vector': '0|0|0|0|1'},
            {'vehicle_id': 'v4', 'window_count': 100, 'fault_vector': '1|0|1|0|0'},
            {'vehicle_id': 'v5', 'window_count': 100, 'fault_vector': '0|1|0|1|1'},
            {'vehicle_id': 'v6', 'window_count': 100, 'fault_vector': '1|0|0|0|1'},
            {'vehicle_id': 'v7', 'window_count': 100, 'fault_vector': '0|1|1|0|0'},
            {'vehicle_id': 'v8', 'window_count': 100, 'fault_vector': '0|0|0|1|1'},
            {'vehicle_id': 'v9', 'window_count': 100, 'fault_vector': '0|0|0|0|0'},
            {'vehicle_id': 'v10', 'window_count': 100, 'fault_vector': '0|0|0|0|0'},
        ]
    )

    assigned = assign_vehicle_splits(
        vehicle_meta,
        train_count=2,
        val_count=4,
        test_count=4,
        seed=20260407,
        prioritize_fault_coverage=True,
    )

    for split_name in ['val', 'test']:
        split_rows = assigned[assigned['split'] == split_name]
        coverage = [0, 0, 0, 0, 0]
        for value in split_rows['fault_vector']:
            bits = [int(bit) for bit in str(value).split('|')]
            coverage = [max(left, right) for left, right in zip(coverage, bits)]
        assert coverage == [1, 1, 1, 1, 1]


def test_current_structured_vehicle_split_uses_core_hard_and_rare_soft_constraints():
    project_root = Path(__file__).resolve().parents[1]
    split_path = project_root / 'artifacts' / 'dataset_meta' / 'structured_vehicle_split.csv'
    constraints_path = project_root / 'artifacts' / 'dataset_meta' / 'selection_constraints.md'
    if not split_path.exists() or not constraints_path.exists():
        pytest.skip('requires generated dataset split artifacts, which are not committed')
    split_df = pd.read_csv(split_path)
    constraints_text = constraints_path.read_text(encoding='utf-8-sig')

    for split_name in ['train', 'val', 'test']:
        split_rows = split_df[split_df['split'] == split_name]
        for fault in CORE_ID_FAULTS:
            assert (split_rows[f'y_id_{fault}_positive'] > 0).any()

    for split_name in ['val', 'test']:
        split_rows = split_df[split_df['split'] == split_name]
        for fault in REQUIRED_WARNING_FAULTS:
            assert (split_rows[f'y_warn_{fault}_positive'] > 0).any()

    isc_rows = split_df[split_df['y_id_isc_positive'] > 0]
    if len(isc_rows) >= 2:
        assert (isc_rows['split'] == 'train').any()
        assert isc_rows['split'].isin(['val', 'test']).any()

    conn_rows = split_df[split_df['y_id_conn_positive'] > 0]
    if not conn_rows.empty:
        assert conn_rows['split'].isin(['val', 'test']).any()

    assert '第1轮划分约束说明' in constraints_text
    assert '数据限制说明' in constraints_text
