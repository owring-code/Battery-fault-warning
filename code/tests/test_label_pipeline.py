from pathlib import Path

import numpy as np
import pandas as pd

from battery_thesis.label_pipeline import build_raw_label_artifacts, build_structured_label_artifacts, partition_summary_by_dataset
from battery_thesis.rule_reconstruction import (
    reconstruct_connection_labels,
    reconstruct_insulation_labels,
    reconstruct_internal_short_labels,
    reconstruct_sampling_labels,
    reconstruct_self_discharge_labels,
)


def test_reconstruct_self_discharge_labels_flags_persistent_weak_cell():
    voltages = np.array([[3.7, 3.7, 3.7, 3.7, 3.7] for _ in range(10)], dtype=float)
    voltages[:, 1] -= np.linspace(0.0, 0.08, 10)

    labels = reconstruct_self_discharge_labels(voltages, rolling_window=3)

    assert labels.sum() >= 1
    assert labels[-1] == 1


def test_reconstruct_sampling_labels_flags_neighbor_contradiction_during_discharge():
    voltages = np.array([
        [3.70, 3.70, 3.70, 3.70, 3.70],
        [3.70, 3.40, 4.00, 3.70, 3.70],
        [3.70, 3.40, 4.00, 3.70, 3.70],
        [3.70, 3.40, 4.00, 3.70, 3.70],
        [3.70, 3.70, 3.70, 3.70, 3.70],
        [3.70, 3.70, 3.70, 3.70, 3.70],
    ], dtype=float)
    charge_status = np.array([3, 3, 3, 3, 3, 3], dtype=int)

    labels = reconstruct_sampling_labels(voltages, charge_status, window_length=3)

    assert labels.sum() >= 1
    assert labels[1] == 1


def test_reconstruct_insulation_labels_marks_low_resistance_and_extends():
    charge_status = np.array([3, 3, 3, 3], dtype=int)
    insulation = np.array([400.0, 150.0, 150.1, 150.2], dtype=float)

    labels = reconstruct_insulation_labels(charge_status, insulation)

    assert labels.tolist() == [0, 1, 1, 1]


def test_reconstruct_internal_short_labels_flags_local_voltage_drop_with_heat():
    voltages = np.full((6, 6), 3.70, dtype=float)
    temperatures = np.full((6, 2), 35.0, dtype=float)
    voltages[3:, 2] = [3.55, 3.40, 3.30]
    temperatures[3:, 0] = [60.0, 85.0, 110.0]

    labels = reconstruct_internal_short_labels(voltages, temperatures)

    assert labels.sum() >= 1
    assert labels[-1] == 1


def test_reconstruct_connection_labels_flags_large_jump_with_low_adjacent_icc():
    voltages = np.array(
        [
            [3.70, 3.70, 3.70, 3.70],
            [3.70, 3.70, 3.70, 3.70],
            [3.70, 3.82, 3.69, 3.70],
            [3.70, 3.55, 3.68, 3.70],
            [3.70, 3.85, 3.67, 3.70],
            [3.70, 3.52, 3.66, 3.70],
        ],
        dtype=float,
    )

    labels = reconstruct_connection_labels(voltages)

    assert labels.sum() >= 1
    assert labels[2] == 1 or labels[3] == 1


def test_build_structured_label_artifacts_reconstructs_core_rules_from_signals(tmp_path: Path):
    csv_path = tmp_path / 'WCVT1.csv'
    rows = []
    base_voltages = [3.7, 3.7, 3.7, 3.7, 3.7]
    for i in range(6):
        row = {
            'TIME': 1 + i * 10,
            'CHARGE_STATUS': 3,
            'SPEED': 0.0,
            'SUM_VOLTAGE': 18.5,
            'SUM_CURRENT': 0.0,
            'SOC': 20,
            'INSULATION_RESISTANCE': 400.0 if i == 0 else 150.0 + min(i, 3) * 0.1,
            'sampling_exception': 0,
            'failure_of_insulation': 0,
            'connection_exception': 0,
            'sudden_short_circuit': 0,
            'Abnormal_self_discharge': 0,
            'T_1': 30,
            'T_2': 30,
        }
        voltages = base_voltages.copy()
        if 1 <= i <= 3:
            voltages[1] = 3.4
            voltages[2] = 4.0
        for idx, value in enumerate(voltages, start=1):
            row[f'U_{idx}'] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')

    frame_df, events_df = build_structured_label_artifacts(csv_path)

    assert frame_df['label_raw_ins'].sum() >= 1
    assert frame_df['label_raw_samp'].sum() >= 1
    assert frame_df['label_source'].iloc[0] == 'structured_reconstructed_rules'
    assert set(events_df['fault_type']) >= {'ins', 'samp'}


def test_build_structured_label_artifacts_reconstructs_isc_and_conn_rules(tmp_path: Path):
    csv_path = tmp_path / 'WCVT2.csv'
    rows = []
    for i in range(8):
        row = {
            'TIME': 1 + i * 10,
            'CHARGE_STATUS': 3,
            'SPEED': 0.0,
            'SUM_VOLTAGE': 22.2,
            'SUM_CURRENT': 0.0,
            'SOC': 20,
            'INSULATION_RESISTANCE': 500.0,
            'sampling_exception': 0,
            'failure_of_insulation': 0,
            'connection_exception': 0,
            'sudden_short_circuit': 0,
            'Abnormal_self_discharge': 0,
            'T_1': 35 if i < 5 else [50, 75, 105][i - 5],
            'T_2': 30,
        }
        voltages = [3.70, 3.70, 3.70, 3.70, 3.70, 3.70]
        if i >= 5:
            voltages[2] = [3.55, 3.42, 3.30][i - 5]
        if 2 <= i <= 6:
            voltages[1] = [3.82, 3.55, 3.85, 3.52, 3.83][i - 2]
            voltages[2] = min(voltages[2], [3.69, 3.68, 3.67, 3.66, 3.65][i - 2])
        for idx, value in enumerate(voltages, start=1):
            row[f'U_{idx}'] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')

    frame_df, events_df = build_structured_label_artifacts(csv_path)

    assert frame_df['label_raw_isc'].sum() >= 1
    assert frame_df['label_raw_conn'].sum() >= 1
    assert set(events_df['fault_type']) >= {'isc', 'conn'}


def test_partition_summary_by_dataset_returns_scoped_views():
    summary_df = pd.DataFrame(
        [
            {'vehicle_id': 's1', 'source_dataset': 'structured_dataset'},
            {'vehicle_id': 'r1', 'source_dataset': 'raw_dataset'},
        ]
    )

    scoped = partition_summary_by_dataset(summary_df)

    assert list(scoped['structured_dataset']['vehicle_id']) == ['s1']
    assert list(scoped['raw_dataset']['vehicle_id']) == ['r1']
    assert list(scoped['final']['vehicle_id']) == ['s1', 'r1']



def test_build_raw_label_artifacts_reconstructs_formal_raw_rules(tmp_path: Path):
    csv_path = tmp_path / 'vin1.csv'
    rows = []
    for i in range(6):
        voltages = [3.70, 3.70, 3.70, 3.70, 3.70]
        if 1 <= i <= 3:
            voltages[1] = 3.40
            voltages[2] = 4.00
        rows.append(
            {
                'terminaltime': 1 + i * 10,
                'chargestatus': 3,
                'speed': 0.0,
                'totalvoltage': sum(voltages),
                'totalcurrent': 0.0,
                'soc': 20,
                'insulationresistance': 400.0 if i == 0 else 150.0 + min(i, 3) * 0.1,
                'batteryvoltage': '~'.join(str(value) for value in voltages),
                'probetemperatures': '30~31',
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')

    frame_df, events_df = build_raw_label_artifacts(csv_path, 'vin1')

    assert frame_df['label_raw_ins'].sum() >= 1
    assert frame_df['label_raw_samp'].sum() >= 1
    assert frame_df['label_source'].iloc[0] == 'raw_reconstructed_rules'
    assert (frame_df['quality_flag'] == 'ok').all()
    assert set(events_df['fault_type']) >= {'ins', 'samp'}


def test_build_raw_label_artifacts_marks_partial_ins_when_column_missing(tmp_path: Path):
    csv_path = tmp_path / 'vin2.csv'
    pd.DataFrame(
        [
            {
                'terminaltime': 1,
                'chargestatus': 3,
                'speed': 0.0,
                'totalvoltage': 18.5,
                'totalcurrent': 0.0,
                'soc': 20,
                'batteryvoltage': '3.7~3.7~3.7~3.7~3.7',
                'probetemperatures': '30~31',
            },
            {
                'terminaltime': 11,
                'chargestatus': 3,
                'speed': 0.0,
                'totalvoltage': 18.2,
                'totalcurrent': 0.0,
                'soc': 20,
                'batteryvoltage': '3.7~3.4~4.0~3.7~3.7',
                'probetemperatures': '30~31',
            },
        ]
    ).to_csv(csv_path, index=False, encoding='utf-8')

    frame_df, _ = build_raw_label_artifacts(csv_path, 'vin2')

    assert frame_df['label_raw_ins'].sum() == 0
    assert frame_df['label_source'].iloc[0] == 'raw_reconstructed_rules_partial_ins'

def test_build_raw_label_artifacts_accepts_sparse_missing_terminaltime(tmp_path: Path):
    csv_path = tmp_path / 'vin3.csv'
    pd.DataFrame(
        [
            {
                'terminaltime': 38.0,
                'chargestatus': 3,
                'speed': 0.0,
                'totalvoltage': 18.5,
                'totalcurrent': 0.0,
                'soc': 20,
                'insulationresistance': 400.0,
                'batteryvoltage': '3.7~3.7~3.7~3.7~3.7',
                'probetemperatures': '30~31',
            },
            {
                'terminaltime': np.nan,
                'chargestatus': 3,
                'speed': 0.0,
                'totalvoltage': 18.5,
                'totalcurrent': 0.0,
                'soc': 20,
                'insulationresistance': 400.0,
                'batteryvoltage': '3.7~3.7~3.7~3.7~3.7',
                'probetemperatures': '30~31',
            },
            {
                'terminaltime': 58.0,
                'chargestatus': 3,
                'speed': 0.0,
                'totalvoltage': 18.5,
                'totalcurrent': 0.0,
                'soc': 20,
                'insulationresistance': 400.0,
                'batteryvoltage': '3.7~3.7~3.7~3.7~3.7',
                'probetemperatures': '30~31',
            },
        ]
    ).to_csv(csv_path, index=False, encoding='utf-8')

    frame_df, _ = build_raw_label_artifacts(csv_path, 'vin3')

    assert frame_df['timestamp'].tolist() == [38, 48, 58]
