from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FRAME_INTERVAL_SECONDS, MAX_SEGMENT_GAP_SECONDS
from .field_mapping import normalize_sensor_sequences, parse_sensor_series
from .labels import extract_events, merge_short_gaps
from .profiling import _series_to_epoch_seconds, STRUCTURED_LABEL_COLUMNS
from .rule_reconstruction import (
    reconstruct_connection_labels,
    reconstruct_insulation_labels,
    reconstruct_internal_short_labels,
    reconstruct_sampling_labels,
    reconstruct_self_discharge_labels,
    reconstruct_structured_core_labels,
)


FINAL_LABEL_MAP = {
    'sd': ('label_raw_sd', 'label_final_sd', STRUCTURED_LABEL_COLUMNS['sd']),
    'isc': ('label_raw_isc', 'label_final_isc', STRUCTURED_LABEL_COLUMNS['isc']),
    'conn': ('label_raw_conn', 'label_final_conn', STRUCTURED_LABEL_COLUMNS['conn']),
    'samp': ('label_raw_samp', 'label_final_samp', STRUCTURED_LABEL_COLUMNS['samp']),
    'ins': ('label_raw_ins', 'label_final_ins', STRUCTURED_LABEL_COLUMNS['ins']),
}

RECONSTRUCTED_FAULTS = {'sd', 'isc', 'conn', 'samp', 'ins'}

EMPTY_EVENT_COLUMNS = [
    'vehicle_id',
    'fault_type',
    'event_id',
    'start_frame',
    'end_frame',
    'duration_frames',
    'duration_seconds',
]


RAW_REQUIRED_COLUMNS = ['terminaltime', 'chargestatus', 'batteryvoltage', 'probetemperatures']
RAW_OPTIONAL_COLUMNS = ['speed', 'totalvoltage', 'totalcurrent', 'soc', 'insulationresistance']


def _build_segment_ids(timestamps: list[int]) -> list[int]:
    segment_ids = []
    current_segment = 0
    previous_time = None
    for current_time in timestamps:
        if previous_time is None or current_time - previous_time > MAX_SEGMENT_GAP_SECONDS:
            current_segment += 1
        segment_ids.append(current_segment)
        previous_time = current_time
    return segment_ids


def build_structured_label_artifacts(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = pd.read_csv(csv_path)
    source_df = source_df.reset_index(drop=True)
    timestamps = _series_to_epoch_seconds(source_df['TIME'])

    frame_df = pd.DataFrame(
        {
            'vehicle_id': csv_path.stem,
            'source_dataset': 'structured_dataset',
            'source_file': str(csv_path),
            'timestamp': timestamps,
            'frame_index': source_df.index,
            'charge_status': pd.to_numeric(source_df['CHARGE_STATUS'], errors='coerce').fillna(0).astype(int),
            'quality_flag': 'ok',
        }
    )

    frame_df['segment_id'] = _build_segment_ids(timestamps)
    frame_df['is_continuous_segment'] = True

    reconstructed = reconstruct_structured_core_labels(source_df)
    events = []
    for fault, (raw_col, final_col, source_col) in FINAL_LABEL_MAP.items():
        if fault in RECONSTRUCTED_FAULTS:
            raw_values = pd.Series(reconstructed[fault]).fillna(0).astype(int).tolist()
        else:
            raw_values = pd.to_numeric(source_df.get(source_col, 0), errors='coerce').fillna(0).astype(int).tolist()
        final_values = merge_short_gaps(raw_values, max_gap=2)
        frame_df[raw_col] = raw_values
        frame_df[final_col] = final_values
        for event in extract_events(final_values, frame_interval_sec=FRAME_INTERVAL_SECONDS):
            events.append(
                {
                    'vehicle_id': csv_path.stem,
                    'fault_type': fault,
                    **event,
                }
            )

    frame_df['label_version'] = 'v3'
    frame_df['label_source'] = 'structured_reconstructed_rules'

    keep_columns = [
        'vehicle_id', 'source_dataset', 'source_file', 'timestamp', 'frame_index', 'charge_status',
        'is_continuous_segment', 'segment_id',
        'label_raw_sd', 'label_raw_isc', 'label_raw_conn', 'label_raw_samp', 'label_raw_ins',
        'label_final_sd', 'label_final_isc', 'label_final_conn', 'label_final_samp', 'label_final_ins',
        'label_version', 'label_source', 'quality_flag',
    ]
    events_df = pd.DataFrame(events, columns=EMPTY_EVENT_COLUMNS)
    return frame_df[keep_columns], events_df


def build_raw_label_artifacts(csv_path: Path, vehicle_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    header = pd.read_csv(csv_path, nrows=0)
    available_columns = header.columns.tolist()
    missing_required = [column for column in RAW_REQUIRED_COLUMNS if column not in available_columns]
    if missing_required:
        raise ValueError(f'Missing required RAW columns {missing_required} in {csv_path}.')

    usecols = [*RAW_REQUIRED_COLUMNS, *[column for column in RAW_OPTIONAL_COLUMNS if column in available_columns]]
    source_df = pd.read_csv(csv_path, usecols=usecols).reset_index(drop=True)
    timestamps = _series_to_epoch_seconds(source_df['terminaltime'])
    charge_status = pd.to_numeric(source_df['chargestatus'], errors='coerce').fillna(0).astype(int)

    voltage_lists = source_df['batteryvoltage'].map(parse_sensor_series)
    temperature_lists = source_df['probetemperatures'].map(parse_sensor_series)
    normalized_voltages, voltage_ok, _ = normalize_sensor_sequences(voltage_lists)
    normalized_temperatures, temperature_ok, _ = normalize_sensor_sequences(temperature_lists)
    quality_ok = voltage_ok & temperature_ok

    if 'insulationresistance' in source_df.columns:
        insulation_series = pd.to_numeric(source_df['insulationresistance'], errors='coerce')
        insulation_supported = bool(insulation_series.notna().any())
        insulation_values = insulation_series.fillna(0.0).to_numpy(dtype=float)
    else:
        insulation_supported = False
        insulation_values = np.zeros(len(source_df), dtype=float)

    voltage_matrix = np.asarray(normalized_voltages, dtype=float) if normalized_voltages else np.zeros((len(source_df), 0), dtype=float)
    temperature_matrix = np.asarray(normalized_temperatures, dtype=float) if normalized_temperatures else np.zeros((len(source_df), 0), dtype=float)

    reconstructed = {
        'sd': reconstruct_self_discharge_labels(voltage_matrix),
        'isc': reconstruct_internal_short_labels(voltage_matrix, temperature_matrix),
        'conn': reconstruct_connection_labels(voltage_matrix),
        'samp': reconstruct_sampling_labels(voltage_matrix, charge_status.to_numpy(dtype=int)),
        'ins': reconstruct_insulation_labels(charge_status.to_numpy(dtype=int), insulation_values) if insulation_supported else np.zeros(len(source_df), dtype=np.int64),
    }

    frame_df = pd.DataFrame(
        {
            'vehicle_id': vehicle_id,
            'source_dataset': 'raw_dataset',
            'source_file': str(csv_path),
            'timestamp': timestamps,
            'frame_index': source_df.index,
            'charge_status': charge_status,
            'quality_flag': np.where(quality_ok, 'ok', 'low'),
            'segment_id': _build_segment_ids(timestamps),
            'is_continuous_segment': True,
        }
    )

    events = []
    for fault, (raw_col, final_col, _) in FINAL_LABEL_MAP.items():
        raw_values = pd.Series(reconstructed[fault]).fillna(0).astype(int).tolist()
        final_values = merge_short_gaps(raw_values, max_gap=2)
        frame_df[raw_col] = raw_values
        frame_df[final_col] = final_values
        for event in extract_events(final_values, frame_interval_sec=FRAME_INTERVAL_SECONDS):
            events.append(
                {
                    'vehicle_id': vehicle_id,
                    'fault_type': fault,
                    **event,
                }
            )

    frame_df['label_version'] = 'v4'
    frame_df['label_source'] = 'raw_reconstructed_rules' if insulation_supported else 'raw_reconstructed_rules_partial_ins'

    keep_columns = [
        'vehicle_id', 'source_dataset', 'source_file', 'timestamp', 'frame_index', 'charge_status',
        'is_continuous_segment', 'segment_id',
        'label_raw_sd', 'label_raw_isc', 'label_raw_conn', 'label_raw_samp', 'label_raw_ins',
        'label_final_sd', 'label_final_isc', 'label_final_conn', 'label_final_samp', 'label_final_ins',
        'label_version', 'label_source', 'quality_flag',
    ]
    events_df = pd.DataFrame(events, columns=EMPTY_EVENT_COLUMNS)
    return frame_df[keep_columns], events_df


def save_label_bundle(frame_df: pd.DataFrame, events_df: pd.DataFrame, per_vehicle_root: Path, events_root: Path, vehicle_id: str) -> tuple[Path, Path]:
    per_vehicle_root.mkdir(parents=True, exist_ok=True)
    events_root.mkdir(parents=True, exist_ok=True)
    label_path = per_vehicle_root / f'{vehicle_id}_labels.csv'
    event_path = events_root / f'{vehicle_id}_events.csv'
    frame_df.to_csv(label_path, index=False, encoding='utf-8-sig')
    events_df.to_csv(event_path, index=False, encoding='utf-8-sig')
    return label_path, event_path


def partition_summary_by_dataset(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = summary_df.reset_index(drop=True)
    return {
        'structured_dataset': ordered[ordered['source_dataset'] == 'structured_dataset'].reset_index(drop=True),
        'raw_dataset': ordered[ordered['source_dataset'] == 'raw_dataset'].reset_index(drop=True),
        'final': ordered,
    }
