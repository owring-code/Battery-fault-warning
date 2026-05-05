from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FAULT_LABEL_COLUMNS,
    FAULT_ORDER,
    FINAL_LABELS_ROOT,
    FORECAST_HORIZON_FRAMES,
    MAX_SEGMENT_GAP_SECONDS,
    STEP_SIZE_FRAMES,
    WINDOW_SIZE_FRAMES,
)
from .metadata import scan_project_datasets
from .splitting import assign_vehicle_splits


STRUCTURED_LABEL_COLUMNS = {
    'sd': 'Abnormal_self_discharge',
    'isc': 'sudden_short_circuit',
    'conn': 'connection_exception',
    'samp': 'sampling_exception',
    'ins': 'failure_of_insulation',
}


def _fill_sparse_missing(values: pd.Series) -> pd.Series:
    return values.astype(float).interpolate(limit_direction='both')



def _series_to_epoch_seconds(series: pd.Series) -> list[int]:
    numeric = pd.to_numeric(series, errors='coerce')
    original_non_null = series.notna()
    if numeric.notna().sum() == int(original_non_null.sum()) and numeric.notna().any():
        numeric_filled = _fill_sparse_missing(numeric)
        if numeric_filled.notna().all():
            return numeric_filled.round().astype(int).tolist()

    datetimes = pd.to_datetime(series, errors='coerce')
    if datetimes.notna().sum() == int(original_non_null.sum()) and datetimes.notna().any():
        epoch_seconds = pd.Series((datetimes.astype('int64') // 10**9), index=series.index, dtype='float64')
        epoch_filled = _fill_sparse_missing(epoch_seconds)
        if epoch_filled.notna().all():
            return epoch_filled.round().astype(int).tolist()

    raise ValueError('Unsupported time column format encountered during profiling.')


def _empty_window_profile() -> dict[str, object]:
    profile: dict[str, object] = {
        'window_count': 0,
        'fault_vector': '0|0|0|0|0',
        'warning_fault_vector': '0|0|0|0|0',
    }
    for fault in FAULT_ORDER:
        profile[f'y_id_{fault}_positive'] = 0
        profile[f'y_warn_{fault}_positive'] = 0
    return profile


def profile_window_labels(
    frame_labels: pd.DataFrame,
    window_size: int = WINDOW_SIZE_FRAMES,
    step_size: int = STEP_SIZE_FRAMES,
    forecast_horizon_frames: int = FORECAST_HORIZON_FRAMES,
    respect_quality_flag: bool = True,
) -> dict[str, object]:
    if frame_labels.empty:
        return _empty_window_profile()

    ordered = frame_labels.reset_index(drop=True).copy()
    max_start = len(ordered) - window_size - forecast_horizon_frames + 1
    if max_start < 1:
        return _empty_window_profile()

    starts = np.arange(0, max_start, step_size, dtype=int)
    ends = starts + window_size - 1

    segment_values = pd.to_numeric(ordered['segment_id'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64)
    valid_mask = segment_values[starts] == segment_values[ends]
    if respect_quality_flag:
        quality_values = ordered['quality_flag'].astype(str).str.lower().to_numpy(dtype=str)
        valid_mask &= quality_values[ends] != 'low'

    profile: dict[str, object] = {'window_count': int(valid_mask.sum())}
    id_bits: list[str] = []
    warn_bits: list[str] = []

    for fault in FAULT_ORDER:
        label_values = pd.to_numeric(ordered[FAULT_LABEL_COLUMNS[fault]], errors='coerce').fillna(0).astype(int).to_numpy(dtype=np.int64)
        id_sum = label_values[ends] + label_values[ends - 1] + label_values[ends - 2]
        id_positive = (id_sum >= 2) & valid_mask

        prefix = np.concatenate(([0], np.cumsum(label_values, dtype=np.int64)))
        future_end = ends + forecast_horizon_frames
        future_positive = (prefix[future_end + 1] - prefix[ends + 1]) > 0
        warn_positive = (~id_positive) & future_positive & valid_mask

        id_count = int(id_positive.sum())
        warn_count = int(warn_positive.sum())
        profile[f'y_id_{fault}_positive'] = id_count
        profile[f'y_warn_{fault}_positive'] = warn_count
        id_bits.append('1' if id_count > 0 else '0')
        warn_bits.append('1' if warn_count > 0 else '0')

    profile['fault_vector'] = '|'.join(id_bits)
    profile['warning_fault_vector'] = '|'.join(warn_bits)
    return profile


def profile_structured_vehicle(csv_path: Path, chunk_size: int = 50000) -> dict[str, object]:
    usecols = ['TIME', 'CHARGE_STATUS', *STRUCTURED_LABEL_COLUMNS.values()]
    total_rows = 0
    valid_rows = 0
    segment_count = 0
    previous_time = None
    charge_values = set()
    fault_counts = {fault: 0 for fault in STRUCTURED_LABEL_COLUMNS}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
        total_rows += len(chunk)
        valid_rows += len(chunk)
        charge_values.update(pd.to_numeric(chunk['CHARGE_STATUS'], errors='coerce').dropna().astype(int).tolist())
        times = _series_to_epoch_seconds(chunk['TIME'])
        for current_time in times:
            if previous_time is None or current_time - previous_time > MAX_SEGMENT_GAP_SECONDS:
                segment_count += 1
            previous_time = current_time
        for fault, column in STRUCTURED_LABEL_COLUMNS.items():
            fault_counts[fault] += int(pd.to_numeric(chunk[column], errors='coerce').fillna(0).astype(int).sum())

    fault_vector = '|'.join('1' if fault_counts[fault] > 0 else '0' for fault in STRUCTURED_LABEL_COLUMNS)
    approx_windows = max((valid_rows - 30) // 3 + 1, 0)
    return {
        'vehicle_id': csv_path.stem,
        'source_dataset': 'structured_dataset',
        'source_file': str(csv_path),
        'is_extracted': True,
        'total_rows': total_rows,
        'valid_rows': valid_rows,
        'segment_count': segment_count,
        'window_count': approx_windows,
        'charge_status_coverage': '|'.join(str(value) for value in sorted(charge_values)),
        'fault_vector': fault_vector,
        **{f'{fault}_positive': fault_counts[fault] for fault in fault_counts},
    }


def profile_raw_vehicle(csv_path: Path, vehicle_id: str, chunk_size: int = 50000) -> dict[str, object]:
    usecols = ['terminaltime', 'chargestatus']
    total_rows = 0
    valid_rows = 0
    segment_count = 0
    previous_time = None
    charge_values = set()

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
        total_rows += len(chunk)
        valid_rows += len(chunk)
        charge_values.update(pd.to_numeric(chunk['chargestatus'], errors='coerce').dropna().astype(int).tolist())
        times = _series_to_epoch_seconds(chunk['terminaltime'])
        for current_time in times:
            if previous_time is None or current_time - previous_time > MAX_SEGMENT_GAP_SECONDS:
                segment_count += 1
            previous_time = current_time

    approx_windows = max((valid_rows - 30) // 3 + 1, 0)
    return {
        'vehicle_id': vehicle_id,
        'source_dataset': 'raw_dataset',
        'source_file': str(csv_path),
        'is_extracted': True,
        'total_rows': total_rows,
        'valid_rows': valid_rows,
        'segment_count': segment_count,
        'window_count': approx_windows,
        'charge_status_coverage': '|'.join(str(value) for value in sorted(charge_values)),
        'fault_vector': '0|0|0|0|0',
        'sd_positive': 0,
        'isc_positive': 0,
        'conn_positive': 0,
        'samp_positive': 0,
        'ins_positive': 0,
    }


def build_project_profiles(
    structured_root: Path,
    raw_root: Path | None,
    final_labels_root: Path = FINAL_LABELS_ROOT,
    include_raw: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = scan_project_datasets(structured_root=structured_root, raw_root=raw_root if include_raw else None)

    label_summary_path = final_labels_root / 'summary.csv'
    label_summary = pd.read_csv(label_summary_path) if label_summary_path.exists() else pd.DataFrame()
    structured_summary = label_summary[label_summary['source_dataset'] == 'structured_dataset'].set_index('vehicle_id') if not label_summary.empty else pd.DataFrame()

    structured_profiles = []
    for _, row in manifest.loc[manifest['source_dataset'] == 'structured_dataset'].iterrows():
        vehicle_id = str(row['vehicle_id'])
        label_path = final_labels_root / 'per_vehicle' / f"{vehicle_id}_labels.csv"
        if not label_path.exists():
            continue

        label_df = pd.read_csv(
            label_path,
            usecols=['timestamp', 'frame_index', 'segment_id', 'quality_flag', *FAULT_LABEL_COLUMNS.values()],
        )
        window_profile = profile_window_labels(label_df)
        summary_row = structured_summary.loc[vehicle_id] if not structured_summary.empty and vehicle_id in structured_summary.index else None
        frame_fault_counts = {}
        for fault in FAULT_ORDER:
            if summary_row is not None:
                frame_fault_counts[f'{fault}_positive'] = int(summary_row.get(f'{fault}_positive', 0) or 0)
            else:
                frame_fault_counts[f'{fault}_positive'] = int(pd.to_numeric(label_df[FAULT_LABEL_COLUMNS[fault]], errors='coerce').fillna(0).astype(int).sum())

        profile = {
            'vehicle_id': vehicle_id,
            'source_dataset': 'structured_dataset',
            'source_file': str(summary_row['source_file']) if summary_row is not None and 'source_file' in summary_row else str(row['source_file']),
            'is_extracted': True,
            'total_rows': int(summary_row['total_rows']) if summary_row is not None and 'total_rows' in summary_row else int(len(label_df)),
            'valid_rows': int(len(label_df)),
            'segment_count': int(pd.to_numeric(label_df['segment_id'], errors='coerce').fillna(-1).nunique()),
            'window_count': int(window_profile['window_count']),
            'charge_status_coverage': '',
            'fault_vector': str(window_profile['fault_vector']),
            **frame_fault_counts,
            **window_profile,
        }
        structured_profiles.append(profile)

    raw_profiles = []
    if include_raw:
        raw_rows = manifest[(manifest['source_dataset'] == 'raw_dataset') & (manifest['is_extracted'] == True)]
        for _, row in raw_rows.iterrows():
            raw_profiles.append(profile_raw_vehicle(Path(row['source_file']), row['vehicle_id']))

    structured_df = pd.DataFrame(structured_profiles).sort_values('vehicle_id').reset_index(drop=True)
    raw_df = pd.DataFrame(raw_profiles).sort_values('vehicle_id').reset_index(drop=True) if raw_profiles else pd.DataFrame()
    return manifest, structured_df, raw_df


def build_structured_split(profile_df: pd.DataFrame, seed: int = 20260407) -> pd.DataFrame:
    return assign_vehicle_splits(
        profile_df.copy(),
        train_count=12,
        val_count=4,
        test_count=4,
        seed=seed,
        prioritize_fault_coverage=True,
        required_warning_faults=['sd', 'samp', 'ins'],
        core_id_faults=['sd', 'samp', 'ins'],
        rare_fault_rules={'isc': 'train_plus_eval', 'conn': 'eval_only'},
    )
