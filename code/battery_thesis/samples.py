from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FAULT_LABEL_COLUMNS, FAULT_ORDER
from .labels import derive_identification_label, derive_warning_label


SEQUENCE_FEATURE_COLUMNS = [
    'charge_status',
    'soc',
    'speed',
    'sum_voltage',
    'sum_current',
    'insulation_resistance',
    'v_range',
    't_range',
]


LABEL_COLUMNS = [FAULT_LABEL_COLUMNS[fault] for fault in FAULT_ORDER]
IDENTIFICATION_COLUMNS = [f'y_id_{fault}' for fault in FAULT_ORDER]
WARNING_COLUMNS = [f'y_warn_{fault}' for fault in FAULT_ORDER]

CORE_ID_FAULTS = ['sd', 'samp', 'ins']
CORE_WARNING_FAULTS = ['sd', 'samp', 'ins']
RARE_FAULT_RULES = {'isc': 'train_plus_eval', 'conn': 'eval_only'}


def _sanitize_numeric_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def build_samples_master(
    frame_labels: pd.DataFrame,
    vehicle_id: str,
    source_dataset: str,
    split: str,
    window_size: int,
    step_size: int,
    forecast_horizon_frames: int,
    frame_interval_sec: int,
    respect_quality_flag: bool = True,
) -> pd.DataFrame:
    ordered = frame_labels.reset_index(drop=True)
    max_start = len(ordered) - window_size - forecast_horizon_frames + 1
    if max_start < 1:
        return pd.DataFrame()

    starts = np.arange(0, max_start, step_size, dtype=int)
    ends = starts + window_size - 1

    segment_values = pd.to_numeric(ordered['segment_id'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64)
    valid_mask = segment_values[starts] == segment_values[ends]
    if respect_quality_flag:
        quality_values = ordered['quality_flag'].astype(str).str.lower().to_numpy(dtype=str)
        valid_mask &= quality_values[ends] != 'low'

    starts = starts[valid_mask]
    ends = ends[valid_mask]
    if len(starts) == 0:
        return pd.DataFrame()

    timestamps = pd.to_numeric(ordered['timestamp'], errors='coerce').fillna(0).astype(float).astype(int).to_numpy(dtype=np.int64)
    frame_indices = pd.to_numeric(ordered['frame_index'], errors='coerce').fillna(0).astype(int).to_numpy(dtype=np.int64)
    quality_values = ordered['quality_flag'].astype(str).to_numpy(dtype=object)

    sample_count = len(starts)
    samples = pd.DataFrame(
        {
            'sample_id': [f'{vehicle_id}_{idx:06d}' for idx in range(sample_count)],
            'vehicle_id': vehicle_id,
            'source_dataset': source_dataset,
            'segment_id': segment_values[ends],
            'start_time': timestamps[starts],
            'end_time': timestamps[ends],
            'start_frame': frame_indices[starts],
            'end_frame': frame_indices[ends],
            'split': split,
            'quality_flag': quality_values[ends],
        }
    )

    sentinel = forecast_horizon_frames * frame_interval_sec + frame_interval_sec
    lead_time_matrix = np.full((sample_count, len(FAULT_ORDER)), sentinel, dtype=np.int64)

    for fault_idx, fault in enumerate(FAULT_ORDER):
        label_col = FAULT_LABEL_COLUMNS[fault]
        label_values = pd.to_numeric(ordered[label_col], errors='coerce').fillna(0).astype(int).to_numpy(dtype=np.int64)
        id_labels = (label_values[ends] + label_values[ends - 1] + label_values[ends - 2] >= 2)

        future_matrix = np.column_stack(
            [label_values[ends + 1 + offset] for offset in range(forecast_horizon_frames)]
        )
        future_any = future_matrix.any(axis=1)
        first_future_idx = future_matrix.argmax(axis=1) + 1
        warn_labels = (~id_labels) & future_any
        lead_times = np.where(warn_labels, first_future_idx * frame_interval_sec, sentinel)

        samples[f'y_id_{fault}'] = id_labels.astype(int)
        samples[f'y_warn_{fault}'] = warn_labels.astype(int)
        lead_time_matrix[:, fault_idx] = lead_times

    min_leads = lead_time_matrix.min(axis=1)
    first_fault_indices = lead_time_matrix.argmin(axis=1)
    samples['future_first_fault'] = [FAULT_ORDER[idx] if min_leads[pos] < sentinel else None for pos, idx in enumerate(first_fault_indices)]
    samples['lead_time_sec'] = [int(min_leads[pos]) if min_leads[pos] < sentinel else None for pos in range(sample_count)]
    return samples


def compute_window_features(window: pd.DataFrame) -> dict[str, float]:
    voltages = _sanitize_numeric_array(window['voltages'].tolist())
    temperatures = _sanitize_numeric_array(window['temperatures'].tolist())
    insulation = _sanitize_numeric_array(pd.to_numeric(window['insulation_resistance'], errors='coerce').fillna(0.0).to_numpy(dtype=float))
    sum_voltage = _sanitize_numeric_array(pd.to_numeric(window['sum_voltage'], errors='coerce').fillna(0.0).to_numpy(dtype=float))

    cell_medians = np.median(voltages, axis=1)
    weakest_offsets = np.min(voltages - cell_medians[:, None], axis=1)
    voltage_change = np.diff(voltages, axis=0) if len(voltages) > 1 else np.zeros((0, voltages.shape[1]))
    voltage_ranges = np.max(voltages, axis=1) - np.min(voltages, axis=1)
    temp_ranges = np.max(temperatures, axis=1) - np.min(temperatures, axis=1)
    temperature_max = np.max(temperatures, axis=1)
    voltage_min = np.min(voltages, axis=1)

    features = {
        'shared_charge_status': float(window['charge_status'].iloc[-1]),
        'shared_soc': float(window['soc'].iloc[-1]),
        'shared_speed': float(window['speed'].iloc[-1]),
        'shared_sum_voltage': float(window['sum_voltage'].iloc[-1]),
        'shared_sum_current': float(window['sum_current'].iloc[-1]),
        'shared_insulation_resistance': float(insulation[-1]),
        'shared_v_range': round(float(np.max(voltage_ranges)), 6),
        'shared_t_range': round(float(np.max(temp_ranges)), 6),
        'sd_cell_dev_min': round(float(np.min(weakest_offsets)), 6),
        'sd_cell_dev_slope': round(float(weakest_offsets[-1] - weakest_offsets[0]), 6),
        'sd_weak_cell_repeat_ratio': round(float(_weak_cell_repeat_ratio(voltages)), 6),
        'sd_rest_drop_rate': round(float(voltage_min[-1] - voltage_min[0]), 6),
        'isc_module_vdrop_max': round(float(np.min(np.diff(voltage_min))) if len(voltage_min) > 1 else 0.0, 6),
        'isc_module_vrange': round(float(np.max(voltage_ranges)), 6),
        'isc_module_tmax': round(float(np.max(temperatures)), 6),
        'isc_module_trise_max': round(float(np.max(np.diff(temperature_max))) if len(temperature_max) > 1 else 0.0, 6),
        'conn_dvdt_max': round(float(np.max(np.abs(voltage_change))) if voltage_change.size else 0.0, 6),
        'conn_icc_min': round(float(_adjacent_icc_min(voltages)), 6),
        'conn_neighbor_response_gap': round(float(_neighbor_response_gap(voltages)), 6),
        'conn_abnormal_point_ratio': round(float(np.mean(np.abs(voltage_change) > 0.015)) if voltage_change.size else 0.0, 6),
        'samp_local_median_residual': round(float(np.mean(np.abs(voltages - cell_medians[:, None]))), 6),
        'samp_sensor_jump_count': float(np.sum(np.abs(voltage_change) > 0.05)) if voltage_change.size else 0.0,
        'samp_pack_sum_residual': round(float(np.mean(sum_voltage - np.sum(voltages, axis=1))), 6),
        'samp_abnormal_count_ratio': round(float(np.mean(np.abs(voltages - cell_medians[:, None]) > 0.03)), 6),
        'ins_rins_min': round(float(insulation.min()), 6),
        'ins_rins_slope': round(float(insulation[-1] - insulation[0]), 6),
        'ins_threshold_margin': round(float(insulation[-1] - _insulation_threshold(window['charge_status'].iloc[-1], sum_voltage[-1])), 6),
        'ins_low_rins_persistence': float((insulation < np.median(insulation)).sum()),
    }
    return features


def assemble_sample_artifacts(
    frame_df: pd.DataFrame,
    vehicle_id: str,
    source_dataset: str,
    split: str,
    window_size: int,
    step_size: int,
    forecast_horizon_frames: int,
    frame_interval_sec: int,
    respect_quality_flag: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    samples_master = build_samples_master(
        frame_labels=frame_df[['timestamp', 'frame_index', 'segment_id', 'quality_flag', *LABEL_COLUMNS]].copy(),
        vehicle_id=vehicle_id,
        source_dataset=source_dataset,
        split=split,
        window_size=window_size,
        step_size=step_size,
        forecast_horizon_frames=forecast_horizon_frames,
        frame_interval_sec=frame_interval_sec,
        respect_quality_flag=respect_quality_flag,
    )

    feature_rows: list[dict[str, Any]] = []
    sequence_rows: list[np.ndarray] = []
    for sample in samples_master.to_dict('records'):
        start_idx = int(sample['start_frame'])
        end_idx = int(sample['end_frame']) + 1
        window = frame_df.iloc[start_idx:end_idx].copy()
        feature_rows.append({'sample_id': sample['sample_id'], **compute_window_features(window)})
        sequence_rows.append(_window_to_sequence(window))

    features_all = pd.DataFrame(feature_rows)
    if not features_all.empty:
        numeric_columns = [column for column in features_all.columns if column != 'sample_id']
        features_all[numeric_columns] = (
            features_all[numeric_columns]
            .apply(pd.to_numeric, errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    feature_matrix = _sanitize_numeric_array(features_all.drop(columns=['sample_id']).to_numpy(dtype=float)) if not features_all.empty else np.zeros((0, 0), dtype=float)
    sequence_tensor = _sanitize_numeric_array(np.stack(sequence_rows)).astype(np.float32) if sequence_rows else np.zeros((0, window_size, len(SEQUENCE_FEATURE_COLUMNS)), dtype=np.float32)
    tensors = {
        'X_seq': sequence_tensor,
        'X_feat': feature_matrix.astype(np.float32, copy=False),
        'y_id': samples_master[IDENTIFICATION_COLUMNS].to_numpy(dtype=np.int64) if not samples_master.empty else np.zeros((0, len(IDENTIFICATION_COLUMNS)), dtype=np.int64),
        'y_warn': samples_master[WARNING_COLUMNS].to_numpy(dtype=np.int64) if not samples_master.empty else np.zeros((0, len(WARNING_COLUMNS)), dtype=np.int64),
        'sample_id': samples_master['sample_id'].to_numpy(dtype=str) if not samples_master.empty else np.asarray([], dtype=str),
        'sequence_feature_columns': np.asarray(SEQUENCE_FEATURE_COLUMNS, dtype=str),
        'feature_columns': np.asarray(features_all.columns[1:].tolist(), dtype=str) if not features_all.empty else np.asarray([], dtype=str),
    }
    return samples_master, features_all, tensors


def save_sample_artifacts(
    output_root: Path,
    samples_master: pd.DataFrame,
    features_all: pd.DataFrame,
    tensors: dict[str, np.ndarray],
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    samples_path = output_root / 'samples_master.csv'
    features_path = output_root / 'features_all.csv'
    dataset_pack_path = output_root / 'dataset_pack.npz'

    samples_master.to_csv(samples_path, index=False, encoding='utf-8-sig')
    features_all.to_csv(features_path, index=False, encoding='utf-8-sig')
    np.savez_compressed(dataset_pack_path, **tensors)

    return {
        'samples_master': samples_path,
        'features_all': features_path,
        'dataset_pack': dataset_pack_path,
    }


def _weak_cell_repeat_ratio(voltages: np.ndarray) -> float:
    weakest = np.argmin(voltages, axis=1)
    _, counts = np.unique(weakest, return_counts=True)
    return counts.max() / len(weakest)


def _adjacent_icc_min(voltages: np.ndarray) -> float:
    if voltages.shape[1] < 2:
        return 1.0
    iccs = []
    for idx in range(voltages.shape[1] - 1):
        x1 = voltages[:, idx]
        x2 = voltages[:, idx + 1]
        x_bar = (np.sum(x1) + np.sum(x2)) / (2 * len(x1))
        s2 = (np.sum((x1 - x_bar) ** 2) + np.sum((x2 - x_bar) ** 2)) / (2 * len(x1))
        if s2 == 0:
            iccs.append(1.0)
            continue
        covariance = np.sum((x1 - x_bar) * (x2 - x_bar))
        iccs.append(covariance / (len(x1) * s2))
    return min(iccs)


def _neighbor_response_gap(voltages: np.ndarray) -> float:
    if voltages.shape[1] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(voltages, axis=1))))


def _insulation_threshold(charge_status: float, sum_voltage: float) -> float:
    if int(charge_status) in {1, 2}:
        return 100 * float(sum_voltage)
    return 500 * float(sum_voltage)


def _window_to_sequence(window: pd.DataFrame) -> np.ndarray:
    voltages = _sanitize_numeric_array(window['voltages'].tolist())
    temperatures = _sanitize_numeric_array(window['temperatures'].tolist())
    v_range = np.max(voltages, axis=1) - np.min(voltages, axis=1)
    t_range = np.max(temperatures, axis=1) - np.min(temperatures, axis=1)
    matrix = np.column_stack(
        [
            pd.to_numeric(window['charge_status'], errors='coerce').fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(window['soc'], errors='coerce').fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(window['speed'], errors='coerce').fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(window['sum_voltage'], errors='coerce').fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(window['sum_current'], errors='coerce').fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(window['insulation_resistance'], errors='coerce').fillna(0.0).to_numpy(dtype=float),
            v_range,
            t_range,
        ]
    )
    return _sanitize_numeric_array(matrix).astype(np.float32)


def downsample_split_samples(
    samples: pd.DataFrame,
    split: str,
    target_count: int,
    seed: int,
    required_identification_faults: list[str],
    required_warning_faults: list[str],
) -> pd.DataFrame:
    split_df = samples[samples['split'] == split].copy()
    if split_df.empty:
        return split_df

    rng = np.random.default_rng(seed)
    selected_indices: set[int] = set()

    for fault in required_identification_faults:
        fault_df = split_df[split_df[f'y_id_{fault}'] == 1]
        if not fault_df.empty:
            selected_indices.add(int(rng.choice(fault_df.index.to_numpy(dtype=int))))

    for fault in required_warning_faults:
        fault_df = split_df[split_df[f'y_warn_{fault}'] == 1]
        if not fault_df.empty:
            selected_indices.add(int(rng.choice(fault_df.index.to_numpy(dtype=int))))

    positive_mask = np.zeros(len(split_df), dtype=bool)
    for fault in FAULT_ORDER:
        positive_mask |= split_df[f'y_id_{fault}'].to_numpy(dtype=int) == 1
        positive_mask |= split_df[f'y_warn_{fault}'].to_numpy(dtype=int) == 1

    positive_df = split_df.loc[positive_mask].copy()
    selected_indices.update(int(index) for index in positive_df.index.tolist())

    selected_df = split_df.loc[sorted(selected_indices)].copy()
    if len(selected_df) >= target_count:
        return selected_df.sort_values(['vehicle_id', 'sample_id']).reset_index(drop=True)

    negative_pool = split_df.loc[(~positive_mask) & (~split_df.index.isin(selected_indices))].copy()
    if negative_pool.empty:
        return selected_df.sort_values(['vehicle_id', 'sample_id']).reset_index(drop=True)

    remaining = target_count - len(selected_df)
    negative_selected = _sample_negative_rows_by_vehicle(negative_pool, remaining, rng)
    combined = pd.concat([selected_df, negative_selected], ignore_index=False)
    return combined.sort_values(['vehicle_id', 'sample_id']).reset_index(drop=True)


def _sample_negative_rows_by_vehicle(negative_pool: pd.DataFrame, target_count: int, rng: np.random.Generator) -> pd.DataFrame:
    if target_count <= 0 or negative_pool.empty:
        return negative_pool.iloc[0:0].copy()

    vehicle_groups = negative_pool.groupby('vehicle_id')
    vehicle_sizes = vehicle_groups.size().sort_index()
    total = int(vehicle_sizes.sum())
    allocations: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []

    for vehicle_id, size in vehicle_sizes.items():
        raw_share = target_count * (int(size) / total)
        base = min(int(size), int(np.floor(raw_share)))
        allocations[str(vehicle_id)] = base
        fractional.append((raw_share - base, str(vehicle_id)))

    assigned = sum(allocations.values())
    for _, vehicle_id in sorted(fractional, reverse=True):
        if assigned >= target_count:
            break
        available = int(vehicle_sizes[vehicle_id]) - allocations[vehicle_id]
        if available <= 0:
            continue
        allocations[vehicle_id] += 1
        assigned += 1

    selected_parts = []
    for vehicle_id, group_df in vehicle_groups:
        take_n = allocations.get(str(vehicle_id), 0)
        if take_n <= 0:
            continue
        chosen = rng.choice(group_df.index.to_numpy(dtype=int), size=take_n, replace=False)
        selected_parts.append(group_df.loc[np.sort(chosen)])

    if not selected_parts:
        return negative_pool.iloc[0:0].copy()
    return pd.concat(selected_parts, ignore_index=False)


def summarize_sample_selection(samples_master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ['train', 'val', 'test', 'external_test']:
        split_df = samples_master[samples_master['split'] == split]
        if split_df.empty:
            continue
        row: dict[str, int | str] = {'split': split, 'sample_count': int(len(split_df))}
        for fault in FAULT_ORDER:
            row[f'y_id_{fault}'] = int(split_df[f'y_id_{fault}'].sum())
            row[f'y_warn_{fault}'] = int(split_df[f'y_warn_{fault}'].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def validate_medium_split_coverage(
    summary_df: pd.DataFrame,
    core_id_faults: list[str] | None = None,
    core_warning_faults: list[str] | None = None,
    rare_fault_rules: dict[str, str] | None = None,
    raise_on_blocking: bool = True,
) -> dict[str, object]:
    core_id_faults = core_id_faults or CORE_ID_FAULTS
    core_warning_faults = core_warning_faults or CORE_WARNING_FAULTS
    rare_fault_rules = rare_fault_rules or RARE_FAULT_RULES

    blocking_errors: list[str] = []
    rare_fault_notes: list[str] = []
    data_limitations: list[str] = []

    for split in ['train', 'val', 'test']:
        split_rows = summary_df[summary_df['split'] == split]
        if split_rows.empty:
            blocking_errors.append(f'{split}:sample_count')
            continue
        row = split_rows.iloc[0]
        if int(row.get('sample_count', 0) or 0) <= 0:
            blocking_errors.append(f'{split}:sample_count')
        for fault in core_id_faults:
            if int(row.get(f'y_id_{fault}', 0) or 0) <= 0:
                blocking_errors.append(f'{split}:y_id_{fault}')
        if split in {'val', 'test'}:
            for fault in core_warning_faults:
                if int(row.get(f'y_warn_{fault}', 0) or 0) <= 0:
                    blocking_errors.append(f'{split}:y_warn_{fault}')

    for fault, rule in rare_fault_rules.items():
        train_positive = int(summary_df.loc[summary_df['split'] == 'train', f'y_id_{fault}'].sum()) if f'y_id_{fault}' in summary_df.columns else 0
        val_positive = int(summary_df.loc[summary_df['split'] == 'val', f'y_id_{fault}'].sum()) if f'y_id_{fault}' in summary_df.columns else 0
        test_positive = int(summary_df.loc[summary_df['split'] == 'test', f'y_id_{fault}'].sum()) if f'y_id_{fault}' in summary_df.columns else 0
        eval_positive = val_positive + test_positive
        total_positive = train_positive + eval_positive

        if rule == 'train_plus_eval':
            if total_positive == 0:
                note = f'{fault}: 当前没有可用的窗口级正例样本。'
                rare_fault_notes.append(note)
                data_limitations.append(note)
            elif train_positive <= 0 or eval_positive <= 0:
                note = f'{fault}: 当前未同时满足训练集至少1个、验证或测试至少1个窗口级正例。'
                rare_fault_notes.append(note)
                data_limitations.append(note)
        elif rule == 'eval_only':
            if total_positive == 0:
                note = f'{fault}: 当前没有可用的窗口级正例样本。'
                rare_fault_notes.append(note)
                data_limitations.append(note)
            elif eval_positive <= 0:
                note = f'{fault}: 当前没有保留到验证集或测试集的窗口级正例。'
                rare_fault_notes.append(note)
                data_limitations.append(note)

    report = {
        'blocking_errors': blocking_errors,
        'rare_fault_notes': rare_fault_notes,
        'data_limitations': data_limitations,
        'core_passed': not blocking_errors,
        'can_enter_round2': not blocking_errors,
    }
    if blocking_errors and raise_on_blocking:
        raise ValueError('Medium split coverage check failed: ' + ', '.join(blocking_errors))
    return report


def summarize_vehicle_selection(samples_master: pd.DataFrame) -> pd.DataFrame:
    if samples_master.empty:
        return pd.DataFrame()
    return (
        samples_master.groupby(['split', 'vehicle_id'], as_index=False)
        .agg(sample_count=('sample_id', 'count'))
        .sort_values(['split', 'vehicle_id'])
        .reset_index(drop=True)
    )


def validate_full_split_consistency(
    samples_master: pd.DataFrame,
    split_mapping: pd.DataFrame,
    raise_on_blocking: bool = True,
) -> dict[str, object]:
    blocking_errors: list[str] = []

    if samples_master.empty:
        blocking_errors.append('samples_master:empty')
        report = {
            'blocking_errors': blocking_errors,
            'split_vehicle_counts': {},
            'sample_counts': {},
            'core_passed': False,
            'can_enter_round2': False,
        }
        if raise_on_blocking:
            raise ValueError('Full split consistency check failed: ' + ', '.join(blocking_errors))
        return report

    samples = samples_master.copy()
    split_map = split_mapping.copy()
    samples['vehicle_id'] = samples['vehicle_id'].astype(str)
    samples['split'] = samples['split'].astype(str)
    if 'source_dataset' not in samples.columns:
        samples['source_dataset'] = 'structured_dataset'
    samples['source_dataset'] = samples['source_dataset'].astype(str)
    split_map['vehicle_id'] = split_map['vehicle_id'].astype(str)
    split_map['split'] = split_map['split'].astype(str)

    sample_vehicle_split = (
        samples.groupby('vehicle_id')['split']
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    structured_samples = samples[samples['source_dataset'] == 'structured_dataset'].copy()
    structured_vehicle_split = (
        structured_samples.groupby('vehicle_id')['split']
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    mapping_dict = dict(zip(split_map['vehicle_id'], split_map['split']))

    for vehicle_id, splits in sample_vehicle_split.items():
        if len(splits) > 1:
            blocking_errors.append(f'vehicle_id {vehicle_id} spans multiple splits: {", ".join(splits)}')

    for vehicle_id, splits in structured_vehicle_split.items():
        if len(splits) > 1:
            continue
        sample_split = splits[0]
        mapped_split = mapping_dict.get(vehicle_id)
        if mapped_split is None:
            blocking_errors.append(f'vehicle_id {vehicle_id} missing from structured_vehicle_split.csv')
        elif mapped_split != sample_split:
            blocking_errors.append(f'vehicle_id {vehicle_id} split mismatch: samples={sample_split}, mapping={mapped_split}')

    sample_vehicle_ids = set(structured_vehicle_split.keys())
    mapping_vehicle_ids = set(mapping_dict.keys())
    for vehicle_id in sorted(mapping_vehicle_ids - sample_vehicle_ids):
        blocking_errors.append(f'vehicle_id {vehicle_id} missing from samples_master.csv')

    split_vehicle_counts = (
        samples.groupby('split')['vehicle_id']
        .nunique()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    sample_counts = (
        samples.groupby('split')['sample_id']
        .count()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    report = {
        'blocking_errors': blocking_errors,
        'split_vehicle_counts': split_vehicle_counts,
        'sample_counts': sample_counts,
        'core_passed': not blocking_errors,
        'can_enter_round2': not blocking_errors,
    }
    if blocking_errors and raise_on_blocking:
        raise ValueError('Full split consistency check failed: ' + ', '.join(blocking_errors))
    return report
