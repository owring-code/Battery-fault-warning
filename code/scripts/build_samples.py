from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import (
    DATASET_META_ROOT,
    FINAL_LABELS_ROOT,
    FORECAST_HORIZON_FRAMES,
    FRAME_INTERVAL_SECONDS,
    SAMPLES_ROOT,
    STEP_SIZE_FRAMES,
    WINDOW_SIZE_FRAMES,
    ensure_project_directories,
)
from battery_thesis.field_mapping import normalize_sensor_sequences, parse_sensor_series
from battery_thesis.samples import (
    FAULT_ORDER,
    SEQUENCE_FEATURE_COLUMNS,
    _window_to_sequence,
    build_samples_master,
    compute_window_features,
    downsample_split_samples,
    save_sample_artifacts,
    validate_full_split_consistency,
    validate_medium_split_coverage,
)

DEFAULT_STRUCTURED_ROOT = Path(r'F:\Data_set')
DEFAULT_RAW_ROOT = Path(r'F:\RAW_DATA\data')
REQUIRED_WARNING_FAULTS = ['sd', 'samp', 'ins']


def parse_selected_vehicle_ids(raw_value: str | None) -> set[str] | None:
    if raw_value is None or not raw_value.strip():
        return None
    return {value.strip() for value in raw_value.split(',') if value.strip()}


def _sorted_prefixed(columns: list[str], prefix: str) -> list[str]:
    return sorted((column for column in columns if column.startswith(prefix)), key=lambda value: int(value.split('_')[1]))



def load_structured_feature_frame(csv_path: Path) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    all_columns = header.columns.tolist()
    voltage_columns = _sorted_prefixed(all_columns, 'U_')
    temperature_columns = _sorted_prefixed(all_columns, 'T_')
    usecols = [
        'TIME',
        'CHARGE_STATUS',
        'SPEED',
        'SUM_VOLTAGE',
        'SUM_CURRENT',
        'SOC',
        'INSULATION_RESISTANCE',
        *voltage_columns,
        *temperature_columns,
    ]
    raw_df = pd.read_csv(csv_path, usecols=usecols)
    feature_df = pd.DataFrame(
        {
            'timestamp': pd.to_numeric(raw_df['TIME'], errors='coerce').fillna(0).astype(float).astype(int),
            'frame_index': np.arange(len(raw_df), dtype=int),
            'charge_status': pd.to_numeric(raw_df['CHARGE_STATUS'], errors='coerce').fillna(0).astype(int),
            'speed': pd.to_numeric(raw_df['SPEED'], errors='coerce').fillna(0.0),
            'sum_voltage': pd.to_numeric(raw_df['SUM_VOLTAGE'], errors='coerce').fillna(0.0),
            'sum_current': pd.to_numeric(raw_df['SUM_CURRENT'], errors='coerce').fillna(0.0),
            'soc': pd.to_numeric(raw_df['SOC'], errors='coerce').fillna(0.0),
            'insulation_resistance': pd.to_numeric(raw_df['INSULATION_RESISTANCE'], errors='coerce').fillna(0.0),
        }
    )
    feature_df['voltages'] = raw_df[voltage_columns].to_numpy(dtype=float).tolist()
    feature_df['temperatures'] = raw_df[temperature_columns].to_numpy(dtype=float).tolist()
    return feature_df



def load_raw_feature_frame(csv_path: Path) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    available_columns = header.columns.tolist()
    usecols = [
        'terminaltime',
        'soc',
        'speed',
        'chargestatus',
        'totalvoltage',
        'totalcurrent',
        'batteryvoltage',
        'probetemperatures',
        *(['insulationresistance'] if 'insulationresistance' in available_columns else []),
    ]
    raw_df = pd.read_csv(csv_path, usecols=usecols)
    voltage_lists = raw_df['batteryvoltage'].map(parse_sensor_series)
    temperature_lists = raw_df['probetemperatures'].map(parse_sensor_series)
    normalized_voltages, _, _ = normalize_sensor_sequences(voltage_lists)
    normalized_temperatures, _, _ = normalize_sensor_sequences(temperature_lists)
    insulation_values = (
        pd.to_numeric(raw_df['insulationresistance'], errors='coerce').fillna(0.0)
        if 'insulationresistance' in raw_df.columns
        else pd.Series(np.zeros(len(raw_df), dtype=float))
    )
    feature_df = pd.DataFrame(
        {
            'timestamp': pd.to_numeric(raw_df['terminaltime'], errors='coerce').fillna(0).astype(float).astype(int),
            'frame_index': np.arange(len(raw_df), dtype=int),
            'charge_status': pd.to_numeric(raw_df['chargestatus'], errors='coerce').fillna(0).astype(int),
            'speed': pd.to_numeric(raw_df['speed'], errors='coerce').fillna(0.0),
            'sum_voltage': pd.to_numeric(raw_df['totalvoltage'], errors='coerce').fillna(0.0),
            'sum_current': pd.to_numeric(raw_df['totalcurrent'], errors='coerce').fillna(0.0),
            'soc': pd.to_numeric(raw_df['soc'], errors='coerce').fillna(0.0),
            'insulation_resistance': insulation_values.astype(float),
        }
    )
    feature_df['voltages'] = normalized_voltages
    feature_df['temperatures'] = normalized_temperatures
    return feature_df



def merge_feature_and_label_frames(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    label_columns = [
        'segment_id',
        'quality_flag',
        'label_final_sd',
        'label_final_isc',
        'label_final_conn',
        'label_final_samp',
        'label_final_ins',
    ]
    if len(feature_df) != len(label_df):
        raise ValueError(f'Feature row count {len(feature_df)} does not match label row count {len(label_df)}.')
    merged = feature_df.reset_index(drop=True).copy()
    merged[label_columns] = label_df[label_columns].reset_index(drop=True)
    return merged



def load_split_mapping() -> dict[str, str]:
    split_path = DATASET_META_ROOT / 'structured_vehicle_split.csv'
    if not split_path.exists():
        raise FileNotFoundError(f'Missing structured split file: {split_path}')
    split_df = pd.read_csv(split_path)
    return dict(zip(split_df['vehicle_id'], split_df['split']))


def load_split_frame(selected_vehicle_ids: set[str] | None = None) -> pd.DataFrame:
    split_path = DATASET_META_ROOT / 'structured_vehicle_split.csv'
    if not split_path.exists():
        raise FileNotFoundError(f'Missing structured split file: {split_path}')
    split_df = pd.read_csv(split_path)
    split_df['vehicle_id'] = split_df['vehicle_id'].astype(str)
    split_df['split'] = split_df['split'].astype(str)
    if selected_vehicle_ids is not None:
        split_df = split_df[split_df['vehicle_id'].isin(selected_vehicle_ids)].copy()
    return split_df.reset_index(drop=True)



def resolve_source_csv(vehicle_id: str, source_dataset: str, structured_root: Path, raw_root: Path) -> Path | None:
    if source_dataset == 'structured_dataset':
        candidate = structured_root / f'{vehicle_id}.csv'
        return candidate if candidate.exists() else None
    nested_candidate = raw_root / vehicle_id / f'{vehicle_id}.csv'
    if nested_candidate.exists():
        return nested_candidate
    flat_candidate = raw_root / f'{vehicle_id}.csv'
    return flat_candidate if flat_candidate.exists() else None



def resolve_label_path(vehicle_id: str, source_dataset: str) -> Path:
    return FINAL_LABELS_ROOT / 'per_vehicle' / f'{vehicle_id}_labels.csv'



def load_label_frame(vehicle_id: str, source_dataset: str, label_kind: str, ignore_quality_flag: bool) -> pd.DataFrame:
    label_path = resolve_label_path(vehicle_id, source_dataset)
    label_df = pd.read_csv(label_path)
    if label_kind == 'raw':
        for fault in FAULT_ORDER:
            label_df[f'label_final_{fault}'] = pd.to_numeric(label_df[f'label_raw_{fault}'], errors='coerce').fillna(0).astype(int)
    else:
        for fault in FAULT_ORDER:
            label_df[f'label_final_{fault}'] = pd.to_numeric(label_df[f'label_final_{fault}'], errors='coerce').fillna(0).astype(int)
    if ignore_quality_flag:
        label_df['quality_flag'] = 'ok'
    keep_columns = ['timestamp', 'frame_index', 'segment_id', 'quality_flag', *[f'label_final_{fault}' for fault in FAULT_ORDER]]
    return label_df[keep_columns].copy()



def _iter_dataset_rows(include_raw: bool, selected_vehicle_ids: set[str] | None = None) -> list[dict[str, str]]:
    try:
        split_mapping = load_split_mapping()
    except FileNotFoundError:
        if not include_raw:
            raise
        split_mapping = {}
    dataset_rows = [
        {'vehicle_id': vehicle_id, 'source_dataset': 'structured_dataset', 'split': split}
        for vehicle_id, split in split_mapping.items()
        if selected_vehicle_ids is None or vehicle_id in selected_vehicle_ids
    ]
    if include_raw:
        final_summary = FINAL_LABELS_ROOT / 'summary.csv'
        if final_summary.exists():
            final_df = pd.read_csv(final_summary)
            raw_df = final_df[final_df['source_dataset'] == 'raw_dataset']
            for _, row in raw_df.iterrows():
                if selected_vehicle_ids is not None and row['vehicle_id'] not in selected_vehicle_ids:
                    continue
                dataset_rows.append({'vehicle_id': row['vehicle_id'], 'source_dataset': 'raw_dataset', 'split': 'external_test'})
    return dataset_rows



def build_sample_pool(include_raw: bool, label_kind: str, ignore_quality_flag: bool, selected_vehicle_ids: set[str] | None = None) -> pd.DataFrame:
    parts = []
    for row in _iter_dataset_rows(include_raw=include_raw, selected_vehicle_ids=selected_vehicle_ids):
        label_df = load_label_frame(
            vehicle_id=row['vehicle_id'],
            source_dataset=row['source_dataset'],
            label_kind=label_kind,
            ignore_quality_flag=ignore_quality_flag,
        )
        sample_df = build_samples_master(
            frame_labels=label_df,
            vehicle_id=row['vehicle_id'],
            source_dataset=row['source_dataset'],
            split=row['split'],
            window_size=WINDOW_SIZE_FRAMES,
            step_size=STEP_SIZE_FRAMES,
            forecast_horizon_frames=FORECAST_HORIZON_FRAMES,
            frame_interval_sec=FRAME_INTERVAL_SECONDS,
            respect_quality_flag=not ignore_quality_flag,
        )
        if not sample_df.empty:
            parts.append(sample_df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()



def collect_full_sample_artifacts(
    include_raw: bool,
    label_kind: str,
    ignore_quality_flag: bool,
    structured_root: Path,
    raw_root: Path,
    selected_vehicle_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    sample_pool = build_sample_pool(
        include_raw=include_raw,
        label_kind=label_kind,
        ignore_quality_flag=ignore_quality_flag,
        selected_vehicle_ids=selected_vehicle_ids,
    )
    return materialize_selected_samples(sample_pool, label_kind=label_kind, ignore_quality_flag=ignore_quality_flag, structured_root=structured_root, raw_root=raw_root)



def collect_medium_sample_artifacts(
    include_raw: bool,
    label_kind: str,
    ignore_quality_flag: bool,
    train_target: int,
    val_target: int,
    test_target: int,
    seed: int,
    structured_root: Path,
    raw_root: Path,
    selected_vehicle_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sample_pool = build_sample_pool(
        include_raw=include_raw,
        label_kind=label_kind,
        ignore_quality_flag=ignore_quality_flag,
        selected_vehicle_ids=selected_vehicle_ids,
    )
    if sample_pool.empty:
        empty_summary = pd.DataFrame()
        empty_report = validate_medium_split_coverage(empty_summary, raise_on_blocking=False)
        return pd.DataFrame(), pd.DataFrame(), _empty_tensors(), empty_summary, pd.DataFrame(), empty_report

    selected_parts = []
    split_targets = {'train': train_target, 'val': val_target, 'test': test_target}
    for split, target in split_targets.items():
        if split not in sample_pool['split'].unique():
            continue
        selected_parts.append(
            downsample_split_samples(
                sample_pool,
                split=split,
                target_count=target,
                seed=seed,
                required_identification_faults=FAULT_ORDER,
                required_warning_faults=REQUIRED_WARNING_FAULTS if split in {'val', 'test'} else [],
            )
        )

    external_test = sample_pool[sample_pool['split'] == 'external_test'].copy()
    if not external_test.empty:
        selected_parts.append(external_test)

    selected_samples = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    selected_samples = selected_samples.drop_duplicates(subset=['sample_id']).reset_index(drop=True)
    split_summary = summarize_sample_selection(selected_samples)
    selection_summary = summarize_vehicle_selection(selected_samples)
    validation_report = validate_medium_split_coverage(split_summary, raise_on_blocking=False)

    if validation_report['blocking_errors']:
        return pd.DataFrame(), pd.DataFrame(), _empty_tensors(), split_summary, selection_summary, validation_report

    samples_master, features_all, tensors = materialize_selected_samples(
        selected_samples,
        label_kind=label_kind,
        ignore_quality_flag=ignore_quality_flag,
        structured_root=structured_root,
        raw_root=raw_root,
    )
    return samples_master, features_all, tensors, split_summary, selection_summary, validation_report



def materialize_selected_samples(
    selected_samples: pd.DataFrame,
    label_kind: str,
    ignore_quality_flag: bool,
    structured_root: Path,
    raw_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    if selected_samples.empty:
        return pd.DataFrame(), pd.DataFrame(), _empty_tensors()

    split_rank = selected_samples['split'].map({'train': 0, 'val': 1, 'test': 2, 'external_test': 3}).fillna(99)
    ordered_samples = (
        selected_samples.assign(_split_rank=split_rank)
        .sort_values(['_split_rank', 'vehicle_id', 'sample_id'])
        .drop(columns=['_split_rank'])
        .reset_index(drop=True)
    )
    feature_rows = []
    sequence_rows = []
    y_id_rows = []
    y_warn_rows = []
    sample_ids = []

    for (vehicle_id, source_dataset), sample_group in ordered_samples.groupby(['vehicle_id', 'source_dataset'], sort=False):
        source_csv = resolve_source_csv(vehicle_id, source_dataset, structured_root=structured_root, raw_root=raw_root)
        if source_csv is None:
            continue
        label_df = load_label_frame(vehicle_id, source_dataset, label_kind=label_kind, ignore_quality_flag=ignore_quality_flag)
        feature_df = load_structured_feature_frame(source_csv) if source_dataset == 'structured_dataset' else load_raw_feature_frame(source_csv)
        frame_df = merge_feature_and_label_frames(feature_df, label_df)

        for _, sample in sample_group.iterrows():
            start_idx = int(sample['start_frame'])
            end_idx = int(sample['end_frame']) + 1
            window = frame_df.iloc[start_idx:end_idx].copy()
            feature_row = {'sample_id': sample['sample_id'], **compute_window_features(window)}
            feature_rows.append(feature_row)
            sequence_rows.append(_window_to_sequence(window))
            y_id_rows.append([int(sample[f'y_id_{fault}']) for fault in FAULT_ORDER])
            y_warn_rows.append([int(sample[f'y_warn_{fault}']) for fault in FAULT_ORDER])
            sample_ids.append(sample['sample_id'])

    samples_master = ordered_samples.copy()
    features_all = pd.DataFrame(feature_rows)
    if sample_ids and len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Duplicate sample_id encountered while materializing selected samples.')
    if sample_ids:
        sample_order = pd.Index(sample_ids, dtype=object)
        reorder_index = sample_order.get_indexer(samples_master['sample_id'])
        if (reorder_index < 0).any():
            raise ValueError('Failed to align dataset tensors with samples_master ordering.')
        sequence_array = np.stack(sequence_rows).astype(np.float32)[reorder_index]
        y_id_array = np.asarray(y_id_rows, dtype=np.int64)[reorder_index]
        y_warn_array = np.asarray(y_warn_rows, dtype=np.int64)[reorder_index]
        sample_id_array = sample_order.to_numpy(dtype=str)[reorder_index]
        features_all = features_all.set_index('sample_id').loc[samples_master['sample_id']].reset_index()
    else:
        sequence_array = np.zeros((0, WINDOW_SIZE_FRAMES, len(SEQUENCE_FEATURE_COLUMNS)), dtype=np.float32)
        y_id_array = np.zeros((0, len(FAULT_ORDER)), dtype=np.int64)
        y_warn_array = np.zeros((0, len(FAULT_ORDER)), dtype=np.int64)
        sample_id_array = np.asarray([], dtype=str)
    tensors = {
        'X_seq': sequence_array,
        'X_feat': features_all.drop(columns=['sample_id']).to_numpy(dtype=np.float32) if not features_all.empty else np.zeros((0, 0), dtype=np.float32),
        'y_id': y_id_array,
        'y_warn': y_warn_array,
        'sample_id': sample_id_array,
        'feature_columns': np.asarray(features_all.columns[1:].tolist(), dtype=str) if not features_all.empty else np.asarray([], dtype=str),
        'sequence_feature_columns': np.asarray(SEQUENCE_FEATURE_COLUMNS, dtype=str),
    }
    return samples_master, features_all, tensors



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



def summarize_vehicle_selection(samples_master: pd.DataFrame) -> pd.DataFrame:
    if samples_master.empty:
        return pd.DataFrame()
    return (
        samples_master.groupby(['split', 'vehicle_id'], as_index=False)
        .agg(sample_count=('sample_id', 'count'))
        .sort_values(['split', 'vehicle_id'])
        .reset_index(drop=True)
    )



def _empty_tensors() -> dict[str, np.ndarray]:
    return {
        'X_seq': np.zeros((0, WINDOW_SIZE_FRAMES, len(SEQUENCE_FEATURE_COLUMNS)), dtype=np.float32),
        'X_feat': np.zeros((0, 0), dtype=np.float32),
        'y_id': np.zeros((0, len(FAULT_ORDER)), dtype=np.int64),
        'y_warn': np.zeros((0, len(FAULT_ORDER)), dtype=np.int64),
        'sample_id': np.asarray([], dtype=str),
        'feature_columns': np.asarray([], dtype=str),
        'sequence_feature_columns': np.asarray(SEQUENCE_FEATURE_COLUMNS, dtype=str),
    }



def collect_external_validation_sample_artifacts(
    label_kind: str,
    ignore_quality_flag: bool,
    raw_root: Path,
    selected_vehicle_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    final_summary_path = FINAL_LABELS_ROOT / 'summary.csv'
    if not final_summary_path.exists():
        raise FileNotFoundError(f'Missing final label summary: {final_summary_path}')

    final_summary = pd.read_csv(final_summary_path)
    raw_vehicle_ids = set(final_summary.loc[final_summary['source_dataset'] == 'raw_dataset', 'vehicle_id'].astype(str))
    if selected_vehicle_ids is not None:
        raw_vehicle_ids &= set(selected_vehicle_ids)
    if not raw_vehicle_ids:
        return pd.DataFrame(), pd.DataFrame(), _empty_tensors(), pd.DataFrame(), pd.DataFrame()

    sample_pool = build_sample_pool(
        include_raw=True,
        label_kind=label_kind,
        ignore_quality_flag=ignore_quality_flag,
        selected_vehicle_ids=raw_vehicle_ids,
    )
    external_samples = sample_pool[
        (sample_pool['source_dataset'] == 'raw_dataset') & (sample_pool['split'] == 'external_test')
    ].copy()
    external_samples = external_samples.sort_values(['vehicle_id', 'sample_id']).reset_index(drop=True)
    samples_master, features_all, tensors = materialize_selected_samples(
        external_samples,
        label_kind=label_kind,
        ignore_quality_flag=ignore_quality_flag,
        structured_root=Path('.'),
        raw_root=raw_root,
    )
    split_summary = summarize_sample_selection(samples_master)
    selection_summary = summarize_vehicle_selection(samples_master)
    return samples_master, features_all, tensors, split_summary, selection_summary



def write_full_reports(
    output_root: Path,
    split_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    validation_report: dict[str, object],
    args: argparse.Namespace,
) -> None:
    split_summary.to_csv(output_root / 'full_scale_summary.csv', index=False, encoding='utf-8-sig')
    selection_summary.to_csv(output_root / 'full_scale_selection.csv', index=False, encoding='utf-8-sig')

    lines = [
        '# Full Scale Build Notes',
        '',
        f'- mode: {args.mode}',
        f'- label_kind: {args.label_kind}',
        f'- ignore_quality_flag: {args.ignore_quality_flag}',
        f'- include_raw: {args.include_raw}',
        '- Data_set full-data build is the authoritative source for round2 formal experiments.',
        *( ['- RAW_DATA extracted vehicles are included as external_test for formal external validation.'] if args.include_raw else ['- RAW_DATA external validation is deferred and is not part of this round2 full-data package.'] ),
        '',
        '## Split Consistency',
        '',
    ]
    blocking_errors = list(validation_report.get('blocking_errors', []))
    if blocking_errors:
        lines.extend(f'- blocking: {item}' for item in blocking_errors)
    else:
        lines.append('- blocking: none')

    lines.extend(['', '## Vehicle Counts', ''])
    split_vehicle_counts = dict(validation_report.get('split_vehicle_counts', {}))
    if split_vehicle_counts:
        for split, count in split_vehicle_counts.items():
            lines.append(f'- {split}: {count} vehicles')
    else:
        lines.append('- none')

    lines.extend(['', '## Sample Counts', ''])
    sample_counts = dict(validation_report.get('sample_counts', {}))
    if sample_counts:
        for split, count in sample_counts.items():
            lines.append(f'- {split}: {count} samples')
    else:
        lines.append('- none')

    (output_root / 'full_scale_notes.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_medium_reports(
    output_root: Path,
    split_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    validation_report: dict[str, object],
    args: argparse.Namespace,
) -> None:
    split_summary.to_csv(output_root / 'medium_scale_summary.csv', index=False, encoding='utf-8-sig')
    selection_summary.to_csv(output_root / 'medium_scale_selection.csv', index=False, encoding='utf-8-sig')

    lines = [
        '# Medium Scale Build Notes',
        '',
        f'- mode: {args.mode}',
        f'- label_kind: {args.label_kind}',
        f'- ignore_quality_flag: {args.ignore_quality_flag}',
        f'- include_raw: {args.include_raw}',
        f'- train_target: {args.train_target}',
        f'- val_target: {args.val_target}',
        f'- test_target: {args.test_target}',
        f'- seed: {args.seed}',
        '- all positive windows are retained within each split; negatives are downsampled by vehicle proportion.',
        '- round 1 now uses hard acceptance for sd/samp/ins and soft retention for isc/conn.',
        '',
        '## Core Acceptance',
        '',
    ]
    blocking_errors = list(validation_report.get('blocking_errors', []))
    if blocking_errors:
        lines.extend(f'- blocking: {item}' for item in blocking_errors)
    else:
        lines.append('- blocking: none')

    lines.extend(['', '## Rare Fault Notes', ''])
    rare_fault_notes = list(validation_report.get('rare_fault_notes', []))
    if rare_fault_notes:
        lines.extend(f'- {item}' for item in rare_fault_notes)
    else:
        lines.append('- none')

    lines.extend(['', '## Data Limitations', ''])
    data_limitations = list(validation_report.get('data_limitations', []))
    if data_limitations:
        lines.extend(f'- {item}' for item in data_limitations)
    else:
        lines.append('- none')

    (output_root / 'medium_scale_notes.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')



def main() -> None:
    parser = argparse.ArgumentParser(description='Build samples_master.csv, features_all.csv, and dataset_pack.npz')
    parser.add_argument('--mode', choices=['full', 'medium'], default='full')
    parser.add_argument('--samples-root', default=str(SAMPLES_ROOT))
    parser.add_argument('--structured-root', default=str(DEFAULT_STRUCTURED_ROOT))
    parser.add_argument('--raw-root', default=str(DEFAULT_RAW_ROOT))
    parser.add_argument('--include-raw', action='store_true', help='Include extracted RAW_DATA vehicles as external_test samples.')
    parser.add_argument('--vehicle-ids', default=None, help='Optional comma-separated vehicle ids for fast local verification runs.')
    parser.add_argument('--label-kind', choices=['final', 'raw'], default='final')
    parser.add_argument('--ignore-quality-flag', action='store_true')
    parser.add_argument('--train-target', type=int, default=12000)
    parser.add_argument('--val-target', type=int, default=3000)
    parser.add_argument('--test-target', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=20260407)
    args = parser.parse_args()

    ensure_project_directories()
    output_root = Path(args.samples_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_vehicle_ids = parse_selected_vehicle_ids(args.vehicle_ids)
    structured_root = Path(args.structured_root)
    raw_root = Path(args.raw_root)

    if args.mode == 'medium':
        samples_master_df, features_all_df, tensors, split_summary, selection_summary, validation_report = collect_medium_sample_artifacts(
            include_raw=args.include_raw,
            label_kind=args.label_kind,
            ignore_quality_flag=args.ignore_quality_flag,
            train_target=args.train_target,
            val_target=args.val_target,
            test_target=args.test_target,
            seed=args.seed,
            structured_root=structured_root,
            raw_root=raw_root,
            selected_vehicle_ids=selected_vehicle_ids,
        )
        write_medium_reports(output_root, split_summary, selection_summary, validation_report, args)
        if validation_report['blocking_errors']:
            raise ValueError('Medium split coverage check failed: ' + ', '.join(validation_report['blocking_errors']))
    else:
        samples_master_df, features_all_df, tensors = collect_full_sample_artifacts(
            include_raw=args.include_raw,
            label_kind=args.label_kind,
            ignore_quality_flag=args.ignore_quality_flag,
            structured_root=structured_root,
            raw_root=raw_root,
            selected_vehicle_ids=selected_vehicle_ids,
        )
        split_summary = summarize_sample_selection(samples_master_df)
        selection_summary = summarize_vehicle_selection(samples_master_df)
        validation_report = validate_full_split_consistency(
            samples_master_df,
            load_split_frame(selected_vehicle_ids=selected_vehicle_ids),
            raise_on_blocking=False,
        )
        write_full_reports(output_root, split_summary, selection_summary, validation_report, args)
        if validation_report['blocking_errors']:
            raise ValueError('Full split consistency check failed: ' + ', '.join(validation_report['blocking_errors']))

    paths = save_sample_artifacts(output_root, samples_master_df, features_all_df, tensors)
    print(f"Saved samples to: {paths['samples_master']}")
    print(f"Saved features to: {paths['features_all']}")
    print(f"Saved tensor pack to: {paths['dataset_pack']}")
    print(f'Sample count: {len(samples_master_df)}')


if __name__ == '__main__':
    main()
