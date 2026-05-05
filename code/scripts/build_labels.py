from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import FINAL_LABELS_ROOT, RAW_LABELS_ROOT, STRUCTURED_LABELS_ROOT, ensure_project_directories
from battery_thesis.label_pipeline import (
    EMPTY_EVENT_COLUMNS,
    build_raw_label_artifacts,
    build_structured_label_artifacts,
    partition_summary_by_dataset,
    save_label_bundle,
)
from battery_thesis.metadata import scan_project_datasets

DEFAULT_STRUCTURED_ROOT = Path(r'F:\Data_set')
DEFAULT_RAW_ROOT = Path(r'F:\RAW_DATA\data')


def _build_summary_row(
    vehicle_id: str,
    source_dataset: str,
    source_file: str,
    label_file: Path,
    event_file: Path,
    frame_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        'vehicle_id': vehicle_id,
        'source_dataset': source_dataset,
        'source_file': source_file,
        'label_file': str(label_file),
        'event_file': str(event_file),
        'total_rows': len(frame_df),
        'sd_positive': int(frame_df['label_final_sd'].sum()),
        'isc_positive': int(frame_df['label_final_isc'].sum()),
        'conn_positive': int(frame_df['label_final_conn'].sum()),
        'samp_positive': int(frame_df['label_final_samp'].sum()),
        'ins_positive': int(frame_df['label_final_ins'].sum()),
    }


def _write_label_statistics(scoped_summaries: dict[str, pd.DataFrame]) -> None:
    stats_rows = []
    for dataset_name, df in scoped_summaries.items():
        if df.empty:
            continue
        stats_rows.append(
            {
                'dataset_scope': dataset_name,
                'vehicle_count': int(df['vehicle_id'].nunique()),
                'total_rows': int(df['total_rows'].sum()),
                'sd_positive': int(df['sd_positive'].sum()),
                'isc_positive': int(df['isc_positive'].sum()),
                'conn_positive': int(df['conn_positive'].sum()),
                'samp_positive': int(df['samp_positive'].sum()),
                'ins_positive': int(df['ins_positive'].sum()),
            }
        )
    pd.DataFrame(stats_rows).to_csv(FINAL_LABELS_ROOT.parent / 'label_stats.csv', index=False, encoding='utf-8-sig')


def _write_label_cases() -> None:
    case_rows = []
    for event_path in sorted((FINAL_LABELS_ROOT / 'events').glob('*_events.csv')):
        events_df = pd.read_csv(event_path)
        if events_df.empty:
            continue
        first_cases = events_df.sort_values(['fault_type', 'start_frame']).groupby('fault_type', as_index=False).first()
        for _, row in first_cases.iterrows():
            case_rows.append(
                {
                    'vehicle_id': row['vehicle_id'],
                    'fault_type': row['fault_type'],
                    'event_id': row['event_id'],
                    'start_frame': row['start_frame'],
                    'end_frame': row['end_frame'],
                    'duration_seconds': row['duration_seconds'],
                    'event_file': str(event_path),
                }
            )
    pd.DataFrame(case_rows).to_csv(FINAL_LABELS_ROOT.parent / 'label_cases.csv', index=False, encoding='utf-8-sig')


def _write_quality_notes() -> None:
    notes = '\n'.join(
        [
            '# 标签质量说明',
            '',
            '- structured_dataset 当前对 5 类故障统一使用 identify_code 风格规则进行重构，其中 sd / samp / ins 为核心正式任务，isc / conn 仍按稀缺故障探索项处理。',
            '- structured_dataset 标签在规则输出后仍进行 2 帧间隙合并，用于缓解单点抖动。',
            '- raw_dataset 当前使用同一套重构规则生成正式标签；若原始文件缺失 insulationresistance，则 ins 标签保留为 0 并在 label_source 中显式标记。',
            '- raw_dataset 的 quality_flag 依据原始电压/温度序列完整性生成，不再统一置为 low。',
            '- 最终训练标签读取 artifacts/labels/final/per_vehicle。',
            '- 识别标签采用窗口末尾 3 帧多数表决；预警标签采用未来 60 秒首次进入故障事件。',
        ]
    )
    (FINAL_LABELS_ROOT.parent / 'label_quality_notes.md').write_text(notes, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build final per-vehicle labels for structured_dataset and raw_dataset.')
    parser.add_argument('--structured-root', default=str(DEFAULT_STRUCTURED_ROOT))
    parser.add_argument('--raw-root', default=str(DEFAULT_RAW_ROOT))
    args = parser.parse_args()

    ensure_project_directories()
    manifest = scan_project_datasets(Path(args.structured_root), Path(args.raw_root))
    summaries = []

    structured_rows = manifest[manifest['source_dataset'] == 'structured_dataset']
    for _, row in structured_rows.iterrows():
        source_path = Path(row['source_file'])
        frame_df, events_df = build_structured_label_artifacts(source_path)
        save_label_bundle(
            frame_df,
            events_df,
            STRUCTURED_LABELS_ROOT / 'per_vehicle',
            STRUCTURED_LABELS_ROOT / 'events',
            row['vehicle_id'],
        )
        final_label_path, final_event_path = save_label_bundle(
            frame_df,
            events_df,
            FINAL_LABELS_ROOT / 'per_vehicle',
            FINAL_LABELS_ROOT / 'events',
            row['vehicle_id'],
        )
        summaries.append(
            _build_summary_row(
                vehicle_id=row['vehicle_id'],
                source_dataset='structured_dataset',
                source_file=str(source_path),
                label_file=final_label_path,
                event_file=final_event_path,
                frame_df=frame_df,
            )
        )

    raw_rows = manifest[(manifest['source_dataset'] == 'raw_dataset') & (manifest['is_extracted'] == True)]
    for _, row in raw_rows.iterrows():
        source_path = Path(row['source_file'])
        frame_df, events_df = build_raw_label_artifacts(source_path, row['vehicle_id'])
        save_label_bundle(
            frame_df,
            events_df,
            RAW_LABELS_ROOT / 'per_vehicle',
            RAW_LABELS_ROOT / 'events',
            row['vehicle_id'],
        )
        final_label_path, final_event_path = save_label_bundle(
            frame_df,
            events_df,
            FINAL_LABELS_ROOT / 'per_vehicle',
            FINAL_LABELS_ROOT / 'events',
            row['vehicle_id'],
        )
        summaries.append(
            _build_summary_row(
                vehicle_id=row['vehicle_id'],
                source_dataset='raw_dataset',
                source_file=str(source_path),
                label_file=final_label_path,
                event_file=final_event_path,
                frame_df=frame_df,
            )
        )

    summary_df = pd.DataFrame(summaries)
    scoped_summaries = partition_summary_by_dataset(summary_df)
    scoped_summaries['structured_dataset'].to_csv(STRUCTURED_LABELS_ROOT / 'summary.csv', index=False, encoding='utf-8-sig')
    scoped_summaries['raw_dataset'].to_csv(RAW_LABELS_ROOT / 'summary.csv', index=False, encoding='utf-8-sig')
    scoped_summaries['final'].to_csv(FINAL_LABELS_ROOT / 'summary.csv', index=False, encoding='utf-8-sig')
    _write_label_statistics(scoped_summaries)
    _write_label_cases()
    _write_quality_notes()


if __name__ == '__main__':
    main()
