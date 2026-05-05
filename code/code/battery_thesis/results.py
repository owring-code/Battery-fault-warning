from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [
    'experiment_name',
    'model_name',
    'task_type',
    'fault_type',
    'split',
    'accuracy',
    'f1',
    'recall',
    'precision',
    'pr_auc',
    'roc_auc',
    'warning_f1',
    'warning_recall',
    'false_alarm_rate',
    'avg_lead_time',
    'macro_f1',
    'micro_f1',
]

CORE_FAULTS = ['sd', 'samp', 'ins']
RARE_FAULTS = ['isc', 'conn']
CORE_RECOGNITION_MODELS = ['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'main_dual_task']
CORE_WARNING_MODELS = ['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'main_dual_task']
DEFAULT_DEFERRED_ITEMS = ['RAW_DATA 外部验证（实验 6）已延期，不纳入本轮 Data_set 正式结论。']


def build_summary_row(
    experiment_name: str,
    model_name: str,
    task_type: str,
    fault_type: str,
    split: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    row = {column: None for column in RESULT_COLUMNS}
    row.update(
        {
            'experiment_name': experiment_name,
            'model_name': model_name,
            'task_type': task_type,
            'fault_type': fault_type,
            'split': split,
        }
    )
    row.update(metrics)
    return row


def write_result_bundle(
    output_dir: Path,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    data_points: pd.DataFrame,
    notes: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'summary': output_dir / 'summary.csv',
        'predictions': output_dir / 'predictions.csv',
        'data_points': output_dir / 'data_points.csv',
        'notes': output_dir / 'notes.md',
    }
    summary.to_csv(paths['summary'], index=False, encoding='utf-8-sig')
    predictions.to_csv(paths['predictions'], index=False, encoding='utf-8-sig')
    data_points.to_csv(paths['data_points'], index=False, encoding='utf-8-sig')
    paths['notes'].write_text(notes or '', encoding='utf-8-sig')
    return paths


def aggregate_round1_tables(results_root: Path, acceptance: dict[str, object] | None = None) -> dict[str, Path]:
    results_root = Path(results_root)
    tables_root = results_root / 'tables'
    tables_root.mkdir(parents=True, exist_ok=True)

    outputs = {
        'recognition': tables_root / 'recognition_core_summary.csv',
        'warning': tables_root / 'warning_core_summary.csv',
        'ablation': tables_root / 'ablation_summary.csv',
        'rare_recognition': tables_root / 'recognition_rare_faults_summary.csv',
        'acceptance_report': tables_root / 'round1_acceptance_report.md',
    }

    recognition_df = _collect_named_summaries(results_root / 'recognition', CORE_RECOGNITION_MODELS)
    warning_df = _collect_named_summaries(results_root / 'warning', CORE_WARNING_MODELS)
    ablation_df = _collect_all_summaries(results_root / 'ablation')

    recognition_core_df = _filter_faults(recognition_df, CORE_FAULTS)
    warning_core_df = _filter_faults(warning_df, CORE_FAULTS)
    ablation_core_df = _filter_faults(ablation_df, CORE_FAULTS)
    recognition_rare_df = _filter_faults(recognition_df, RARE_FAULTS)

    recognition_core_df.to_csv(outputs['recognition'], index=False, encoding='utf-8-sig')
    warning_core_df.to_csv(outputs['warning'], index=False, encoding='utf-8-sig')
    ablation_core_df.to_csv(outputs['ablation'], index=False, encoding='utf-8-sig')
    recognition_rare_df.to_csv(outputs['rare_recognition'], index=False, encoding='utf-8-sig')

    resolved_acceptance = acceptance or _derive_acceptance(
        recognition_core_df,
        warning_core_df,
        ablation_core_df,
        recognition_rare_df,
    )
    outputs['acceptance_report'].write_text(
        _build_acceptance_report(resolved_acceptance),
        encoding='utf-8-sig',
    )
    return outputs


def _derive_acceptance(
    recognition_core_df: pd.DataFrame,
    warning_core_df: pd.DataFrame,
    ablation_core_df: pd.DataFrame,
    recognition_rare_df: pd.DataFrame,
) -> dict[str, object]:
    core_passed = not recognition_core_df.empty and not warning_core_df.empty and not ablation_core_df.empty
    rare_fault_notes: list[str] = []
    if recognition_rare_df.empty:
        rare_fault_notes.append('isc / conn 未产出可用识别结果，当前仅保留为稀缺故障探索项。')
    else:
        available_faults = sorted(recognition_rare_df['fault_type'].dropna().astype(str).unique().tolist())
        rare_fault_notes.append('稀缺故障已保留在补充结果中：' + ', '.join(available_faults))
    data_limitations = ['当前数据集中 conn、isc 的窗口级正例车辆极少，相关结论仅作探索性参考。']
    return {
        'core_passed': core_passed,
        'rare_fault_notes': rare_fault_notes,
        'data_limitations': data_limitations,
        'deferred_items': DEFAULT_DEFERRED_ITEMS,
        'can_enter_round2': core_passed,
    }


def _build_acceptance_report(acceptance: dict[str, object]) -> str:
    core_passed = bool(acceptance.get('core_passed', False))
    can_enter_round2 = bool(acceptance.get('can_enter_round2', core_passed))
    rare_fault_notes = [str(note) for note in acceptance.get('rare_fault_notes', [])]
    data_limitations = [str(note) for note in acceptance.get('data_limitations', [])]
    deferred_items = [str(note) for note in acceptance.get('deferred_items', [])]

    lines = [
        '# 第1轮验收报告',
        '',
        f'- 核心任务状态：{"通过" if core_passed else "未通过"}',
        f'- 是否可以进入第2轮：{"可以" if can_enter_round2 else "暂不可以"}',
        '',
        '## 核心任务说明',
        '',
        '- 第1轮正式定位为核心故障识别与预警可行性验证。',
        '- 核心正式任务为 sd / samp / ins。',
        '- 稀缺探索任务为 isc / conn。',
        '',
        '## 稀缺故障说明',
        '',
    ]
    if rare_fault_notes:
        lines.extend(f'- {note}' for note in rare_fault_notes)
    else:
        lines.append('- 当前没有额外的稀缺故障说明。')

    lines.extend([
        '',
        '## 数据限制说明',
        '',
    ])
    if data_limitations:
        lines.extend(f'- {note}' for note in data_limitations)
    else:
        lines.append('- 当前没有额外的数据限制说明。')

    lines.extend([
        '',
        '## 延期事项',
        '',
    ])
    if deferred_items:
        lines.extend(f'- {note}' for note in deferred_items)
    else:
        lines.append('- 当前没有延期事项。')

    return '\n'.join(lines) + '\n'


def _filter_faults(frame: pd.DataFrame, faults: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    if 'fault_type' not in frame.columns:
        return frame.copy()
    filtered = frame[frame['fault_type'].astype(str).isin(faults)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=frame.columns)
    return filtered.reset_index(drop=True)


def _collect_named_summaries(parent: Path, subdirs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in subdirs:
        summary_path = parent / name / 'summary.csv'
        if summary_path.exists():
            frames.append(pd.read_csv(summary_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)


def _collect_all_summaries(parent: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if parent.exists():
        for child in sorted(parent.iterdir()):
            summary_path = child / 'summary.csv'
            if child.is_dir() and summary_path.exists():
                frames.append(pd.read_csv(summary_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
