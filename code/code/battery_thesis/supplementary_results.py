from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


CORE_FAULTS = ['sd', 'samp', 'ins']

MODEL_DISPLAY_NAMES = {
    'lightgbm': 'LightGBM',
    'lstm': 'LSTM',
    'transformer': 'Vanilla Transformer',
    'shared_encoder_expert_heads': 'Proposed Method',
    'threshold_trend': 'Threshold Trend',
    'shared_encoder_no_fault_specific_features': 'No Fault-Specific Features',
    'shared_encoder_no_expert_heads': 'No Expert Heads',
    'shared_encoder_no_warning_task': 'No Warning Task',
    'shared_encoder_no_label_quality_control': 'No Label Quality Control',
}


@dataclass
class SelectedCase:
    case_name: str
    fault_type: str
    case_type: str
    sample_id: str
    vehicle_id: str
    title: str
    plot_frame: pd.DataFrame
    summary_row: dict[str, object]


def normalize_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name))


def compute_multifault_summary_from_predictions(
    prediction_frames: dict[str, pd.DataFrame],
    core_faults: list[str] | None = None,
) -> pd.DataFrame:
    core_faults = list(core_faults or CORE_FAULTS)
    rows: list[dict[str, object]] = []
    for model_key, prediction_df in prediction_frames.items():
        for scope_name, fault_list in [('all_faults', None), ('core_faults', core_faults)]:
            metrics = _compute_macro_micro_metrics(prediction_df, fault_list=fault_list)
            rows.append(
                {
                    'model_key': model_key,
                    'model_name': normalize_model_name(model_key),
                    'fault_scope': scope_name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_core_metrics(
    summary_df: pd.DataFrame,
    metric_columns: list[str],
    core_faults: list[str] | None = None,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    core_faults = list(core_faults or CORE_FAULTS)
    frame = summary_df.copy()
    if 'fault_type' in frame.columns:
        frame = frame[frame['fault_type'].astype(str).isin(core_faults)].copy()
    frame = frame[frame['split'].astype(str) == 'test'].copy()
    if frame.empty:
        return pd.DataFrame(columns=['model_key', 'model_name', 'fault_count', *metric_columns])

    frame['model_key'] = frame['model_name'].astype(str)
    grouped = frame.groupby('model_key', dropna=False)
    summary_rows: list[dict[str, object]] = []
    for model_key, model_frame in grouped:
        row: dict[str, object] = {
            'model_key': model_key,
            'model_name': normalize_model_name(model_key),
            'fault_count': int(model_frame['fault_type'].nunique()) if 'fault_type' in model_frame.columns else int(len(model_frame)),
        }
        for metric_name in metric_columns:
            if metric_name not in model_frame.columns:
                row[metric_name] = np.nan
                continue
            metric_values = pd.to_numeric(model_frame[metric_name], errors='coerce')
            row[metric_name] = float(metric_values.mean()) if metric_values.notna().any() else np.nan
        summary_rows.append(row)

    result = pd.DataFrame(summary_rows)
    if model_order:
        order_map = {name: index for index, name in enumerate(model_order)}
        result['sort_key'] = result['model_key'].map(lambda value: order_map.get(value, len(order_map)))
        result = result.sort_values(['sort_key', 'model_name']).drop(columns=['sort_key']).reset_index(drop=True)
    else:
        result = result.sort_values('model_name').reset_index(drop=True)
    return result


def build_metric_bar_frame(
    comparison_df: pd.DataFrame,
    category_column: str,
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in comparison_df.iterrows():
        category = row[category_column]
        for metric_name in metric_columns:
            if metric_name not in row or pd.isna(row[metric_name]):
                continue
            rows.append(
                {
                    'plot_type': 'bar_metric',
                    'series': metric_name,
                    'category': category,
                    'value': float(row[metric_name]),
                }
            )
    return pd.DataFrame(rows)


def select_case_examples(
    samples_master: pd.DataFrame,
    predictions_df: pd.DataFrame,
    success_fault: str = 'sd',
    failure_fault: str = 'samp',
    context_radius: int = 15,
) -> list[SelectedCase]:
    merged = predictions_df.merge(
        samples_master[['sample_id', 'vehicle_id', 'end_time', 'future_first_fault', 'lead_time_sec']],
        on=['sample_id', 'vehicle_id'],
        how='left',
    )
    merged['margin'] = pd.to_numeric(merged['y_score'], errors='coerce').fillna(0.0) - pd.to_numeric(
        merged['threshold'], errors='coerce'
    ).fillna(0.0)

    cases = []
    success_case = _select_single_case(
        merged,
        fault_type=success_fault,
        case_type='true_positive',
        filter_mask=lambda frame: (frame['y_true'] == 1) & (frame['y_pred'] == 1),
        sort_columns=['margin', 'y_score'],
        ascending=[False, False],
        context_radius=context_radius,
        title_prefix='正确识别案例',
    )
    if success_case is not None:
        cases.append(success_case)

    failure_case = _select_single_case(
        merged,
        fault_type=failure_fault,
        case_type='false_negative',
        filter_mask=lambda frame: (frame['y_true'] == 1) & (frame['y_pred'] == 0),
        sort_columns=['margin', 'y_score'],
        ascending=[True, False],
        context_radius=context_radius,
        title_prefix='漏识别案例',
    )
    if failure_case is not None:
        cases.append(failure_case)
    return cases


def case_rows_to_frame(cases: list[SelectedCase]) -> pd.DataFrame:
    rows = []
    for case in cases:
        row = dict(case.summary_row)
        row['title'] = case.title
        rows.append(row)
    return pd.DataFrame(rows)


def load_recognition_prediction_frames(results_root: Path) -> dict[str, pd.DataFrame]:
    return _load_prediction_frames(results_root, task_name='recognition', model_dirs=['lightgbm', 'lstm', 'transformer', 'main_dual_task'])


def load_warning_prediction_frames(results_root: Path) -> dict[str, pd.DataFrame]:
    return _load_prediction_frames(
        results_root,
        task_name='warning',
        model_dirs=['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'main_dual_task'],
    )


def _load_prediction_frames(results_root: Path, task_name: str, model_dirs: list[str]) -> dict[str, pd.DataFrame]:
    result_frames: dict[str, pd.DataFrame] = {}
    for model_dir in model_dirs:
        path = Path(results_root) / task_name / model_dir / 'predictions.csv'
        frame = pd.read_csv(path)
        model_key = 'shared_encoder_expert_heads' if model_dir == 'main_dual_task' else model_dir
        result_frames[model_key] = frame
    return result_frames


def _compute_macro_micro_metrics(predictions_df: pd.DataFrame, fault_list: list[str] | None) -> dict[str, float]:
    frame = predictions_df.copy()
    if fault_list is not None:
        frame = frame[frame['fault_type'].isin(fault_list)].copy()
    if frame.empty:
        return {'macro_f1': 0.0, 'micro_f1': 0.0, 'sample_count': 0, 'fault_count': 0}

    y_true = (
        frame.pivot(index='sample_id', columns='fault_type', values='y_true')
        .sort_index(axis=1)
        .fillna(0)
        .astype(int)
    )
    y_pred = (
        frame.pivot(index='sample_id', columns='fault_type', values='y_pred')
        .reindex(index=y_true.index, columns=y_true.columns)
        .fillna(0)
        .astype(int)
    )
    return {
        'macro_f1': float(f1_score(y_true.to_numpy(), y_pred.to_numpy(), average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true.to_numpy(), y_pred.to_numpy(), average='micro', zero_division=0)),
        'sample_count': int(y_true.shape[0]),
        'fault_count': int(y_true.shape[1]),
    }


def _choose_case_candidate(
    fault_df: pd.DataFrame,
    candidates: pd.DataFrame,
    case_type: str,
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.Series:
    if case_type != 'true_positive':
        return candidates.sort_values(sort_columns, ascending=ascending).iloc[0]

    tp_mask = (fault_df['y_true'].astype(int) == 1) & (fault_df['y_pred'].astype(int) == 1)
    vehicle_change = fault_df['vehicle_id'].ne(fault_df['vehicle_id'].shift())
    state_change = tp_mask.ne(tp_mask.shift(fill_value=False))
    run_id = (vehicle_change | state_change).cumsum()
    row_position = pd.Series(np.arange(len(fault_df)), index=fault_df.index)
    run_stats = (
        pd.DataFrame({'run_id': run_id, 'row_position': row_position})[tp_mask]
        .groupby('run_id', dropna=False)['row_position']
        .agg(run_length='size', run_center='mean')
    )
    ranked = candidates.copy()
    ranked['_run_id'] = run_id.loc[ranked.index].to_numpy()
    ranked['_row_position'] = row_position.loc[ranked.index].to_numpy()
    ranked = ranked.join(run_stats, on='_run_id')
    if ranked['run_length'].isna().all():
        return candidates.sort_values(sort_columns, ascending=ascending).iloc[0]
    ranked['run_length'] = ranked['run_length'].fillna(0)
    ranked['run_center_distance'] = (ranked['_row_position'] - ranked['run_center']).abs().fillna(len(fault_df))
    ranked = ranked.sort_values(
        ['run_length', 'run_center_distance', 'margin', 'y_score'],
        ascending=[False, True, False, False],
    )
    return ranked.drop(columns=['_run_id', '_row_position', 'run_length', 'run_center', 'run_center_distance']).iloc[0]

def _select_single_case(
    merged_predictions: pd.DataFrame,
    fault_type: str,
    case_type: str,
    filter_mask,
    sort_columns: list[str],
    ascending: list[bool],
    context_radius: int,
    title_prefix: str,
) -> SelectedCase | None:
    fault_df = merged_predictions[merged_predictions['fault_type'] == fault_type].copy()
    fault_df['end_time'] = pd.to_numeric(fault_df['end_time'], errors='coerce')
    fault_df = fault_df.dropna(subset=['end_time']).sort_values(['vehicle_id', 'end_time', 'sample_id']).reset_index(drop=True)
    if fault_df.empty:
        return None

    candidates = fault_df[filter_mask(fault_df)].copy()
    if candidates.empty:
        return None
    chosen = _choose_case_candidate(fault_df, candidates, case_type, sort_columns, ascending)

    vehicle_df = fault_df[fault_df['vehicle_id'] == chosen['vehicle_id']].reset_index(drop=True)
    center_index = int(vehicle_df.index[vehicle_df['sample_id'] == chosen['sample_id']][0])
    if case_type == 'true_positive':
        tp_mask = (vehicle_df['y_true'].astype(int) == 1) & (vehicle_df['y_pred'].astype(int) == 1)
        run_start = center_index
        while run_start > 0 and bool(tp_mask.iloc[run_start - 1]):
            run_start -= 1
        run_end = center_index
        while run_end + 1 < len(vehicle_df) and bool(tp_mask.iloc[run_end + 1]):
            run_end += 1
        start = max(run_start - min(3, context_radius), 0)
        end = run_end + 1
    else:
        start = max(center_index - context_radius, 0)
        end = min(center_index + context_radius + 1, len(vehicle_df))
    context_df = vehicle_df.iloc[start:end].copy().reset_index(drop=True)
    selected_relative_index = center_index - start
    context_df['relative_index'] = np.arange(len(context_df)) - selected_relative_index
    plot_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    'x': context_df['relative_index'],
                    'x_abs': context_df['end_time'],
                    'y': context_df['y_score'],
                    'series': 'score',
                }
            ),
            pd.DataFrame(
                {
                    'x': context_df['relative_index'],
                    'x_abs': context_df['end_time'],
                    'y': context_df['threshold'],
                    'series': 'threshold',
                }
            ),
            pd.DataFrame(
                {
                    'x': context_df['relative_index'],
                    'x_abs': context_df['end_time'],
                    'y': context_df['y_true'],
                    'series': 'y_true',
                }
            ),
            pd.DataFrame({'x': [0], 'x_abs': [chosen['end_time']], 'y': [0.0], 'series': 'selected_window'}),
        ],
        ignore_index=True,
    )
    case_name = f'{fault_type}_{case_type}'
    title = f'{title_prefix}: {fault_type} / {chosen["vehicle_id"]}'
    summary_row = {
        'case_name': case_name,
        'fault_type': fault_type,
        'case_type': case_type,
        'sample_id': str(chosen['sample_id']),
        'vehicle_id': str(chosen['vehicle_id']),
        'y_true': int(chosen['y_true']),
        'y_pred': int(chosen['y_pred']),
        'y_score': float(chosen['y_score']),
        'threshold': float(chosen['threshold']),
        'margin': float(chosen['margin']),
        'end_time': int(chosen['end_time']),
        'window_size': int(len(context_df)),
        'future_first_fault': chosen.get('future_first_fault'),
        'lead_time_sec': chosen.get('lead_time_sec'),
    }
    return SelectedCase(
        case_name=case_name,
        fault_type=fault_type,
        case_type=case_type,
        sample_id=str(chosen['sample_id']),
        vehicle_id=str(chosen['vehicle_id']),
        title=title,
        plot_frame=plot_frame,
        summary_row=summary_row,
    )
