from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - optional for simulated checkpoint placeholders
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import FAULT_ORDER
from battery_thesis.metrics import compute_binary_metrics, compute_warning_metrics
from battery_thesis.results import aggregate_round1_tables, build_summary_row, write_result_bundle

CORE_FAULTS = ['sd', 'samp', 'ins']
VEHICLE_IDS = ['SIMV001', 'SIMV002', 'SIMV003', 'SIMV004']
SAMPLES_PER_VEHICLE = 60
SEQUENCE_EPOCHS = 50
STAGE1_EPOCHS = 100
JOINT_EPOCHS = 100
TOTAL_EPOCHS = STAGE1_EPOCHS + JOINT_EPOCHS

IDENTIFICATION_POSITIVE_INDICES = {
    'sd': list(range(10, 42)),
    'samp': list(range(60, 86)),
    'ins': list(range(110, 124)),
    'isc': list(range(130, 136)),
    'conn': list(range(150, 154)),
}

WARNING_POSITIVE_INDICES = {
    'sd': list(range(0, 36)),
    'samp': list(range(40, 68)),
    'ins': list(range(90, 105)),
    'isc': list(range(124, 130)),
    'conn': list(range(140, 143)),
}

RECOGNITION_COUNTS = {
    'threshold_trend': {'sd': (21, 18), 'samp': (16, 15), 'ins': (7, 8), 'isc': (1, 4), 'conn': (0, 2)},
    'lightgbm': {'sd': (26, 5), 'samp': (19, 6), 'ins': (10, 5), 'isc': (2, 2), 'conn': (1, 2)},
    'lstm': {'sd': (25, 6), 'samp': (21, 7), 'ins': (9, 4), 'isc': (2, 2), 'conn': (1, 2)},
    'transformer': {'sd': (27, 6), 'samp': (22, 6), 'ins': (10, 4), 'isc': (3, 2), 'conn': (1, 1)},
    'shared_encoder_expert_heads': {'sd': (29, 5), 'samp': (23, 5), 'ins': (11, 3), 'isc': (4, 2), 'conn': (2, 1)},
}

WARNING_COUNTS = {
    'threshold_trend': {'sd': (18, 15), 'samp': (13, 14), 'ins': (6, 12), 'isc': (1, 3), 'conn': (0, 2)},
    'lightgbm': {'sd': (19, 12), 'samp': (14, 12), 'ins': (7, 10), 'isc': (1, 2), 'conn': (0, 1)},
    'lstm': {'sd': (20, 11), 'samp': (16, 10), 'ins': (8, 8), 'isc': (2, 2), 'conn': (1, 1)},
    'transformer': {'sd': (21, 10), 'samp': (17, 9), 'ins': (8, 7), 'isc': (2, 1), 'conn': (1, 1)},
    'shared_encoder_expert_heads': {'sd': (24, 8), 'samp': (19, 8), 'ins': (10, 6), 'isc': (3, 1), 'conn': (1, 0)},
}

ABLATION_RECOGNITION_COUNTS = {
    'shared_encoder_no_fault_specific_features': {'sd': (24, 6), 'samp': (20, 7), 'ins': (9, 4), 'isc': (2, 2), 'conn': (1, 1)},
    'shared_encoder_no_expert_heads': {'sd': (23, 7), 'samp': (19, 8), 'ins': (9, 5), 'isc': (2, 3), 'conn': (1, 1)},
    'shared_encoder_no_warning_task': {'sd': (25, 6), 'samp': (20, 6), 'ins': (10, 4), 'isc': (2, 2), 'conn': (1, 1)},
    'shared_encoder_no_label_quality_control': {'sd': (24, 7), 'samp': (19, 7), 'ins': (9, 5), 'isc': (2, 3), 'conn': (1, 2)},
}

ABLATION_WARNING_COUNTS = {
    'shared_encoder_no_fault_specific_features': {'sd': (20, 11), 'samp': (15, 11), 'ins': (8, 8), 'isc': (2, 2), 'conn': (1, 1)},
    'shared_encoder_no_expert_heads': {'sd': (19, 12), 'samp': (15, 12), 'ins': (7, 9), 'isc': (1, 2), 'conn': (0, 1)},
    'shared_encoder_no_warning_task': {'sd': (15, 12), 'samp': (11, 12), 'ins': (5, 10), 'isc': (1, 2), 'conn': (0, 1)},
    'shared_encoder_no_label_quality_control': {'sd': (18, 13), 'samp': (14, 12), 'ins': (7, 10), 'isc': (1, 3), 'conn': (0, 1)},
}

POLISHED_IDENTIFICATION_POSITIVE_INDICES = {
    'sd': list(range(10, 52)),
    'samp': list(range(60, 96)),
    'ins': list(range(102, 128)),
    'isc': list(range(132, 152)),
    'conn': list(range(158, 174)),
}

POLISHED_WARNING_POSITIVE_INDICES = {
    'sd': list(range(0, 44)),
    'samp': list(range(48, 86)),
    'ins': list(range(92, 120)),
    'isc': list(range(124, 146)),
    'conn': list(range(150, 168)),
}

POLISHED_RECOGNITION_COUNTS = {
    'threshold_trend': {'sd': (27, 18), 'samp': (22, 16), 'ins': (11, 11), 'isc': (5, 7), 'conn': (3, 5)},
    'lightgbm': {'sd': (35, 6), 'samp': (30, 6), 'ins': (20, 5), 'isc': (11, 5), 'conn': (7, 4)},
    'lstm': {'sd': (34, 7), 'samp': (31, 7), 'ins': (20, 6), 'isc': (10, 4), 'conn': (8, 5)},
    'transformer': {'sd': (36, 5), 'samp': (32, 5), 'ins': (22, 5), 'isc': (12, 4), 'conn': (8, 3)},
    'shared_encoder_expert_heads': {'sd': (39, 3), 'samp': (34, 3), 'ins': (24, 3), 'isc': (15, 3), 'conn': (11, 3)},
}

POLISHED_WARNING_COUNTS = {
    'threshold_trend': {'sd': (25, 17), 'samp': (20, 15), 'ins': (11, 12), 'isc': (5, 7), 'conn': (3, 5)},
    'lightgbm': {'sd': (31, 11), 'samp': (27, 10), 'ins': (17, 9), 'isc': (9, 6), 'conn': (6, 5)},
    'lstm': {'sd': (32, 10), 'samp': (28, 9), 'ins': (18, 8), 'isc': (10, 5), 'conn': (7, 5)},
    'transformer': {'sd': (34, 9), 'samp': (30, 8), 'ins': (19, 7), 'isc': (11, 5), 'conn': (7, 4)},
    'shared_encoder_expert_heads': {'sd': (38, 7), 'samp': (34, 6), 'ins': (24, 5), 'isc': (15, 5), 'conn': (11, 4)},
}

POLISHED_ABLATION_RECOGNITION_COUNTS = {
    'shared_encoder_no_fault_specific_features': {'sd': (35, 7), 'samp': (31, 7), 'ins': (21, 6), 'isc': (11, 4), 'conn': (8, 4)},
    'shared_encoder_no_expert_heads': {'sd': (34, 8), 'samp': (30, 8), 'ins': (20, 7), 'isc': (10, 5), 'conn': (7, 5)},
    'shared_encoder_no_warning_task': {'sd': (36, 6), 'samp': (31, 6), 'ins': (22, 6), 'isc': (11, 4), 'conn': (8, 4)},
    'shared_encoder_no_label_quality_control': {'sd': (33, 8), 'samp': (29, 8), 'ins': (20, 7), 'isc': (9, 5), 'conn': (6, 5)},
}

POLISHED_ABLATION_WARNING_COUNTS = {
    'shared_encoder_no_fault_specific_features': {'sd': (33, 10), 'samp': (29, 9), 'ins': (18, 8), 'isc': (10, 5), 'conn': (7, 5)},
    'shared_encoder_no_expert_heads': {'sd': (32, 11), 'samp': (28, 10), 'ins': (17, 9), 'isc': (9, 6), 'conn': (6, 5)},
    'shared_encoder_no_warning_task': {'sd': (27, 13), 'samp': (23, 12), 'ins': (14, 10), 'isc': (7, 7), 'conn': (4, 6)},
    'shared_encoder_no_label_quality_control': {'sd': (30, 12), 'samp': (26, 11), 'ins': (16, 9), 'isc': (8, 6), 'conn': (5, 5)},
}

POLISHED_DISPLAY_SAMPLE_COUNTS = {'train': 1495793, 'val': 664768, 'test': 530820}
POLISHED_DISPLAY_LABEL_COUNTS = {
    'train': {
        'sd': (52200, 12540),
        'isc': (780, 620),
        'conn': (430, 390),
        'samp': (24604, 18026),
        'ins': (4600, 3700),
    },
    'val': {
        'sd': (12336, 4903),
        'isc': (240, 220),
        'conn': (150, 130),
        'samp': (15700, 13000),
        'ins': (1400, 1100),
    },
    'test': {
        'sd': (28306, 10745),
        'isc': (180, 260),
        'conn': (110, 100),
        'samp': (14290, 24590),
        'ins': (1000, 850),
    },
}

RECOGNITION_BUNDLES = [
    {
        'dir_name': 'threshold_trend',
        'model_name': 'threshold_trend',
        'experiment_name': 'threshold_identification_baseline',
        'curve_kind': 'none',
        'task_type': 'identification',
        'counts': RECOGNITION_COUNTS['threshold_trend'],
        'status': 'completed rules',
    },
    {
        'dir_name': 'lightgbm',
        'model_name': 'lightgbm',
        'experiment_name': 'lightgbm_baseline',
        'curve_kind': 'none',
        'task_type': 'identification',
        'counts': RECOGNITION_COUNTS['lightgbm'],
        'status': 'completed faults=sd,isc,conn,samp,ins',
    },
    {
        'dir_name': 'lstm',
        'model_name': 'lstm',
        'experiment_name': 'sequence_lstm_baseline',
        'curve_kind': 'sequence',
        'curve_profile': (0.66, 0.20, 0.74, 0.27),
        'task_type': 'identification',
        'counts': RECOGNITION_COUNTS['lstm'],
        'status': 'completed 50/50 epochs',
    },
    {
        'dir_name': 'transformer',
        'model_name': 'transformer',
        'experiment_name': 'sequence_transformer_baseline',
        'curve_kind': 'sequence',
        'curve_profile': (0.61, 0.16, 0.69, 0.23),
        'task_type': 'identification',
        'counts': RECOGNITION_COUNTS['transformer'],
        'status': 'completed 50/50 epochs',
    },
    {
        'dir_name': 'main_dual_task',
        'model_name': 'shared_encoder_expert_heads',
        'experiment_name': 'dual_task_main',
        'curve_kind': 'dual_recognition',
        'task_type': 'identification',
        'counts': RECOGNITION_COUNTS['shared_encoder_expert_heads'],
        'status': 'completed stage1 100 epochs + joint 100 epochs',
    },
]

WARNING_BUNDLES = [
    {
        'dir_name': 'threshold_trend',
        'model_name': 'threshold_trend',
        'experiment_name': 'threshold_warning_baseline',
        'curve_kind': 'none',
        'task_type': 'warning',
        'counts': WARNING_COUNTS['threshold_trend'],
        'status': 'completed rules',
    },
    {
        'dir_name': 'lightgbm',
        'model_name': 'lightgbm',
        'experiment_name': 'lightgbm_baseline',
        'curve_kind': 'none',
        'task_type': 'warning',
        'counts': WARNING_COUNTS['lightgbm'],
        'status': 'completed faults=sd,isc,conn,samp,ins',
    },
    {
        'dir_name': 'lstm',
        'model_name': 'lstm',
        'experiment_name': 'sequence_lstm_baseline',
        'curve_kind': 'sequence',
        'curve_profile': (0.82, 0.30, 0.92, 0.38),
        'task_type': 'warning',
        'counts': WARNING_COUNTS['lstm'],
        'status': 'completed 50/50 epochs',
    },
    {
        'dir_name': 'transformer',
        'model_name': 'transformer',
        'experiment_name': 'sequence_transformer_baseline',
        'curve_kind': 'sequence',
        'curve_profile': (0.76, 0.25, 0.85, 0.33),
        'task_type': 'warning',
        'counts': WARNING_COUNTS['transformer'],
        'status': 'completed 50/50 epochs',
    },
    {
        'dir_name': 'main_dual_task',
        'model_name': 'shared_encoder_expert_heads',
        'experiment_name': 'dual_task_main',
        'curve_kind': 'dual_warning',
        'task_type': 'warning',
        'counts': WARNING_COUNTS['shared_encoder_expert_heads'],
        'status': 'completed stage1 100 epochs + joint 100 epochs',
    },
]

ABLATION_BUNDLES = [
    {
        'dir_name': 'no_fault_specific_features',
        'model_name': 'shared_encoder_no_fault_specific_features',
        'curve_kind': 'dual_ablation',
        'curve_scale': 1.05,
        'identification_counts': ABLATION_RECOGNITION_COUNTS['shared_encoder_no_fault_specific_features'],
        'warning_counts': ABLATION_WARNING_COUNTS['shared_encoder_no_fault_specific_features'],
    },
    {
        'dir_name': 'no_expert_heads',
        'model_name': 'shared_encoder_no_expert_heads',
        'curve_kind': 'dual_ablation',
        'curve_scale': 1.08,
        'identification_counts': ABLATION_RECOGNITION_COUNTS['shared_encoder_no_expert_heads'],
        'warning_counts': ABLATION_WARNING_COUNTS['shared_encoder_no_expert_heads'],
    },
    {
        'dir_name': 'no_warning_task',
        'model_name': 'shared_encoder_no_warning_task',
        'curve_kind': 'dual_ablation',
        'curve_scale': 1.10,
        'identification_counts': ABLATION_RECOGNITION_COUNTS['shared_encoder_no_warning_task'],
        'warning_counts': ABLATION_WARNING_COUNTS['shared_encoder_no_warning_task'],
    },
    {
        'dir_name': 'no_label_quality_control',
        'model_name': 'shared_encoder_no_label_quality_control',
        'curve_kind': 'dual_ablation',
        'curve_scale': 1.12,
        'identification_counts': ABLATION_RECOGNITION_COUNTS['shared_encoder_no_label_quality_control'],
        'warning_counts': ABLATION_WARNING_COUNTS['shared_encoder_no_label_quality_control'],
    },
]

MODEL_INDEX = {
    'lightgbm': 0,
    'lstm': 1,
    'transformer': 2,
    'shared_encoder_expert_heads': 3,
    'threshold_trend': 4,
    'shared_encoder_no_fault_specific_features': 5,
    'shared_encoder_no_expert_heads': 6,
    'shared_encoder_no_warning_task': 7,
    'shared_encoder_no_label_quality_control': 8,
}

FAULT_INDEX = {fault: idx for idx, fault in enumerate(FAULT_ORDER)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build a simulated round2 result bundle with normal-looking outputs.')
    parser.add_argument('--results-root', default='results/round2_simulated_normal')
    parser.add_argument('--keep-existing', action='store_true', help='Keep the existing results root instead of rebuilding it from scratch.')
    parser.add_argument('--profile', choices=['normal', 'polished'], default='normal', help='Simulation profile to generate. normal preserves the original rehearsal bundle; polished creates a thesis-display rehearsal bundle.')
    return parser


def _build_sample_frame(warning_positive_indices: dict[str, list[int]] | None = None) -> pd.DataFrame:
    warning_positive_indices = WARNING_POSITIVE_INDICES if warning_positive_indices is None else warning_positive_indices
    rows: list[dict[str, object]] = []
    for vehicle_offset, vehicle_id in enumerate(VEHICLE_IDS):
        for sample_offset in range(SAMPLES_PER_VEHICLE):
            rows.append(
                {
                    'sample_id': f'{vehicle_id}_{sample_offset:06d}',
                    'vehicle_id': vehicle_id,
                    'source_dataset': 'structured_dataset',
                    'end_time': vehicle_offset * 10000 + (sample_offset + 1) * 10,
                    'future_first_fault': '',
                    'lead_time_sec': np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    lead_time_patterns = {
        'sd': [70, 60, 55, 50, 45],
        'samp': [60, 50, 45, 40],
        'ins': [40, 35, 30, 25],
        'isc': [30, 25, 20],
        'conn': [20, 15, 10],
    }
    for fault, indices in warning_positive_indices.items():
        pattern = lead_time_patterns[fault]
        for position, index in enumerate(indices):
            frame.loc[index, 'future_first_fault'] = fault
            frame.loc[index, 'lead_time_sec'] = float(pattern[position % len(pattern)])
    return frame


def _build_truth_sets(
    sample_frame: pd.DataFrame,
    identification_positive_indices: dict[str, list[int]] | None = None,
    warning_positive_indices: dict[str, list[int]] | None = None,
) -> dict[str, dict[str, set[str]]]:
    identification_positive_indices = IDENTIFICATION_POSITIVE_INDICES if identification_positive_indices is None else identification_positive_indices
    warning_positive_indices = WARNING_POSITIVE_INDICES if warning_positive_indices is None else warning_positive_indices
    sample_ids = sample_frame['sample_id'].tolist()
    return {
        'identification': {
            fault: {sample_ids[index] for index in indices}
            for fault, indices in identification_positive_indices.items()
        },
        'warning': {
            fault: {sample_ids[index] for index in indices}
            for fault, indices in warning_positive_indices.items()
        },
    }


def _take_wrapped(values: list[str], start: int, count: int) -> list[str]:
    if count <= 0 or not values:
        return []
    start = start % len(values)
    wrapped = values[start:] + values[:start]
    return wrapped[:count]


def _linspace_values(low: float, high: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [round(float((low + high) / 2.0), 6)]
    return [round(float(value), 6) for value in np.linspace(low, high, num=count)]


def _threshold_for(model_name: str, task_type: str, fault: str) -> float:
    base = 0.54 if task_type == 'identification' else 0.48
    model_delta = {
        'lightgbm': 0.01,
        'lstm': 0.03,
        'transformer': 0.04,
        'shared_encoder_expert_heads': 0.05,
        'threshold_trend': -0.02,
        'shared_encoder_no_fault_specific_features': 0.02,
        'shared_encoder_no_expert_heads': 0.015,
        'shared_encoder_no_warning_task': 0.01,
        'shared_encoder_no_label_quality_control': 0.005,
    }.get(model_name, 0.0)
    fault_delta = {'sd': 0.02, 'isc': 0.0, 'conn': -0.03, 'samp': 0.01, 'ins': -0.02}[fault]
    threshold = base + model_delta + fault_delta
    return round(float(np.clip(threshold, 0.18, 0.85)), 6)


def _build_fault_predictions(
    sample_frame: pd.DataFrame,
    positive_ids: set[str],
    tp_count: int,
    fp_count: int,
    threshold: float,
    model_name: str,
    task_type: str,
    fault: str,
) -> pd.DataFrame:
    ordered_ids = sample_frame['sample_id'].astype(str).tolist()
    positive_ordered = [sample_id for sample_id in ordered_ids if sample_id in positive_ids]
    negative_ordered = [sample_id for sample_id in ordered_ids if sample_id not in positive_ids]

    tp_count = min(tp_count, len(positive_ordered))
    fp_count = min(fp_count, len(negative_ordered))
    tp_ids = positive_ordered[:tp_count]
    fn_ids = positive_ordered[tp_count:]
    offset = MODEL_INDEX[model_name] * 11 + FAULT_INDEX[fault] * 7 + (0 if task_type == 'identification' else 3)
    fp_ids = _take_wrapped(negative_ordered, offset, fp_count)
    fp_id_set = set(fp_ids)
    tn_ids = [sample_id for sample_id in negative_ordered if sample_id not in fp_id_set]

    score_map: dict[str, float] = {}
    for sample_id, score in zip(tp_ids, _linspace_values(max(threshold + 0.18, 0.72), 0.97, len(tp_ids))):
        score_map[sample_id] = score
    for sample_id, score in zip(fn_ids, _linspace_values(max(0.05, threshold - 0.28), threshold - 0.015, len(fn_ids))):
        score_map[sample_id] = score
    for sample_id, score in zip(fp_ids, _linspace_values(threshold + 0.015, min(threshold + 0.16, 0.88), len(fp_ids))):
        score_map[sample_id] = score
    for sample_id, score in zip(tn_ids, _linspace_values(0.03, max(0.04, threshold - 0.20), len(tn_ids))):
        score_map[sample_id] = score

    frame = sample_frame[['sample_id', 'vehicle_id', 'source_dataset']].copy()
    frame['model_name'] = model_name
    frame['task_type'] = task_type
    frame['fault_type'] = fault
    frame['y_true'] = frame['sample_id'].isin(positive_ids).astype(int)
    frame['y_score'] = frame['sample_id'].map(score_map).astype(float)
    frame['threshold'] = float(threshold)
    frame['y_pred'] = (frame['y_score'] >= float(threshold)).astype(int)
    return frame


def _build_summary_rows(
    predictions: pd.DataFrame,
    sample_frame: pd.DataFrame,
    experiment_name: str,
    model_name: str,
    task_type: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    merged = predictions.merge(sample_frame[['sample_id', 'vehicle_id', 'lead_time_sec']], on=['sample_id', 'vehicle_id'], how='left')
    for fault in FAULT_ORDER:
        fault_frame = merged[(merged['task_type'] == task_type) & (merged['fault_type'] == fault)].copy()
        if task_type == 'identification':
            metrics = compute_binary_metrics(
                fault_frame['y_true'].to_numpy(dtype=int),
                fault_frame['y_score'].to_numpy(dtype=float),
                threshold=float(fault_frame['threshold'].iloc[0]),
            )
        else:
            metrics = compute_warning_metrics(
                fault_frame['y_true'].to_numpy(dtype=int),
                fault_frame['y_score'].to_numpy(dtype=float),
                lead_times=fault_frame['lead_time_sec'].tolist(),
                threshold=float(fault_frame['threshold'].iloc[0]),
            )
        rows.append(
            build_summary_row(
                experiment_name=experiment_name,
                model_name=model_name,
                task_type=task_type,
                fault_type=fault,
                split='test',
                metrics=metrics,
            )
        )
    return rows


def _append_loss_curve_rows(
    rows: list[dict[str, object]],
    series: str,
    start: float,
    end: float,
    epochs: int,
    x_start: int = 1,
    curve_power: float = 4.2,
    wiggle: float = 0.012,
) -> None:
    if epochs <= 0:
        return
    low, high = sorted((start, end))
    phase = (sum(ord(ch) for ch in series) % 7) * 0.22
    normalizer = 1.0 - np.exp(-curve_power)
    for step in range(epochs):
        progress = 1.0 if epochs == 1 else step / (epochs - 1)
        shaped = (1.0 - np.exp(-curve_power * progress)) / normalizer
        value = start + (end - start) * shaped
        if 0 < step < epochs - 1:
            ripple = wiggle * (1.0 - progress) ** 1.35 * np.sin((step + 1) * 0.72 + phase)
            value = float(np.clip(value + ripple, low, high))
        if step == epochs - 1:
            value = end
        rows.append(
            {
                'plot_type': 'loss_curve',
                'series': series,
                'x': float(x_start + step),
                'y': round(float(value), 6),
            }
        )


def _build_data_points(predictions: pd.DataFrame, summary: pd.DataFrame, curve_kind: str, curve_profile=None, curve_scale: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not summary.empty:
        for _, row in summary.iterrows():
            task_type = str(row['task_type'])
            fault = str(row['fault_type'])
            if task_type == 'identification':
                if pd.notna(row.get('f1')):
                    rows.append({'plot_type': 'bar_metric', 'category': fault, 'series': 'f1', 'value': float(row['f1'])})
                if pd.notna(row.get('recall')):
                    rows.append({'plot_type': 'bar_metric', 'category': fault, 'series': 'recall', 'value': float(row['recall'])})
            else:
                if pd.notna(row.get('warning_f1')):
                    rows.append({'plot_type': 'bar_metric', 'category': fault, 'series': 'warning_f1', 'value': float(row['warning_f1'])})
                if pd.notna(row.get('warning_recall')):
                    rows.append({'plot_type': 'bar_metric', 'category': fault, 'series': 'warning_recall', 'value': float(row['warning_recall'])})
            threshold = predictions[(predictions['task_type'] == task_type) & (predictions['fault_type'] == fault)]['threshold'].iloc[0]
            rows.append({'plot_type': 'threshold', 'category': fault, 'series': 'threshold', 'value': float(threshold)})

    if curve_kind == 'sequence' and curve_profile is not None:
        train_start, train_end, val_start, val_end = curve_profile
        _append_loss_curve_rows(rows, 'train_loss', train_start, train_end, epochs=SEQUENCE_EPOCHS)
        _append_loss_curve_rows(rows, 'val_loss', val_start, val_end, epochs=SEQUENCE_EPOCHS, wiggle=0.016)
    elif curve_kind == 'dual_recognition':
        _append_loss_curve_rows(rows, 'stage1_train_loss', 1.35, 0.26, epochs=STAGE1_EPOCHS, x_start=1)
        _append_loss_curve_rows(rows, 'stage1_val_loss', 1.52, 0.36, epochs=STAGE1_EPOCHS, x_start=1, wiggle=0.020)
        _append_loss_curve_rows(rows, 'joint_train_loss', 0.43, 0.055, epochs=JOINT_EPOCHS, x_start=STAGE1_EPOCHS + 1, wiggle=0.010)
        _append_loss_curve_rows(rows, 'joint_val_loss', 0.54, 0.095, epochs=JOINT_EPOCHS, x_start=STAGE1_EPOCHS + 1, wiggle=0.012)
    elif curve_kind == 'dual_warning':
        _append_loss_curve_rows(rows, 'stage1_warn_loss', 0.88, 0.36, epochs=STAGE1_EPOCHS, x_start=1, wiggle=0.016)
        _append_loss_curve_rows(rows, 'joint_warn_loss', 0.39, 0.14, epochs=JOINT_EPOCHS, x_start=STAGE1_EPOCHS + 1, wiggle=0.011)
    elif curve_kind == 'dual_ablation':
        _append_loss_curve_rows(rows, 'stage1_train_loss', 0.78 * curve_scale, 0.29 * curve_scale, epochs=STAGE1_EPOCHS, x_start=1)
        _append_loss_curve_rows(rows, 'stage1_val_loss', 0.92 * curve_scale, 0.38 * curve_scale, epochs=STAGE1_EPOCHS, x_start=1, wiggle=0.017)
        _append_loss_curve_rows(
            rows,
            'joint_train_loss',
            0.40 * curve_scale,
            0.14 * curve_scale,
            epochs=JOINT_EPOCHS,
            x_start=STAGE1_EPOCHS + 1,
            wiggle=0.010,
        )
        _append_loss_curve_rows(
            rows,
            'joint_val_loss',
            0.48 * curve_scale,
            0.19 * curve_scale,
            epochs=JOINT_EPOCHS,
            x_start=STAGE1_EPOCHS + 1,
            wiggle=0.012,
        )
    return pd.DataFrame(rows)


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')


def _write_sequence_progress(output_dir: Path, architecture: str, task_type: str, thresholds: dict[str, float], status: str) -> None:
    _write_markdown(
        output_dir / 'progress.md',
        [
            '# Sequence Baseline Progress',
            '',
            f'- architecture: {architecture}',
            f'- task: {task_type}',
            f'- status: {status}',
            f'- epoch: {SEQUENCE_EPOCHS}/{SEQUENCE_EPOCHS}',
            '- train_samples: simulated',
            '- val_samples: simulated',
            '- test_samples: simulated',
            '- thresholds: ' + ', '.join(f'{fault}:{thresholds[fault]:.3f}' for fault in FAULT_ORDER),
        ],
    )


def _write_lightgbm_progress(output_dir: Path, task_type: str, thresholds: dict[str, float], status: str) -> None:
    _write_markdown(
        output_dir / 'progress.md',
        [
            '# LightGBM Progress',
            '',
            f'- task: {task_type}',
            f'- status: {status}',
            '- completed_faults: ' + ', '.join(FAULT_ORDER),
            '- train_samples: simulated',
            '- val_samples: simulated',
            '- test_samples: simulated',
            '- thresholds: ' + ', '.join(f'{fault}:{thresholds[fault]:.3f}' for fault in FAULT_ORDER),
        ],
    )


def _write_dual_progress(output_dir: Path, ablation_name: str, thresholds_note: str, status: str) -> None:
    _write_markdown(
        output_dir / 'progress.md',
        [
            '# Dual Task Progress',
            '',
            f'- ablation: {ablation_name}',
            f'- status: {status}',
            '- stage: joint',
            f'- epoch: {TOTAL_EPOCHS}/{TOTAL_EPOCHS}',
            '- train_samples: simulated',
            '- val_samples: simulated',
            '- test_samples: simulated',
            f'- thresholds: {thresholds_note}',
        ],
    )


def _write_simulated_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch is None:
        path.write_text('simulated checkpoint placeholder\n', encoding='utf-8')
        return
    torch.save(payload, path)


def _run_script(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)


def _write_bundle_notice(output_dir: Path, model_name: str, task_types: list[str]) -> None:
    lines = [
        f'Simulated normal-performance result bundle for {model_name}.',
        'This directory is synthetic and is intended for paper drafting, layout rehearsal, and sanity comparison only.',
        'Do not mix these outputs with formal experimental conclusions.',
        'tasks=' + ', '.join(task_types),
    ]
    _write_markdown(output_dir / 'notes.md', lines)


def _build_single_task_bundle(
    sample_frame: pd.DataFrame,
    truth_sets: dict[str, dict[str, set[str]]],
    bundle_root: Path,
    dir_name: str,
    model_name: str,
    experiment_name: str,
    task_type: str,
    counts: dict[str, tuple[int, int]],
    curve_kind: str,
    curve_profile,
    status: str,
) -> None:
    frames = []
    thresholds = {}
    for fault in FAULT_ORDER:
        tp_count, fp_count = counts[fault]
        threshold = _threshold_for(model_name, task_type, fault)
        thresholds[fault] = threshold
        frames.append(
            _build_fault_predictions(
                sample_frame=sample_frame,
                positive_ids=truth_sets[task_type][fault],
                tp_count=tp_count,
                fp_count=fp_count,
                threshold=threshold,
                model_name=model_name,
                task_type=task_type,
                fault=fault,
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    summary_rows = _build_summary_rows(predictions, sample_frame, experiment_name, model_name, task_type)
    summary = pd.DataFrame(summary_rows)
    data_points = _build_data_points(predictions, summary, curve_kind, curve_profile=curve_profile)

    output_dir = bundle_root / dir_name
    write_result_bundle(output_dir, summary, predictions, data_points, notes='')
    _write_bundle_notice(output_dir, model_name, [task_type])
    if model_name in {'lstm', 'transformer'}:
        _write_sequence_progress(output_dir, model_name, task_type, thresholds, status)
        _write_simulated_checkpoint(output_dir / 'best_model.pt', {'simulated': True, 'model_name': model_name, 'task_type': task_type})
    elif model_name == 'lightgbm':
        _write_lightgbm_progress(output_dir, task_type, thresholds, status)
    elif model_name == 'threshold_trend':
        _write_markdown(
            output_dir / 'progress.md',
            ['# Threshold Warning Progress', '', f'- task: {task_type}', f'- status: {status}', '- thresholds: ' + ', '.join(f'{fault}:{thresholds[fault]:.3f}' for fault in FAULT_ORDER)],
        )
    elif model_name == 'shared_encoder_expert_heads':
        threshold_note = ', '.join(f'{fault}:{thresholds[fault]:.3f}/{_threshold_for(model_name, "warning" if task_type == "identification" else "identification", fault):.3f}' for fault in FAULT_ORDER)
        _write_dual_progress(output_dir, 'none', threshold_note, status)
        if task_type == 'identification':
            _write_simulated_checkpoint(output_dir / 'best_id_model.pt', {'simulated': True, 'model_name': model_name, 'task_type': task_type})
        else:
            _write_simulated_checkpoint(output_dir / 'best_joint_model.pt', {'simulated': True, 'model_name': model_name, 'task_type': task_type})


def _build_ablation_bundle(sample_frame: pd.DataFrame, truth_sets: dict[str, dict[str, set[str]]], results_root: Path, bundle_spec: dict[str, object]) -> None:
    model_name = str(bundle_spec['model_name'])
    output_dir = results_root / 'ablation' / str(bundle_spec['dir_name'])
    frames = []
    thresholds_id = {}
    thresholds_warn = {}
    for fault in FAULT_ORDER:
        tp_count, fp_count = bundle_spec['identification_counts'][fault]
        threshold = _threshold_for(model_name, 'identification', fault)
        thresholds_id[fault] = threshold
        frames.append(
            _build_fault_predictions(
                sample_frame=sample_frame,
                positive_ids=truth_sets['identification'][fault],
                tp_count=tp_count,
                fp_count=fp_count,
                threshold=threshold,
                model_name=model_name,
                task_type='identification',
                fault=fault,
            )
        )
    for fault in FAULT_ORDER:
        tp_count, fp_count = bundle_spec['warning_counts'][fault]
        threshold = _threshold_for(model_name, 'warning', fault)
        thresholds_warn[fault] = threshold
        frames.append(
            _build_fault_predictions(
                sample_frame=sample_frame,
                positive_ids=truth_sets['warning'][fault],
                tp_count=tp_count,
                fp_count=fp_count,
                threshold=threshold,
                model_name=model_name,
                task_type='warning',
                fault=fault,
            )
        )

    predictions = pd.concat(frames, ignore_index=True)
    summary_rows = _build_summary_rows(predictions, sample_frame, 'dual_task_ablation', model_name, 'identification')
    summary_rows.extend(_build_summary_rows(predictions, sample_frame, 'dual_task_ablation', model_name, 'warning'))
    summary = pd.DataFrame(summary_rows)
    data_points = _build_data_points(
        predictions,
        summary,
        str(bundle_spec['curve_kind']),
        curve_scale=float(bundle_spec['curve_scale']),
    )
    write_result_bundle(output_dir, summary, predictions, data_points, notes='')
    _write_bundle_notice(output_dir, model_name, ['identification', 'warning'])
    thresholds_note = ', '.join(f'{fault}:{thresholds_id[fault]:.3f}/{thresholds_warn[fault]:.3f}' for fault in FAULT_ORDER)
    _write_dual_progress(output_dir, str(bundle_spec['dir_name']), thresholds_note, 'completed stage1+joint')
    _write_simulated_checkpoint(output_dir / 'best_id_model.pt', {'simulated': True, 'model_name': model_name, 'task_type': 'identification'})
    _write_simulated_checkpoint(output_dir / 'best_joint_model.pt', {'simulated': True, 'model_name': model_name, 'task_type': 'warning'})



def _bundle_specs_with_counts(bundle_specs: list[dict[str, object]], counts_by_model: dict[str, dict[str, tuple[int, int]]]) -> list[dict[str, object]]:
    updated_specs: list[dict[str, object]] = []
    for spec in bundle_specs:
        updated = dict(spec)
        updated['counts'] = counts_by_model[str(spec['model_name'])]
        updated_specs.append(updated)
    return updated_specs


def _ablation_specs_with_counts(
    bundle_specs: list[dict[str, object]],
    recognition_counts: dict[str, dict[str, tuple[int, int]]],
    warning_counts: dict[str, dict[str, tuple[int, int]]],
) -> list[dict[str, object]]:
    updated_specs: list[dict[str, object]] = []
    for spec in bundle_specs:
        updated = dict(spec)
        model_name = str(spec['model_name'])
        updated['identification_counts'] = recognition_counts[model_name]
        updated['warning_counts'] = warning_counts[model_name]
        updated_specs.append(updated)
    return updated_specs


def _profile_specs(profile: str) -> dict[str, object]:
    if profile == 'normal':
        return {
            'identification_positive_indices': IDENTIFICATION_POSITIVE_INDICES,
            'warning_positive_indices': WARNING_POSITIVE_INDICES,
            'recognition_bundles': RECOGNITION_BUNDLES,
            'warning_bundles': WARNING_BUNDLES,
            'ablation_bundles': ABLATION_BUNDLES,
        }
    if profile == 'polished':
        return {
            'identification_positive_indices': POLISHED_IDENTIFICATION_POSITIVE_INDICES,
            'warning_positive_indices': POLISHED_WARNING_POSITIVE_INDICES,
            'recognition_bundles': _bundle_specs_with_counts(RECOGNITION_BUNDLES, POLISHED_RECOGNITION_COUNTS),
            'warning_bundles': _bundle_specs_with_counts(WARNING_BUNDLES, POLISHED_WARNING_COUNTS),
            'ablation_bundles': _ablation_specs_with_counts(
                ABLATION_BUNDLES,
                POLISHED_ABLATION_RECOGNITION_COUNTS,
                POLISHED_ABLATION_WARNING_COUNTS,
            ),
        }
    raise ValueError(f'Unknown simulation profile: {profile}')


def _write_polished_sample_summary(samples_root: Path, sample_frame: pd.DataFrame) -> None:
    samples_root.mkdir(parents=True, exist_ok=True)
    sample_frame.to_csv(samples_root / 'samples_master.csv', index=False, encoding='utf-8-sig')
    rows: list[dict[str, object]] = []
    for split, sample_count in POLISHED_DISPLAY_SAMPLE_COUNTS.items():
        row: dict[str, object] = {'split': split, 'sample_count': sample_count}
        for fault in FAULT_ORDER:
            id_count, warn_count = POLISHED_DISPLAY_LABEL_COUNTS[split][fault]
            row[f'y_id_{fault}'] = id_count
            row[f'y_warn_{fault}'] = warn_count
        rows.append(row)
    pd.DataFrame(rows).to_csv(samples_root / 'full_scale_summary.csv', index=False, encoding='utf-8-sig')
    _write_markdown(
        samples_root / 'full_scale_notes.md',
        [
            '# Polished Simulated Full-Scale Summary',
            '',
            '- This file is synthetic and is intended only for thesis-drafting rehearsal.',
            '- It preserves the full-data split sizes while moderately lifting rare-fault positive counts for display stress-testing.',
            '- Do not treat these counts as the formal Data_set sample statistics.',
        ],
    )

def generate_simulated_results(results_root: Path, keep_existing: bool = False, profile: str = 'normal') -> dict[str, Path]:
    profile_spec = _profile_specs(profile)
    results_root = Path(results_root)
    if results_root.exists() and not keep_existing:
        shutil.rmtree(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    sample_frame = _build_sample_frame(profile_spec['warning_positive_indices'])
    truth_sets = _build_truth_sets(
        sample_frame,
        identification_positive_indices=profile_spec['identification_positive_indices'],
        warning_positive_indices=profile_spec['warning_positive_indices'],
    )
    synthetic_samples_root = results_root / '_synthetic_samples'
    synthetic_samples_root.mkdir(parents=True, exist_ok=True)
    sample_frame.to_csv(synthetic_samples_root / 'samples_master.csv', index=False, encoding='utf-8-sig')
    display_samples_root = synthetic_samples_root
    if profile == 'polished':
        display_samples_root = results_root / '_polished_samples'
        _write_polished_sample_summary(display_samples_root, sample_frame)

    _write_markdown(
        results_root / 'SIMULATION_NOTICE.md',
        [
            '# Simulated Round2 Results',
            '',
            '- This bundle is synthetic and is intended only for paper drafting and result-shape rehearsal.',
            '- It should not be reported as a formal experiment result.',
            f'- Profile: {profile}',
            '- The ranking is intentionally shaped to reflect a plausible outcome for paper-writing rehearsal.',
            '- The polished profile lifts core recognition and warning scores while keeping rare faults visibly harder than common faults.',
        ],
    )

    recognition_root = results_root / 'recognition'
    warning_root = results_root / 'warning'
    ablation_root = results_root / 'ablation'
    recognition_root.mkdir(parents=True, exist_ok=True)
    warning_root.mkdir(parents=True, exist_ok=True)
    ablation_root.mkdir(parents=True, exist_ok=True)

    for bundle in profile_spec['recognition_bundles']:
        _build_single_task_bundle(
            sample_frame=sample_frame,
            truth_sets=truth_sets,
            bundle_root=recognition_root,
            dir_name=str(bundle['dir_name']),
            model_name=str(bundle['model_name']),
            experiment_name=str(bundle['experiment_name']),
            task_type=str(bundle['task_type']),
            counts=bundle['counts'],
            curve_kind=str(bundle['curve_kind']),
            curve_profile=bundle.get('curve_profile'),
            status=str(bundle['status']),
        )

    for bundle in profile_spec['warning_bundles']:
        _build_single_task_bundle(
            sample_frame=sample_frame,
            truth_sets=truth_sets,
            bundle_root=warning_root,
            dir_name=str(bundle['dir_name']),
            model_name=str(bundle['model_name']),
            experiment_name=str(bundle['experiment_name']),
            task_type=str(bundle['task_type']),
            counts=bundle['counts'],
            curve_kind=str(bundle['curve_kind']),
            curve_profile=bundle.get('curve_profile'),
            status=str(bundle['status']),
        )

    for bundle in profile_spec['ablation_bundles']:
        _build_ablation_bundle(sample_frame, truth_sets, results_root, bundle)

    aggregate_round1_tables(results_root)

    figures_root = results_root / 'figures'
    _run_script(['scripts/export_recognition_figures.py', '--results-root', str(results_root), '--figures-root', str(figures_root)])
    _run_script(['scripts/export_warning_figures.py', '--results-root', str(results_root), '--figures-root', str(figures_root)])
    _run_script(['scripts/export_ablation_figures.py', '--results-root', str(results_root), '--figures-root', str(figures_root), '--ablation-name', 'no_expert_heads'])
    _run_script(['scripts/export_plan_supplements.py', '--results-root', str(results_root), '--samples-root', str(synthetic_samples_root)])

    return {
        'results_root': results_root,
        'samples_root': synthetic_samples_root,
        'display_samples_root': display_samples_root,
        'tables_root': results_root / 'tables',
        'figures_root': figures_root,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = generate_simulated_results(Path(args.results_root), keep_existing=bool(args.keep_existing), profile=str(args.profile))
    for name, path in outputs.items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
