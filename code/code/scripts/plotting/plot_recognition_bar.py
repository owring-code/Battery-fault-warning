from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.plot_style import apply_academic_plot_style, get_academic_palette, style_axis

SERIES_LABELS = {
    'f1': 'F1',
    'recall': 'Recall',
    'mean_core_f1': 'Mean Core F1',
    'mean_core_recall': 'Mean Core Recall',
}

CATEGORY_ORDER = {
    'sd': 0,
    'isc': 1,
    'ins': 2,
    'samp': 3,
    'conn': 4,
    'Threshold Trend': 10,
    'LightGBM': 11,
    'LSTM': 12,
    'Vanilla Transformer': 13,
    'Proposed Method': 14,
    'No Warning Task': 15,
    'No Fault-Specific Features': 16,
    'No Expert Heads': 17,
    'No Label Quality Control': 18,
}

SERIES_ORDER = {'f1': 0, 'mean_core_f1': 0, 'recall': 1, 'mean_core_recall': 1}

CATEGORY_LABELS = {
    'sd': 'Self-discharge',
    'isc': 'Sudden ISC',
    'ins': 'Insulation failure',
    'samp': 'Sampling anomaly',
    'conn': 'Connection anomaly',
    'Proposed Method': 'Proposed Method',
    'Vanilla Transformer': 'Transformer',
    'LSTM': 'LSTM',
    'LightGBM': 'LightGBM',
    'Threshold Trend': 'Threshold Trend',
    'No Fault-Specific Features': 'No fault-specific features',
    'No Expert Heads': 'No expert heads',
    'No Warning Task': 'No warning task',
    'No Label Quality Control': 'No label-quality control',
}


def _translate_category(value: str) -> str:
    return CATEGORY_LABELS.get(str(value), str(value))


def _translate_series(value: str) -> str:
    return SERIES_LABELS.get(str(value), str(value))


def _series_colors(series_names: list[str]) -> list[str]:
    palette = get_academic_palette()
    preferred = {
        'f1': palette['teal'],
        'recall': palette['coral'],
        'mean_core_f1': palette['teal'],
        'mean_core_recall': palette['coral'],
    }
    fallback = [palette['teal'], palette['coral'], palette['sage'], palette['gold']]
    colors = []
    for index, name in enumerate(series_names):
        colors.append(preferred.get(name, fallback[index % len(fallback)]))
    return colors


def _set_tick_style(ax, raw_labels: list[str]) -> None:
    translated = [_translate_category(label) for label in raw_labels]
    ax.set_xticks(np.arange(len(translated)))
    ax.set_xticklabels(translated)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    for tick, raw in zip(ax.get_xticklabels(), raw_labels):
        if raw == 'Proposed Method':
            tick.set_fontweight('bold')
            tick.set_color(get_academic_palette()['navy'])


def _annotate_bars(ax, containers) -> None:
    for container in containers:
        labels = []
        for bar in container:
            value = bar.get_height()
            labels.append(f'{value:.3f}' if value >= 0.1 else f'{value:.2f}')
        ax.bar_label(container, labels=labels, padding=4, fontsize=11, color=get_academic_palette()['ink'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_png', required=True)
    parser.add_argument('--output_svg', required=True)
    parser.add_argument('--title', default='Core recognition performance')
    args = parser.parse_args()

    apply_academic_plot_style()
    data = pd.read_csv(args.input)
    pivot = data.pivot(index='category', columns='series', values='value').fillna(0.0)
    ordered_categories = sorted(pivot.index.tolist(), key=lambda value: (CATEGORY_ORDER.get(str(value), 999), str(value)))
    ordered_series = sorted(pivot.columns.tolist(), key=lambda value: (SERIES_ORDER.get(str(value), 999), str(value)))
    pivot = pivot.reindex(index=ordered_categories, columns=ordered_series)
    categories = pivot.index.tolist()
    series_names = pivot.columns.tolist()
    colors = _series_colors(series_names)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(categories))
    width = 0.32 if len(series_names) <= 2 else 0.8 / max(len(series_names), 1)
    containers = []
    for idx, series_name in enumerate(series_names):
        offset = (idx - (len(series_names) - 1) / 2.0) * width
        container = ax.bar(
            x + offset,
            pivot[series_name].to_numpy(dtype=float),
            width=width * 0.92,
            label=_translate_series(series_name),
            color=colors[idx],
            edgecolor='white',
            linewidth=0.9,
            alpha=0.96,
        )
        containers.append(container)

    style_axis(ax, grid_axis='y')
    _set_tick_style(ax, categories)
    _annotate_bars(ax, containers)
    ax.set_ylim(0.0, min(1.02, max(0.85, float(pivot.to_numpy(dtype=float).max()) + 0.12)))
    legend = ax.legend(loc='upper center', ncol=max(1, len(series_names)), bbox_to_anchor=(0.5, 1.04))
    for text in legend.get_texts():
        text.set_color(get_academic_palette()['ink'])
    plt.tight_layout()
    plt.savefig(args.output_png)
    plt.savefig(args.output_svg)


if __name__ == '__main__':
    main()
