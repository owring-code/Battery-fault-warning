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
    'train_loss': 'Train loss',
    'val_loss': 'Validation loss',
    'stage1_train_loss': 'Stage-I train loss',
    'stage1_val_loss': 'Stage-I validation loss',
    'joint_train_loss': 'Joint train loss',
    'joint_val_loss': 'Joint validation loss',
    'stage1_warn_loss': 'Stage-I warning loss',
    'joint_warn_loss': 'Joint warning loss',
}

SERIES_ORDER = {
    'stage1_train_loss': 0,
    'stage1_val_loss': 1,
    'joint_train_loss': 2,
    'joint_val_loss': 3,
    'stage1_warn_loss': 4,
    'joint_warn_loss': 5,
    'train_loss': 6,
    'val_loss': 7,
}

SERIES_STYLES = {
    'train_loss': ('teal', '-'),
    'val_loss': ('coral', '-'),
    'stage1_train_loss': ('teal', '-'),
    'stage1_val_loss': ('teal_light', '--'),
    'joint_train_loss': ('coral', '-'),
    'joint_val_loss': ('coral_light', '--'),
    'stage1_warn_loss': ('sage', '-'),
    'joint_warn_loss': ('gold', '-'),
}

MARKERS = ['o', 's', '^', 'D', 'P', 'X']


def _translate_series(series_name: str) -> str:
    return SERIES_LABELS.get(str(series_name), str(series_name))


def _stage_transition_label_y(y_min: float, y_max: float) -> float:
    span = max(float(y_max) - float(y_min), 1e-6)
    return float(y_min) + span * 0.82


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_png', required=True)
    parser.add_argument('--output_svg', required=True)
    parser.add_argument('--title', default='Dual-task training convergence')
    args = parser.parse_args()

    apply_academic_plot_style()
    palette = get_academic_palette()
    data = pd.read_csv(args.input).copy()
    data['x'] = pd.to_numeric(data['x'], errors='coerce')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data = data.dropna(subset=['x', 'y'])

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    grouped_series = sorted(data.groupby('series'), key=lambda item: (SERIES_ORDER.get(str(item[0]), 999), str(item[0])))
    for index, (series_name, subset) in enumerate(grouped_series):
        subset = subset.sort_values('x')
        color_key, line_style = SERIES_STYLES.get(series_name, ('slate', '-'))
        marker = MARKERS[index % len(MARKERS)]
        ax.plot(
            subset['x'],
            subset['y'],
            label=_translate_series(series_name),
            color=palette[color_key],
            linestyle=line_style,
            marker=marker,
            markevery=max(1, len(subset) // 10),
            linewidth=2.2,
            alpha=0.96,
        )

    stage1_mask = data['series'].astype(str).str.startswith('stage1_')
    joint_mask = data['series'].astype(str).str.startswith('joint_')
    if stage1_mask.any() and joint_mask.any():
        transition_x = float(data.loc[stage1_mask, 'x'].max()) + 0.5
        ax.axvline(transition_x, color=palette['slate'], linestyle=':', linewidth=1.3)
        ax.text(transition_x + 0.8, _stage_transition_label_y(float(data['y'].min()), float(data['y'].max())), 'Stage transition', color=palette['slate'], fontsize=12)

    style_axis(ax, grid_axis='both')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(left=max(0.0, float(data['x'].min()) - 1.0), right=float(data['x'].max()) + 1.5)
    ax.set_ylim(bottom=max(0.0, float(data['y'].min()) - 0.03), top=float(data['y'].max()) + 0.05)
    max_epoch = int(data['x'].max())
    if max_epoch >= 160:
        tick_step = 20
    elif max_epoch >= 80:
        tick_step = 10
    elif max_epoch >= 40:
        tick_step = 5
    else:
        tick_step = None
    if tick_step:
        ax.set_xticks(np.arange(0, max_epoch + 1, tick_step))
    legend = ax.legend(loc='upper right', ncol=2)
    for text in legend.get_texts():
        text.set_color(palette['ink'])
    plt.tight_layout()
    plt.savefig(args.output_png)
    plt.savefig(args.output_svg)


if __name__ == '__main__':
    main()
