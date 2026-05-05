from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.plot_style import apply_academic_plot_style, get_academic_palette, style_axis

SERIES_LABELS = {
    'score': 'Prediction score',
    'threshold': 'Decision threshold',
    'y_true': 'Positive interval',
    'selected_window': 'Selected window',
}


def _series(data: pd.DataFrame, name: str) -> pd.DataFrame:
    subset = data[data['series'] == name].copy()
    if subset.empty:
        return subset
    return subset.sort_values('x').reset_index(drop=True)


def _selected_x(data: pd.DataFrame) -> float | None:
    selected = data[data['series'] == 'selected_window']
    if selected.empty:
        return None
    return float(selected['x'].iloc[0])


def _shade_selected_window(ax, selected_x: float | None, color: str) -> None:
    if selected_x is None:
        return
    ax.axvspan(selected_x - 0.5, selected_x + 0.5, color=color, alpha=0.12, linewidth=0)
    ax.axvline(selected_x, color=color, linestyle='-', linewidth=1.8, alpha=0.78)


def _integer_ticks(x_min: float, x_max: float, max_ticks: int = 9) -> list[int]:
    lower = math.floor(float(x_min))
    upper = math.ceil(float(x_max))
    if lower > upper:
        lower, upper = upper, lower
    span = max(upper - lower, 1)
    step = max(1, math.ceil(span / max(max_ticks - 1, 1)))
    ticks = list(range(lower, upper + 1, step))
    if ticks[0] > lower:
        ticks.insert(0, lower)
    if ticks[-1] < upper:
        ticks.append(upper)
    if lower <= 0 <= upper and 0 not in ticks:
        ticks.append(0)
        ticks = sorted(set(ticks))
    return ticks
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_png', required=True)
    parser.add_argument('--output_svg', required=True)
    parser.add_argument('--title', default='Case trend')
    args = parser.parse_args()

    apply_academic_plot_style()
    palette = get_academic_palette()
    data = pd.read_csv(args.input)
    score = _series(data, 'score')
    threshold = _series(data, 'threshold')
    truth = _series(data, 'y_true')
    selected_x = _selected_x(data)

    fig, (ax_score, ax_truth) = plt.subplots(
        2,
        1,
        figsize=(9.4, 5.4),
        sharex=True,
        gridspec_kw={'height_ratios': [3.2, 1.0], 'hspace': 0.08},
    )

    if not score.empty and not threshold.empty:
        merged = score[['x', 'y']].merge(threshold[['x', 'y']], on='x', suffixes=('_score', '_threshold'))
        ax_score.fill_between(
            merged['x'],
            merged['y_threshold'],
            merged['y_score'],
            where=merged['y_score'] >= merged['y_threshold'],
            color=palette['teal_light'],
            alpha=0.24,
            interpolate=True,
            label='_nolegend_',
        )

    if not score.empty:
        ax_score.plot(
            score['x'],
            score['y'],
            label=SERIES_LABELS['score'],
            color=palette['teal'],
            linewidth=2.6,
            marker='o',
            markersize=4.5,
            markerfacecolor='white',
            markeredgewidth=1.3,
        )
    if not threshold.empty:
        ax_score.plot(
            threshold['x'],
            threshold['y'],
            label=SERIES_LABELS['threshold'],
            color=palette['coral'],
            linestyle='--',
            linewidth=2.2,
        )

    if selected_x is not None and not score.empty:
        selected_score = score.loc[score['x'].eq(selected_x), 'y']
        if not selected_score.empty:
            ax_score.scatter(
                [selected_x],
                [float(selected_score.iloc[0])],
                s=96,
                color=palette['navy'],
                edgecolor='white',
                linewidth=1.4,
                zorder=8,
                label=SERIES_LABELS['selected_window'],
            )

    if not truth.empty:
        ax_truth.fill_between(
            truth['x'],
            0,
            1,
            where=truth['y'] >= 0.5,
            step='mid',
            color=palette['navy'],
            alpha=0.22,
            label=SERIES_LABELS['y_true'],
        )
        ax_truth.step(truth['x'], truth['y'], where='mid', color=palette['slate'], linewidth=1.7, alpha=0.95)

    _shade_selected_window(ax_score, selected_x, palette['navy'])
    _shade_selected_window(ax_truth, selected_x, palette['navy'])

    style_axis(ax_score, grid_axis='y')
    style_axis(ax_truth, grid_axis='x')
    ax_score.set_ylabel('Prediction score')
    ax_score.set_ylim(-0.02, 1.05)
    ax_truth.set_ylabel('Ground truth')
    ax_truth.set_xlabel('Relative window index')
    ax_truth.set_ylim(-0.08, 1.08)
    ax_truth.set_yticks([0, 1])
    ax_truth.set_yticklabels(['0', '1'])
    if not score.empty:
        integer_ticks = _integer_ticks(score['x'].min(), score['x'].max())
        ax_truth.set_xticks(integer_ticks)
        ax_truth.set_xticklabels([str(tick) for tick in integer_ticks])

    handles, labels = [], []
    for axis in [ax_score, ax_truth]:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    ax_score.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        fontsize=11,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.13, hspace=0.08)
    plt.savefig(args.output_png)
    plt.savefig(args.output_svg)


if __name__ == '__main__':
    main()
