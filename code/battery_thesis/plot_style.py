from __future__ import annotations

import matplotlib
from matplotlib import font_manager


ACADEMIC_COLORS = {
    'ink': '#243447',
    'grid': '#D7DCE5',
    'teal': '#00A087',
    'teal_light': '#91D1C2',
    'coral': '#E64B35',
    'coral_light': '#F39B7F',
    'sage': '#7E6148',
    'gold': '#B09C85',
    'slate': '#7C8798',
    'navy': '#3C5488',
}


def _pick_first_available(candidates: list[str], installed: set[str]) -> str | None:
    for name in candidates:
        if name in installed:
            return name
    return None


def apply_academic_plot_style() -> dict[str, object]:
    installed = {font.name for font in font_manager.fontManager.ttflist}

    serif_font = _pick_first_available(
        ['Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
        installed,
    )
    cjk_font = _pick_first_available(
        [
            'SimSun',
            'SimHei',
            'Microsoft YaHei',
            'Noto Sans CJK SC',
            'Noto Serif CJK SC',
            'Source Han Sans SC',
            'WenQuanYi Micro Hei',
            'AR PL UMing CN',
            'PingFang SC',
        ],
        installed,
    )

    family: list[str] = []
    if serif_font:
        family.append(serif_font)
    if cjk_font and cjk_font not in family:
        family.append(cjk_font)
    if 'DejaVu Sans' not in family:
        family.append('DejaVu Sans')

    style = {
        'font.family': family,
        'font.sans-serif': [font for font in [cjk_font, 'DejaVu Sans'] if font],
        'font.serif': [font for font in [serif_font, 'DejaVu Serif'] if font],
        'axes.unicode_minus': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': ACADEMIC_COLORS['ink'],
        'axes.labelcolor': ACADEMIC_COLORS['ink'],
        'axes.titlecolor': ACADEMIC_COLORS['ink'],
        'axes.titleweight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': ACADEMIC_COLORS['grid'],
        'grid.linewidth': 0.8,
        'grid.alpha': 0.8,
        'grid.linestyle': '-',
        'xtick.color': ACADEMIC_COLORS['ink'],
        'ytick.color': ACADEMIC_COLORS['ink'],
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.frameon': False,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12,
        'lines.linewidth': 2.0,
        'lines.markersize': 5.5,
    }
    matplotlib.rcParams.update(style)
    return style


def get_academic_palette() -> dict[str, str]:
    return dict(ACADEMIC_COLORS)


def style_axis(ax, grid_axis: str = 'y'):
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=ACADEMIC_COLORS['grid'], linewidth=0.8, alpha=0.8)
    ax.tick_params(axis='both', which='major', length=0)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color(ACADEMIC_COLORS['ink'])
    return ax
