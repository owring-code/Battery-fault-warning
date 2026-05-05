from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTTING_ROOT = PROJECT_ROOT / 'scripts' / 'plotting'


def _run_plotter(script_name: str, input_path: Path, output_png: Path, output_svg: Path, title: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PLOTTING_ROOT / script_name),
            '--input',
            str(input_path),
            '--output_png',
            str(output_png),
            '--output_svg',
            str(output_svg),
            '--title',
            title,
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )


def _copy_plot_script(source_script: str, destination: Path) -> None:
    shutil.copyfile(PLOTTING_ROOT / source_script, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export warning figures from round-one result bundles.')
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--figures-root', default='results/figures')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    figures_root = Path(args.figures_root) / 'warning'
    figures_root.mkdir(parents=True, exist_ok=True)

    core_faults = ['sd', 'samp', 'ins']
    warning_points = pd.read_csv(results_root / 'warning' / 'main_dual_task' / 'data_points.csv')
    bar_data = warning_points[
        (warning_points['plot_type'] == 'bar_metric')
        & (warning_points['category'].isin(core_faults))
    ].copy()
    bar_data_path = figures_root / 'fig4_warning_bar_data.csv'
    bar_data.to_csv(bar_data_path, index=False, encoding='utf-8-sig')
    _copy_plot_script('plot_warning_bar.py', figures_root / 'fig4_warning_bar.py')
    _run_plotter(
        'plot_warning_bar.py',
        bar_data_path,
        figures_root / 'fig4_warning_bar.png',
        figures_root / 'fig4_warning_bar.svg',
        '核心故障预警性能比较',
    )


if __name__ == '__main__':
    main()
