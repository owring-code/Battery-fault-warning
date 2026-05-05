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
    parser = argparse.ArgumentParser(description='Export recognition figures from round-one result bundles.')
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--figures-root', default='results/figures')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    figures_root = Path(args.figures_root) / 'recognition'
    training_figures_root = Path(args.figures_root) / 'training'
    figures_root.mkdir(parents=True, exist_ok=True)
    training_figures_root.mkdir(parents=True, exist_ok=True)

    recognition_points = pd.read_csv(results_root / 'recognition' / 'main_dual_task' / 'data_points.csv')

    core_faults = ['sd', 'samp', 'ins']
    bar_data = recognition_points[
        (recognition_points['plot_type'] == 'bar_metric')
        & (recognition_points['category'].isin(core_faults))
    ].copy()
    bar_data_path = figures_root / 'fig3_recognition_bar_data.csv'
    bar_data.to_csv(bar_data_path, index=False, encoding='utf-8-sig')
    _copy_plot_script('plot_recognition_bar.py', figures_root / 'fig3_recognition_bar.py')
    _run_plotter(
        'plot_recognition_bar.py',
        bar_data_path,
        figures_root / 'fig3_recognition_bar.png',
        figures_root / 'fig3_recognition_bar.svg',
        '核心故障识别性能比较',
    )

    loss_data = recognition_points[recognition_points['plot_type'] == 'loss_curve'].copy()
    loss_data_path = training_figures_root / 'fig2_loss_curve_data.csv'
    loss_data.to_csv(loss_data_path, index=False, encoding='utf-8-sig')
    _copy_plot_script('plot_loss_curve.py', training_figures_root / 'fig2_loss_curve.py')
    _run_plotter(
        'plot_loss_curve.py',
        loss_data_path,
        training_figures_root / 'fig2_loss_curve.png',
        training_figures_root / 'fig2_loss_curve.svg',
        '双任务模型训练收敛曲线',
    )


if __name__ == '__main__':
    main()
