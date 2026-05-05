from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.plot_style import apply_academic_plot_style
from scripts.export_chapter5_simulated_supplements import _export_ablation_figure, _export_ablation_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export ablation figures from result bundles.')
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--figures-root', default='results/figures')
    parser.add_argument('--ablation-name', default='no_expert_heads', help='Retained for backward compatibility; the exported figure compares all ablation variants.')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    apply_academic_plot_style()
    results_root = Path(args.results_root)
    figures_root = Path(args.figures_root) / 'ablation'
    figures_root.mkdir(parents=True, exist_ok=True)

    ablation = _export_ablation_table(results_root, figures_root)
    _export_ablation_figure(results_root, figures_root, ablation)


if __name__ == '__main__':
    main()
