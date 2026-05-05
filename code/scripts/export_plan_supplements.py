from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTTING_ROOT = PROJECT_ROOT / 'scripts' / 'plotting'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.supplementary_results import (
    CORE_FAULTS,
    build_metric_bar_frame,
    case_rows_to_frame,
    compute_multifault_summary_from_predictions,
    load_recognition_prediction_frames,
    normalize_model_name,
    select_case_examples,
    summarize_core_metrics,
)

RECOGNITION_MODEL_ORDER = ['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'shared_encoder_expert_heads']
WARNING_MODEL_ORDER = ['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'shared_encoder_expert_heads']
ALL_FAULTS = ['sd', 'isc', 'ins', 'samp', 'conn']


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
    parser = argparse.ArgumentParser(description='Export supplementary PLAN-aligned tables and figures from round-two results.')
    parser.add_argument('--results-root', default='results/round2')
    parser.add_argument('--samples-root', default='artifacts/round2/samples')
    parser.add_argument('--success-fault', default='isc')
    parser.add_argument('--failure-fault', default='samp')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    samples_root = Path(args.samples_root)
    tables_root = results_root / 'tables'
    comparison_root = results_root / 'figures' / 'comparison'
    cases_root = results_root / 'figures' / 'cases'
    for path in [tables_root, comparison_root, cases_root]:
        path.mkdir(parents=True, exist_ok=True)

    recognition_summaries = _load_named_summaries(results_root / 'recognition', ['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'main_dual_task'])
    warning_summaries = _load_named_summaries(results_root / 'warning', ['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'main_dual_task'])

    recognition_comparison = summarize_core_metrics(
        recognition_summaries,
        metric_columns=['accuracy', 'f1', 'recall', 'precision', 'pr_auc', 'roc_auc'],
        model_order=RECOGNITION_MODEL_ORDER,
    ).rename(
        columns={
            'accuracy': 'mean_core_accuracy',
            'f1': 'mean_core_f1',
            'recall': 'mean_core_recall',
            'precision': 'mean_core_precision',
            'pr_auc': 'mean_core_pr_auc',
            'roc_auc': 'mean_core_roc_auc',
        }
    )
    recognition_comparison.to_csv(tables_root / 'recognition_model_comparison_core.csv', index=False, encoding='utf-8-sig')
    recognition_all_comparison = summarize_core_metrics(
        recognition_summaries,
        metric_columns=['accuracy', 'f1', 'recall', 'precision', 'pr_auc', 'roc_auc'],
        core_faults=ALL_FAULTS,
        model_order=RECOGNITION_MODEL_ORDER,
    ).rename(
        columns={
            'accuracy': 'mean_all_accuracy',
            'f1': 'mean_all_f1',
            'recall': 'mean_all_recall',
            'precision': 'mean_all_precision',
            'pr_auc': 'mean_all_pr_auc',
            'roc_auc': 'mean_all_roc_auc',
        }
    )
    recognition_all_comparison.to_csv(tables_root / 'recognition_model_comparison_all.csv', index=False, encoding='utf-8-sig')
    warning_comparison = summarize_core_metrics(
        warning_summaries,
        metric_columns=['warning_f1', 'warning_recall', 'false_alarm_rate', 'avg_lead_time'],
        model_order=WARNING_MODEL_ORDER,
    ).rename(
        columns={
            'warning_f1': 'mean_core_warning_f1',
            'warning_recall': 'mean_core_warning_recall',
            'false_alarm_rate': 'mean_core_false_alarm_rate',
            'avg_lead_time': 'mean_core_avg_lead_time',
        }
    )
    warning_comparison.to_csv(tables_root / 'warning_model_comparison_core.csv', index=False, encoding='utf-8-sig')
    warning_all_comparison = summarize_core_metrics(
        warning_summaries,
        metric_columns=['warning_f1', 'warning_recall', 'false_alarm_rate', 'avg_lead_time'],
        core_faults=ALL_FAULTS,
        model_order=WARNING_MODEL_ORDER,
    ).rename(
        columns={
            'warning_f1': 'mean_all_warning_f1',
            'warning_recall': 'mean_all_warning_recall',
            'false_alarm_rate': 'mean_all_false_alarm_rate',
            'avg_lead_time': 'mean_all_avg_lead_time',
        }
    )
    warning_all_comparison.to_csv(tables_root / 'warning_model_comparison_all.csv', index=False, encoding='utf-8-sig')
    recognition_prediction_frames = load_recognition_prediction_frames(results_root)
    multifault_summary = compute_multifault_summary_from_predictions(recognition_prediction_frames)
    multifault_summary.to_csv(tables_root / 'recognition_multifault_summary.csv', index=False, encoding='utf-8-sig')

    recognition_bar_data = build_metric_bar_frame(
        recognition_comparison,
        category_column='model_name',
        metric_columns=['mean_core_f1', 'mean_core_recall'],
    )
    recognition_bar_path = comparison_root / 'fig6_recognition_model_comparison_data.csv'
    recognition_bar_data.to_csv(recognition_bar_path, index=False, encoding='utf-8-sig')
    _copy_plot_script('plot_recognition_bar.py', comparison_root / 'fig6_recognition_model_comparison.py')
    _run_plotter(
        'plot_recognition_bar.py',
        recognition_bar_path,
        comparison_root / 'fig6_recognition_model_comparison.png',
        comparison_root / 'fig6_recognition_model_comparison.svg',
        '核心故障识别模型对比',
    )

    warning_bar_data = build_metric_bar_frame(
        warning_comparison,
        category_column='model_name',
        metric_columns=['mean_core_warning_f1', 'mean_core_warning_recall'],
    )
    warning_bar_path = comparison_root / 'fig7_warning_model_comparison_data.csv'
    warning_bar_data.to_csv(warning_bar_path, index=False, encoding='utf-8-sig')
    _copy_plot_script('plot_warning_bar.py', comparison_root / 'fig7_warning_model_comparison.py')
    _run_plotter(
        'plot_warning_bar.py',
        warning_bar_path,
        comparison_root / 'fig7_warning_model_comparison.png',
        comparison_root / 'fig7_warning_model_comparison.svg',
        '核心故障预警模型对比',
    )

    samples_master = pd.read_csv(samples_root / 'samples_master.csv')
    main_predictions = recognition_prediction_frames['shared_encoder_expert_heads']
    case_examples = select_case_examples(
        samples_master,
        main_predictions,
        success_fault=args.success_fault,
        failure_fault=args.failure_fault,
    )
    if len(case_examples) < 2:
        case_examples = _select_cases_with_fallback(samples_master, main_predictions, args.success_fault, args.failure_fault)

    case_summary = case_rows_to_frame(case_examples)
    case_summary.to_csv(tables_root / 'case_analysis_summary.csv', index=False, encoding='utf-8-sig')
    for index, case in enumerate(case_examples, start=8):
        prefix = f'fig{index}_{case.case_name}'
        data_path = cases_root / f'{prefix}_data.csv'
        case.plot_frame.to_csv(data_path, index=False, encoding='utf-8-sig')
        _copy_plot_script('plot_case_trend.py', cases_root / f'{prefix}.py')
        _run_plotter(
            'plot_case_trend.py',
            data_path,
            cases_root / f'{prefix}.png',
            cases_root / f'{prefix}.svg',
            case.title,
        )

    plan_alignment = _build_plan_alignment_notes(
        recognition_comparison,
        warning_comparison,
        multifault_summary,
        case_summary,
        results_root,
    )
    (tables_root / 'plan_alignment_notes.md').write_text(plan_alignment, encoding='utf-8-sig')

    print(f'recognition_model_comparison_core: {tables_root / "recognition_model_comparison_core.csv"}')
    print(f'warning_model_comparison_core: {tables_root / "warning_model_comparison_core.csv"}')
    print(f'recognition_multifault_summary: {tables_root / "recognition_multifault_summary.csv"}')
    print(f'case_analysis_summary: {tables_root / "case_analysis_summary.csv"}')
    print(f'plan_alignment_notes: {tables_root / "plan_alignment_notes.md"}')


def _load_named_summaries(parent: Path, subdirs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in subdirs:
        summary_path = parent / name / 'summary.csv'
        if summary_path.exists():
            frame = pd.read_csv(summary_path)
            frame['model_name'] = frame['model_name'].astype(str).replace({'main_dual_task': 'shared_encoder_expert_heads'})
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _select_cases_with_fallback(samples_master: pd.DataFrame, predictions_df: pd.DataFrame, success_fault: str, failure_fault: str):
    success_candidates = [success_fault, 'isc', 'sd', 'samp', 'ins', 'conn']
    failure_candidates = [failure_fault, 'samp', 'sd', 'ins', 'isc', 'conn']
    for success_name in success_candidates:
        for failure_name in failure_candidates:
            cases = select_case_examples(samples_master, predictions_df, success_fault=success_name, failure_fault=failure_name)
            if len(cases) >= 2:
                return cases
    return select_case_examples(samples_master, predictions_df, success_fault='sd', failure_fault='samp')


def _build_plan_alignment_notes(
    recognition_comparison: pd.DataFrame,
    warning_comparison: pd.DataFrame,
    multifault_summary: pd.DataFrame,
    case_summary: pd.DataFrame,
    results_root: Path,
) -> str:
    recognition_best = _best_model_line(recognition_comparison, 'mean_core_f1')
    warning_best = _best_model_line(warning_comparison, 'mean_core_warning_f1')
    multifault_all = multifault_summary[multifault_summary['fault_scope'] == 'all_faults'].copy()
    macro_best = _best_model_line(multifault_all, 'macro_f1')
    micro_best = _best_model_line(multifault_all, 'micro_f1')

    no_warning_summary = results_root / 'ablation' / 'no_warning_task' / 'summary.csv'
    main_summary = results_root / 'recognition' / 'main_dual_task' / 'summary.csv'
    benefit_note = '- Dual-task benefit check is unavailable because the required summaries were not found.'
    if no_warning_summary.exists() and main_summary.exists():
        main_core = summarize_core_metrics(pd.read_csv(main_summary), ['f1'], model_order=['shared_encoder_expert_heads'])
        no_warning_core = summarize_core_metrics(pd.read_csv(no_warning_summary), ['f1'], model_order=['shared_encoder_no_warning_task'])
        if not main_core.empty and not no_warning_core.empty:
            main_f1 = float(main_core.iloc[0]['f1'])
            no_warning_f1 = float(no_warning_core.iloc[0]['f1'])
            benefit_note = (
                f'- Dual-task benefit check: main core identification F1={main_f1:.6f}, '
                f'no_warning_task core identification F1={no_warning_f1:.6f}. '
                'Current evidence does not support a stronger joint-training benefit claim.'
            )

    exp6_note = '- Experiment 6 remains deferred: RAW_DATA external validation is still outside this round-two formal bundle.'
    case_note = '- Case analysis was exported for one success case and one failure case.' if not case_summary.empty else '- Case analysis export did not find usable cases.'

    lines = [
        '# PLAN Alignment Supplement',
        '',
        '## Added Without Re-running',
        '',
        '- Added an Experiment 2 multifault summary table with all-fault and core-fault Macro-F1 / Micro-F1.',
        '- Added cross-model comparison figures for core recognition and core warning tasks.',
        case_note,
        '',
        '## Current Reading Against PLAN',
        '',
        recognition_best,
        warning_best,
        macro_best,
        micro_best,
        benefit_note,
        exp6_note,
        '',
        '## Recommendation',
        '',
        '- Keep the current results as the formal Data_set full-data baseline package.',
        '- Add one targeted follow-up experiment on the proposed model by reducing warning-task coupling strength before making a stronger dual-task benefit claim.',
    ]
    return '\n'.join(lines) + '\n'


def _best_model_line(frame: pd.DataFrame, metric_name: str) -> str:
    metric_values = pd.to_numeric(frame.get(metric_name), errors='coerce')
    valid = frame.loc[metric_values.notna()].copy()
    if valid.empty:
        return f'- No valid rows were found for {metric_name}.'
    valid[metric_name] = pd.to_numeric(valid[metric_name], errors='coerce')
    best_row = valid.sort_values(metric_name, ascending=False).iloc[0]
    return f'- Best {metric_name}: {best_row["model_name"]} ({float(best_row[metric_name]):.6f}).'


if __name__ == '__main__':
    main()
