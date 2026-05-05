from pathlib import Path
import subprocess
import re
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SAMPLES_ROOT = PROJECT_ROOT / "artifacts" / "round2" / "samples"


pytestmark = pytest.mark.skipif(
    not (SAMPLES_ROOT / "full_scale_summary.csv").exists(),
    reason="requires generated round2 sample artifacts, which are not committed",
)

def _contains_cjk(text: str) -> bool:
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def test_export_chapter5_simulated_supplements_builds_figures_and_draft(tmp_path: Path):
    results_root = tmp_path / 'round2_simulated_normal'
    draft_path = tmp_path / 'chapter5_experiment_simulated_draft.md'

    build_result = subprocess.run(
        [
            PYTHON,
            'scripts/build_simulated_round2_results.py',
            '--results-root',
            str(results_root),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build_result.returncode == 0, build_result.stderr

    export_result = subprocess.run(
        [
            PYTHON,
            'scripts/export_chapter5_simulated_supplements.py',
            '--results-root',
            str(results_root),
            '--draft-path',
            str(draft_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert export_result.returncode == 0, export_result.stderr
    chapter5_root = results_root / 'figures' / 'chapter5'
    expected_files = [
        'fig5_1_label_distribution.png',
        'fig5_1_label_distribution_data.csv',
        'fig5_3_recognition_confusion_heatmap.png',
        'fig5_3_recognition_confusion_heatmap_data.csv',
        'fig5_4_core_pr_curve.png',
        'fig5_4_core_pr_curve_data.csv',
        'fig5_5_core_roc_curve.png',
        'fig5_5_core_roc_curve_data.csv',
        'fig5_6_warning_recall_far_tradeoff.png',
        'fig5_6_warning_recall_far_tradeoff_data.csv',
        'fig5_7_warning_lead_time_distribution.png',
        'fig5_7_warning_lead_time_distribution_data.csv',
        'fig5_8_ablation_comparison.png',
        'fig5_8_ablation_comparison_data.csv',
        'table5_1_experiment_settings.csv',
        'table5_4_ablation_comparison.csv',
        'table5_8_ablation_comparison.csv',
        'table5_7_warning_core_fault_comparison.csv',
        'table5_6_warning_all_fault_comparison.csv',
        'table5_5_recognition_core_fault_comparison.csv',
        'table5_4_recognition_all_fault_comparison.csv',
    ]
    for name in expected_files:
        assert (chapter5_root / name).exists(), name

    confusion = pd.read_csv(chapter5_root / 'fig5_3_recognition_confusion_heatmap_data.csv')
    assert set(confusion['fault_type']) == {'sd', 'isc', 'conn', 'samp', 'ins'}
    assert {'TPR', 'FNR', 'FPR', 'TNR'}.issubset(set(confusion['metric']))

    svg_files = list((results_root / 'figures').rglob('*.svg'))
    assert svg_files, 'no svg figures exported'
    cjk_svg_files = [str(path.relative_to(results_root)) for path in svg_files if _contains_cjk(path.read_text(encoding='utf-8'))]
    assert cjk_svg_files == []
    label_svg = (chapter5_root / 'fig5_1_label_distribution.svg').read_text(encoding='utf-8')
    assert 'Self-discharge' in label_svg
    assert 'Sudden ISC' in label_svg
    pr_curve = pd.read_csv(chapter5_root / 'fig5_4_core_pr_curve_data.csv')
    assert {'model_name', 'recall', 'precision'}.issubset(pr_curve.columns)
    assert pr_curve['model_name'].nunique() >= 4
    assert pr_curve['precision'].max() < 1.0

    recognition_core = pd.read_csv(results_root / 'tables' / 'recognition_model_comparison_core.csv')
    warning_core = pd.read_csv(results_root / 'tables' / 'warning_model_comparison_core.csv')
    recognition_all = pd.read_csv(results_root / 'tables' / 'recognition_model_comparison_all.csv')
    warning_all = pd.read_csv(results_root / 'tables' / 'warning_model_comparison_all.csv')
    ablation = pd.read_csv(chapter5_root / 'table5_4_ablation_comparison.csv')
    proposed_rec = float(recognition_core.loc[recognition_core['model_name'] == 'Proposed Method', 'mean_core_f1'].iloc[0])
    proposed_warn = float(warning_core.loc[warning_core['model_name'] == 'Proposed Method', 'mean_core_warning_f1'].iloc[0])
    proposed_all_rec = float(recognition_all.loc[recognition_all['model_name'] == 'Proposed Method', 'mean_all_f1'].iloc[0])
    proposed_all_warn = float(warning_all.loc[warning_all['model_name'] == 'Proposed Method', 'mean_all_warning_f1'].iloc[0])
    assert proposed_all_rec < proposed_rec
    assert proposed_all_warn < proposed_warn
    assert (chapter5_root / 'table5_4_recognition_all_fault_comparison.csv').exists()
    assert (chapter5_root / 'table5_5_recognition_core_fault_comparison.csv').exists()
    assert (chapter5_root / 'table5_6_warning_all_fault_comparison.csv').exists()
    assert (chapter5_root / 'table5_7_warning_core_fault_comparison.csv').exists()
    full_model = ablation.loc[ablation['模型变体'] == 'Full model'].iloc[0]
    # The ablation Full model row should use the all-fault scope requested for ablation analysis.
    assert float(full_model['ID-F1']) == proposed_all_rec
    assert float(full_model['Warn-F1']) == proposed_all_warn
    draft_text = draft_path.read_text(encoding='utf-8')
    assert '第5章 实验与结果分析' in draft_text
    assert '模拟结果占位' in draft_text
    assert '图5-3' in draft_text
    assert '表5-1 实验参数设置' in draft_text
    assert '图5-8' in draft_text



def test_warning_tradeoff_axis_keeps_high_recall_points_visible():
    from scripts.export_chapter5_simulated_supplements import _warning_tradeoff_axis_limits

    frame = pd.DataFrame(
        [
            {'mean_core_false_alarm_rate': 0.07, 'mean_core_warning_recall': 0.50},
            {'mean_core_false_alarm_rate': 0.03, 'mean_core_warning_recall': 0.872},
        ]
    )

    _, ylim = _warning_tradeoff_axis_limits(frame)

    assert ylim[1] > float(frame['mean_core_warning_recall'].max())
