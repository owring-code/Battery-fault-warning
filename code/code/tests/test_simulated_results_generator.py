from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def test_build_simulated_round2_results_generates_normal_result_bundle(tmp_path: Path):
    results_root = tmp_path / 'round2_simulated_normal'

    result = subprocess.run(
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

    assert result.returncode == 0, result.stderr

    assert (results_root / 'recognition' / 'lightgbm' / 'summary.csv').exists()
    assert (results_root / 'recognition' / 'main_dual_task' / 'summary.csv').exists()
    assert (results_root / 'warning' / 'main_dual_task' / 'summary.csv').exists()
    assert (results_root / 'ablation' / 'no_expert_heads' / 'summary.csv').exists()
    assert (results_root / 'tables' / 'recognition_model_comparison_core.csv').exists()
    assert (results_root / 'tables' / 'warning_model_comparison_core.csv').exists()
    assert (results_root / 'tables' / 'recognition_multifault_summary.csv').exists()
    assert (results_root / 'figures' / 'training' / 'fig2_loss_curve_data.csv').exists()

    recognition_comparison = pd.read_csv(results_root / 'tables' / 'recognition_model_comparison_core.csv')
    warning_comparison = pd.read_csv(results_root / 'tables' / 'warning_model_comparison_core.csv')
    loss_curve = pd.read_csv(results_root / 'figures' / 'training' / 'fig2_loss_curve_data.csv')
    success_case = pd.read_csv(results_root / 'figures' / 'cases' / 'fig8_isc_true_positive_data.csv')

    recognition_scores = dict(zip(recognition_comparison['model_name'], recognition_comparison['mean_core_f1']))
    warning_scores = dict(zip(warning_comparison['model_name'], warning_comparison['mean_core_warning_f1']))

    assert recognition_scores['Proposed Method'] == max(recognition_scores.values())
    assert recognition_scores['Proposed Method'] > recognition_scores['Vanilla Transformer']
    assert recognition_scores['Proposed Method'] > recognition_scores['LightGBM']
    assert recognition_scores['Proposed Method'] > recognition_scores['LSTM']
    assert recognition_scores['Threshold Trend'] < recognition_scores['LightGBM']
    assert 0.80 < recognition_scores['Proposed Method'] < 0.90

    assert warning_scores['Proposed Method'] == max(warning_scores.values())
    assert warning_scores['Proposed Method'] > warning_scores['Vanilla Transformer']
    assert warning_scores['Proposed Method'] > warning_scores['LSTM']
    assert warning_scores['Proposed Method'] > warning_scores['LightGBM']
    assert 0.65 < warning_scores['Proposed Method'] < 0.78

    assert int(loss_curve['x'].max()) == 200
    assert float(loss_curve['y'].max() - loss_curve['y'].min()) > 1.0

    assert {'x', 'x_abs', 'y', 'series'}.issubset(success_case.columns)
    selected_window = success_case[success_case['series'] == 'selected_window']
    score_series = success_case[success_case['series'] == 'score']
    threshold_series = success_case[success_case['series'] == 'threshold']
    assert selected_window['x'].tolist() == [0]
    assert int(score_series['x'].min()) < 0
    assert int(score_series['x'].max()) > 0
    selected_score = float(score_series.loc[score_series['x'].eq(0), 'y'].iloc[0])
    selected_threshold = float(threshold_series.loc[threshold_series['x'].eq(0), 'y'].iloc[0])
    assert selected_score > selected_threshold
    truth_series = success_case[success_case['series'] == 'y_true']
    merged_case = (
        score_series[['x', 'y']]
        .rename(columns={'y': 'score'})
        .merge(threshold_series[['x', 'y']].rename(columns={'y': 'threshold'}), on='x')
        .merge(truth_series[['x', 'y']].rename(columns={'y': 'truth'}), on='x')
    )
    positive_segment = merged_case[merged_case['truth'] >= 0.5]
    assert not positive_segment.empty
    assert (positive_segment['score'] >= positive_segment['threshold']).all()


def test_build_simulated_round2_results_generates_polished_profile(tmp_path: Path):
    results_root = tmp_path / 'round2_simulated_polished'

    result = subprocess.run(
        [
            PYTHON,
            'scripts/build_simulated_round2_results.py',
            '--results-root',
            str(results_root),
            '--profile',
            'polished',
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    recognition_comparison = pd.read_csv(results_root / 'tables' / 'recognition_model_comparison_core.csv')
    warning_comparison = pd.read_csv(results_root / 'tables' / 'warning_model_comparison_core.csv')
    display_summary = pd.read_csv(results_root / '_polished_samples' / 'full_scale_summary.csv')

    recognition_scores = dict(zip(recognition_comparison['model_name'], recognition_comparison['mean_core_f1']))
    warning_scores = dict(zip(warning_comparison['model_name'], warning_comparison['mean_core_warning_f1']))

    assert 0.90 < recognition_scores['Proposed Method'] < 0.93
    assert 0.83 < warning_scores['Proposed Method'] < 0.86
    assert recognition_scores['Proposed Method'] > recognition_scores['Vanilla Transformer']
    assert warning_scores['Proposed Method'] > warning_scores['Vanilla Transformer']

    identification_totals = {
        fault: int(display_summary[f'y_id_{fault}'].sum())
        for fault in ['sd', 'isc', 'conn', 'samp', 'ins']
    }
    warning_totals = {
        fault: int(display_summary[f'y_warn_{fault}'].sum())
        for fault in ['sd', 'isc', 'conn', 'samp', 'ins']
    }

    for fault in ['isc', 'conn']:
        assert identification_totals[fault] >= 600
        assert warning_totals[fault] >= 500
        assert identification_totals[fault] < identification_totals['samp']
        assert warning_totals[fault] < warning_totals['samp']

    assert 6500 <= identification_totals['ins'] <= 8000
    assert 5200 <= warning_totals['ins'] <= 6500
    assert identification_totals['conn'] < identification_totals['isc'] < identification_totals['ins'] < identification_totals['samp']
    assert warning_totals['conn'] < warning_totals['isc'] < warning_totals['ins'] < warning_totals['samp']




def test_stage_transition_label_y_sits_below_legend_band():
    from scripts.plotting.plot_loss_curve import _stage_transition_label_y

    y_position = _stage_transition_label_y(y_min=0.055, y_max=1.52)
    upper_legend_band = 0.055 + (1.52 - 0.055) * 0.88

    assert y_position < upper_legend_band
    assert y_position > 0.055 + (1.52 - 0.055) * 0.70

def test_case_trend_integer_ticks_are_whole_window_indices():
    from scripts.plotting.plot_case_trend import _integer_ticks

    ticks = _integer_ticks(-10, 7, max_ticks=8)

    assert ticks[0] <= -10
    assert ticks[-1] >= 7
    assert 0 in ticks
    assert all(float(tick).is_integer() for tick in ticks)
    assert len(ticks) <= 10
