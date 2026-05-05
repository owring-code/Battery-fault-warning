import pandas as pd

from battery_thesis.results import aggregate_round1_tables, build_summary_row, write_result_bundle


def test_build_summary_row_emits_consistent_metric_columns(tmp_path):
    summary_row = build_summary_row(
        experiment_name='recognition_main',
        model_name='lightgbm',
        task_type='identification',
        fault_type='sd',
        split='test',
        metrics={'f1': 0.8, 'recall': 0.9, 'precision': 0.7},
    )

    assert summary_row['experiment_name'] == 'recognition_main'
    assert summary_row['f1'] == 0.8
    assert 'warning_f1' in summary_row


def test_write_result_bundle_writes_summary_predictions_and_data_points(tmp_path):
    output_dir = tmp_path / 'recognition'
    summary = pd.DataFrame([{'experiment_name': 'e', 'model_name': 'm'}])
    predictions = pd.DataFrame([{'sample_id': 's1', 'y_true': 1, 'y_pred': 1, 'y_score': 0.8}])
    data_points = pd.DataFrame([{'plot_type': 'bar_metric', 'x': 'sd', 'y': 0.8}])

    paths = write_result_bundle(output_dir, summary, predictions, data_points, notes='ok')

    assert paths['summary'].exists()
    assert paths['predictions'].exists()
    assert paths['data_points'].exists()
    assert paths['notes'].exists()


def test_aggregate_round1_tables_collects_core_outputs_and_rare_faults(tmp_path):
    results_root = tmp_path / 'results'

    (results_root / 'recognition' / 'lightgbm').mkdir(parents=True)
    (results_root / 'recognition' / 'main_dual_task').mkdir(parents=True)
    (results_root / 'warning' / 'main_dual_task').mkdir(parents=True)
    (results_root / 'ablation' / 'no_expert_heads').mkdir(parents=True)

    pd.DataFrame([
        {'experiment_name': 'e1', 'model_name': 'lightgbm', 'task_type': 'identification', 'fault_type': 'sd', 'split': 'test'},
        {'experiment_name': 'e1', 'model_name': 'lightgbm', 'task_type': 'identification', 'fault_type': 'isc', 'split': 'test'},
    ]).to_csv(results_root / 'recognition' / 'lightgbm' / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([
        {'experiment_name': 'e2', 'model_name': 'main', 'task_type': 'identification', 'fault_type': 'samp', 'split': 'test'},
        {'experiment_name': 'e2', 'model_name': 'main', 'task_type': 'identification', 'fault_type': 'conn', 'split': 'test'},
    ]).to_csv(results_root / 'recognition' / 'main_dual_task' / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([
        {'experiment_name': 'e3', 'model_name': 'main', 'task_type': 'warning', 'fault_type': 'sd', 'split': 'test'},
        {'experiment_name': 'e3', 'model_name': 'main', 'task_type': 'warning', 'fault_type': 'ins', 'split': 'test'},
    ]).to_csv(results_root / 'warning' / 'main_dual_task' / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([
        {'experiment_name': 'e4', 'model_name': 'abl', 'task_type': 'identification', 'fault_type': 'sd', 'split': 'test'},
        {'experiment_name': 'e4', 'model_name': 'abl', 'task_type': 'identification', 'fault_type': 'isc', 'split': 'test'},
    ]).to_csv(results_root / 'ablation' / 'no_expert_heads' / 'summary.csv', index=False, encoding='utf-8-sig')

    acceptance = {
        'core_passed': True,
        'rare_fault_notes': ['isc retained as rare fault', 'conn retained as rare fault'],
        'data_limitations': ['conn only has one vehicle'],
        'can_enter_round2': True,
    }
    outputs = aggregate_round1_tables(results_root, acceptance=acceptance)

    assert outputs['recognition'].exists()
    assert outputs['warning'].exists()
    assert outputs['ablation'].exists()
    assert outputs['rare_recognition'].exists()
    assert outputs['acceptance_report'].exists()

    recognition_df = pd.read_csv(outputs['recognition'])
    warning_df = pd.read_csv(outputs['warning'])
    ablation_df = pd.read_csv(outputs['ablation'])
    rare_df = pd.read_csv(outputs['rare_recognition'])
    report_text = outputs['acceptance_report'].read_text(encoding='utf-8-sig')

    assert set(recognition_df['fault_type']) == {'sd', 'samp'}
    assert set(warning_df['fault_type']) == {'sd', 'ins'}
    assert set(ablation_df['fault_type']) == {'sd'}
    assert set(rare_df['fault_type']) == {'isc', 'conn'}
    assert '第1轮验收报告' in report_text
    assert '核心任务说明' in report_text
    assert 'conn only has one vehicle' in report_text


def test_aggregate_round1_tables_collects_warning_baselines(tmp_path):
    results_root = tmp_path / 'results'
    for model_name in ['threshold_trend', 'lightgbm', 'lstm', 'transformer', 'main_dual_task']:
        (results_root / 'warning' / model_name).mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    'experiment_name': 'warning_main',
                    'model_name': model_name,
                    'task_type': 'warning',
                    'fault_type': 'sd',
                    'split': 'test',
                    'warning_f1': 0.6,
                },
                {
                    'experiment_name': 'warning_main',
                    'model_name': model_name,
                    'task_type': 'warning',
                    'fault_type': 'isc',
                    'split': 'test',
                    'warning_f1': 0.1,
                },
            ]
        ).to_csv(results_root / 'warning' / model_name / 'summary.csv', index=False, encoding='utf-8-sig')

    outputs = aggregate_round1_tables(results_root)
    warning_df = pd.read_csv(outputs['warning'])

    assert set(warning_df['model_name']) == {'threshold_trend', 'lightgbm', 'lstm', 'transformer', 'main_dual_task'}
    assert set(warning_df['fault_type']) == {'sd'}
