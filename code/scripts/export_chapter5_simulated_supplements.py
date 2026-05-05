from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.plot_style import apply_academic_plot_style, get_academic_palette, style_axis

FAULT_ORDER = ['sd', 'isc', 'ins', 'samp', 'conn']
CORE_FAULTS = ['sd', 'samp', 'ins']
FAULT_LABELS = {
    'sd': 'Self-discharge',
    'isc': 'Sudden ISC',
    'conn': 'Connection anomaly',
    'samp': 'Sampling anomaly',
    'ins': 'Insulation failure',
}
MODEL_LABELS = {
    'Threshold Trend': 'Threshold Trend',
    'LightGBM': 'LightGBM',
    'LSTM': 'LSTM',
    'Vanilla Transformer': 'Transformer',
    'Proposed Method': 'Proposed Method',
}
MODEL_ORDER = ['Threshold Trend', 'LightGBM', 'LSTM', 'Vanilla Transformer', 'Proposed Method']
RECOGNITION_DIRS = {
    'Threshold Trend': 'threshold_trend',
    'LightGBM': 'lightgbm',
    'LSTM': 'lstm',
    'Vanilla Transformer': 'transformer',
    'Proposed Method': 'main_dual_task',
}
RARE_DISPLAY_POSITIVES = {
    'isc': 180,
    'conn': 120,
    'ins': 150,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export chapter-5 supplement figures and draft from simulated round2 results.')
    parser.add_argument('--results-root', default='{results_root.as_posix()}')
    parser.add_argument('--samples-root', default='artifacts/round2/samples')
    parser.add_argument('--real-results-root', default='results/round2')
    parser.add_argument('--draft-path', default='docs/drafts/chapter5_experiment_simulated_draft.md')
    return parser


def _read_predictions(results_root: Path, task: str, model_dir: str) -> pd.DataFrame:
    frame = pd.read_csv(results_root / task / model_dir / 'predictions.csv')
    frame['model_dir'] = model_dir
    return frame


def _read_summary(results_root: Path, task: str, model_dir: str) -> pd.DataFrame:
    return pd.read_csv(results_root / task / model_dir / 'summary.csv')


def _savefig(fig, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_suffix.with_suffix('.png'))
    fig.savefig(path_no_suffix.with_suffix('.svg'))
    plt.close(fig)


def _load_full_scale_summary(samples_root: Path) -> pd.DataFrame:
    path = samples_root / 'full_scale_summary.csv'
    if not path.exists():
        raise FileNotFoundError(f'full-scale summary not found: {path}')
    return pd.read_csv(path)


def _test_positive_count(full_summary: pd.DataFrame, fault: str, label_kind: str) -> int:
    row = full_summary[full_summary['split'] == 'test'].iloc[0]
    return int(row[f'{label_kind}_{fault}'])


def _test_sample_count(full_summary: pd.DataFrame) -> int:
    return int(full_summary.loc[full_summary['split'] == 'test', 'sample_count'].iloc[0])


def _metric_value(summary: pd.DataFrame, fault: str, column: str, fallback: float = 0.0) -> float:
    row = summary[summary['fault_type'] == fault]
    if row.empty or column not in row.columns:
        return fallback
    value = pd.to_numeric(row[column], errors='coerce').iloc[0]
    return fallback if pd.isna(value) else float(value)


def _export_experiment_settings(results_root: Path, samples_root: Path, chapter5_root: Path) -> pd.DataFrame:
    settings = pd.DataFrame(
        [
            {'参数项': '数据来源', '设置值': 'Data_set 全量结构化样本', '说明': 'RAW_DATA 外部验证暂不纳入本章正式实验结论。'},
            {'参数项': '样本包路径', '设置值': str(samples_root), '说明': '样本数量与标签分布读取自 full_scale_summary.csv。'},
            {'参数项': '数据划分策略', '设置值': '车辆级训练集/验证集/测试集（train/val/test）', '说明': '同一车辆的所有窗口样本仅出现在一个数据划分中，以降低车辆级信息泄漏风险。'},
            {'参数项': '时间窗口长度', '设置值': '30 帧', '说明': '每个时序样本由连续 30 条记录组成。'},
            {'参数项': '输入维度', '设置值': 'X_seq: 30 x 8; X_feat: 28', '说明': '时序分支输入连续传感器特征，专家头同时使用统计特征。'},
            {'参数项': '任务设置', '设置值': '多标签识别 + 多标签预警', '说明': '每个任务均对应五类故障的独立二分类输出。'},
            {'参数项': '故障类型', '设置值': 'sd / isc / ins / samp / conn', '说明': '依次对应自放电异常、突发型内短路、绝缘失效、采样异常和连接异常。'},
            {'参数项': '实验组合', '设置值': '样本统计、识别对比、预警对比、消融实验、案例分析', '说明': '分别用于验证数据可靠性、模型性能、预警实用性、框架模块贡献和典型样本解释能力。'},
            {'参数项': '批大小（batch size）', '设置值': '1024', '说明': '用于 full-data GPU 训练；若显存不足，可在保持其他设置一致的前提下降低该值。'},
            {'参数项': '训练轮数（epoch）', '设置值': '时序基线 50；双任务模型 100 + 100', '说明': '双任务模型先进行识别任务预训练，再进行识别与预警联合优化。'},
            {'参数项': '优化器 / LR', '设置值': 'Adam / 1e-3', '说明': '时序基线与本文神经网络模型采用相同初始学习率。'},
            {'参数项': '预警损失权重', '设置值': 'lambda_warn = 0.3', '说明': '用于平衡识别损失与预警损失。'},
            {'参数项': 'LightGBM 参数', '设置值': 'n_estimators=200, learning_rate=0.05, num_leaves=31', '说明': '五类故障分别训练二分类器，并在统一测试集上评估。'},
            {'参数项': '阈值选择', '设置值': '验证集 F1 搜索', '说明': '识别与预警预测均使用验证集得分选择阈值，而非固定 0.5 阈值。'},
            {'参数项': '正式采样策略', '设置值': '完整 split，不启用 max-* 抽样', '说明': 'max-train/val/test 参数仅用于小样本调试，不用于正式模型比较。'},
            {'参数项': '模拟结果目录', '设置值': str(results_root), '说明': '该目录中的数值用于写作预演和与 results/round2 真实结果对照。'},
        ]
    )
    settings.to_csv(chapter5_root / 'table5_1_experiment_settings.csv', index=False, encoding='utf-8-sig')
    return settings


def _export_label_distribution(samples_root: Path, chapter5_root: Path) -> pd.DataFrame:
    full_summary = _load_full_scale_summary(samples_root)
    rows: list[dict[str, object]] = []
    for fault in FAULT_ORDER:
        rows.append(
            {
                'fault_type': fault,
                'fault_label': FAULT_LABELS[fault],
                'task': 'Identification',
                'positive_count': int(full_summary[f'y_id_{fault}'].sum()),
            }
        )
        rows.append(
            {
                'fault_type': fault,
                'fault_label': FAULT_LABELS[fault],
                'task': 'Warning',
                'positive_count': int(full_summary[f'y_warn_{fault}'].sum()),
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(chapter5_root / 'fig5_1_label_distribution_data.csv', index=False, encoding='utf-8-sig')

    palette = get_academic_palette()
    pivot = data.pivot(index='fault_label', columns='task', values='positive_count').reindex([FAULT_LABELS[f] for f in FAULT_ORDER])
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(pivot.index))
    width = 0.34
    bars1 = ax.bar(x - width / 2, pivot['Identification'], width=width, color=palette['teal'], label='Fault identification', edgecolor='white')
    bars2 = ax.bar(x + width / 2, pivot['Warning'], width=width, color=palette['coral'], label='Fault warning', edgecolor='white')
    style_axis(ax, grid_axis='y')
    positive_values = pivot.to_numpy(dtype=float)
    positive_values = positive_values[positive_values > 0]
    axis_min = max(0.8, float(positive_values.min()) * 0.6) if positive_values.size else 0.8
    ax.set_yscale('log')
    ax.set_ylim(axis_min, float(pivot.to_numpy().max()) * 1.8)
    ax.set_ylabel('Positive samples (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0, ha='center')
    ax.legend(loc='upper right')
    ax.bar_label(bars1, padding=3, fontsize=10)
    ax.bar_label(bars2, padding=3, fontsize=10)
    plt.tight_layout()
    _savefig(fig, chapter5_root / 'fig5_1_label_distribution')
    return data


def _export_confusion_rate_heatmap(results_root: Path, samples_root: Path, chapter5_root: Path) -> pd.DataFrame:
    apply_academic_plot_style()
    per_fault = pd.read_csv(results_root / 'tables' / 'per_fault_type_metrics.csv', encoding='utf-8-sig')
    per_fault = per_fault[per_fault['fault_type'].isin(FAULT_ORDER)].set_index('fault_type')
    rows: list[dict[str, object]] = []
    for fault in FAULT_ORDER:
        id_recall = float(per_fault.loc[fault, 'ID-Recall'])
        false_alarm_rate = float(per_fault.loc[fault, 'FPR'])
        metrics = {
            'TPR': id_recall,
            'FNR': 1.0 - id_recall,
            'FPR': false_alarm_rate,
            'TNR': 1.0 - false_alarm_rate,
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    'fault_type': fault,
                    'fault_label': FAULT_LABELS[fault],
                    'metric': metric,
                    'rate': float(value),
                    'source_id_recall': id_recall,
                    'source_false_alarm_rate': false_alarm_rate,
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(chapter5_root / 'fig5_3_recognition_confusion_heatmap_data.csv', index=False, encoding='utf-8-sig')

    matrix = data.pivot(index='fault_label', columns='metric', values='rate').reindex([FAULT_LABELS[f] for f in FAULT_ORDER])
    matrix = matrix[['TPR', 'FNR', 'FPR', 'TNR']]
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap='Reds', vmin=0.0, vmax=1.0, aspect='auto')
    ax.grid(False)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(['TPR', 'FNR', 'FPR', 'TNR'])
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix.iloc[i, j])
            metric_name = str(matrix.columns[j])
            if metric_name in {'FPR', 'TNR'}:
                label = f'{value:.4f}' if value < 0.1 or value > 0.9 else f'{value:.3f}'
            else:
                label = f'{value:.3f}'
            color = 'white' if value > 0.55 else get_academic_palette()['ink']
            ax.text(j, i, label, ha='center', va='center', color=color, fontsize=11)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Rate')
    plt.tight_layout()
    _savefig(fig, chapter5_root / 'fig5_3_recognition_confusion_heatmap')
    return data


def _smooth_pr_curve(base_precision: float, strength: float, recall_grid: np.ndarray, model_name: str) -> np.ndarray:
    start = min(0.97, base_precision + 0.10 + strength * 0.05)
    tail = 0.08 + strength * 0.24
    precision = tail + (start - tail) * np.power(1.0 - recall_grid, 0.55 + strength * 0.20)
    precision -= (0.02 + 0.04 * (1.0 - strength)) * np.clip((recall_grid - 0.76) / 0.24, 0.0, 1.0) ** 1.6
    if model_name == 'LightGBM':
        precision += 0.014 * np.exp(-((recall_grid - 0.42) / 0.20) ** 2)
    if model_name == 'LSTM':
        precision += 0.020 * np.exp(-((recall_grid - 0.64) / 0.18) ** 2)
    if model_name == 'Proposed Method':
        precision += 0.026 * np.exp(-((recall_grid - 0.58) / 0.32) ** 2)
    return np.clip(precision, 0.04, 0.98)


def _export_core_pr_curve(results_root: Path, chapter5_root: Path) -> pd.DataFrame:
    comparison = pd.read_csv(results_root / 'tables' / 'recognition_model_comparison_core.csv')
    comparison = comparison[comparison['model_name'].isin(MODEL_ORDER)].copy()
    comparison['sort_key'] = comparison['model_name'].map({name: i for i, name in enumerate(MODEL_ORDER)})
    comparison = comparison.sort_values('sort_key')
    recall_grid = np.linspace(0.02, 1.0, 160)
    palette = get_academic_palette()
    colors = {
        'Threshold Trend': palette['slate'],
        'LightGBM': palette['sage'],
        'LSTM': palette['coral_light'],
        'Vanilla Transformer': palette['coral'],
        'Proposed Method': palette['teal'],
    }
    markers = {
        'Threshold Trend': 'o',
        'LightGBM': 's',
        'LSTM': '^',
        'Vanilla Transformer': 'D',
        'Proposed Method': 'P',
    }
    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    min_f1 = float(comparison['mean_core_f1'].min())
    max_f1 = float(comparison['mean_core_f1'].max())
    for _, row in comparison.iterrows():
        model_name = str(row['model_name'])
        strength = (float(row['mean_core_f1']) - min_f1) / max(max_f1 - min_f1, 1e-6)
        precision = _smooth_pr_curve(float(row['mean_core_precision']), strength, recall_grid, model_name)
        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(recall_grid, precision, label=label, color=colors.get(model_name, palette['teal']), linewidth=2.3, marker=markers.get(model_name, 'o'), markevery=18, markersize=5.2, markerfacecolor='white', markeredgewidth=1.1)
        for recall_value, precision_value in zip(recall_grid, precision):
            rows.append({'model_name': label, 'recall': float(recall_value), 'precision': float(precision_value)})
    data = pd.DataFrame(rows)
    data.to_csv(chapter5_root / 'fig5_4_core_pr_curve_data.csv', index=False, encoding='utf-8-sig')
    style_axis(ax, grid_axis='both')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.05, 1.0)
    ax.legend(loc='lower left')
    plt.tight_layout()
    _savefig(fig, chapter5_root / 'fig5_4_core_pr_curve')
    return data


def _smooth_roc_curve(target_auc: float, strength: float, fpr_grid: np.ndarray, model_name: str) -> np.ndarray:
    auc = float(np.clip(target_auc, 0.55, 0.995))
    beta = max(0.015, (1.0 / auc) - 1.0)
    tpr = np.power(fpr_grid, beta)
    tpr = 0.985 * tpr + 0.015 * fpr_grid
    if model_name == 'LightGBM':
        tpr += 0.010 * np.exp(-((fpr_grid - 0.22) / 0.16) ** 2)
    if model_name == 'LSTM':
        tpr += 0.012 * np.exp(-((fpr_grid - 0.34) / 0.18) ** 2)
    if model_name == 'Proposed Method':
        tpr += 0.014 * np.exp(-((fpr_grid - 0.18) / 0.22) ** 2)
    tpr = np.maximum.accumulate(np.clip(tpr, 0.0, 1.0))
    tpr[0] = 0.0
    tpr[-1] = 1.0
    return tpr


def _export_core_roc_curve(results_root: Path, chapter5_root: Path) -> pd.DataFrame:
    comparison = pd.read_csv(results_root / 'tables' / 'recognition_model_comparison_core.csv')
    comparison = comparison[comparison['model_name'].isin(MODEL_ORDER)].copy()
    comparison['sort_key'] = comparison['model_name'].map({name: i for i, name in enumerate(MODEL_ORDER)})
    comparison = comparison.sort_values('sort_key')
    fpr_grid = np.linspace(0.0, 1.0, 160)
    palette = get_academic_palette()
    colors = {
        'Threshold Trend': palette['slate'],
        'LightGBM': palette['sage'],
        'LSTM': palette['coral_light'],
        'Vanilla Transformer': palette['coral'],
        'Proposed Method': palette['teal'],
    }
    markers = {
        'Threshold Trend': 'o',
        'LightGBM': 's',
        'LSTM': '^',
        'Vanilla Transformer': 'D',
        'Proposed Method': 'P',
    }
    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    min_auc = float(comparison['mean_core_roc_auc'].min())
    max_auc = float(comparison['mean_core_roc_auc'].max())
    for _, row in comparison.iterrows():
        model_name = str(row['model_name'])
        strength = (float(row['mean_core_roc_auc']) - min_auc) / max(max_auc - min_auc, 1e-6)
        tpr = _smooth_roc_curve(float(row['mean_core_roc_auc']), strength, fpr_grid, model_name)
        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(fpr_grid, tpr, label=label, color=colors.get(model_name, palette['teal']), linewidth=2.3, marker=markers.get(model_name, 'o'), markevery=20, markersize=5.2, markerfacecolor='white', markeredgewidth=1.1)
        for fpr_value, tpr_value in zip(fpr_grid, tpr):
            rows.append({'model_name': label, 'fpr': float(fpr_value), 'tpr': float(tpr_value)})
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1.2, color=palette['slate'], alpha=0.55)
    data = pd.DataFrame(rows)
    data.to_csv(chapter5_root / 'fig5_5_core_roc_curve_data.csv', index=False, encoding='utf-8-sig')
    style_axis(ax, grid_axis='both')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc='lower right')
    plt.tight_layout()
    _savefig(fig, chapter5_root / 'fig5_5_core_roc_curve')
    return data



def _warning_tradeoff_axis_limits(data: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    false_alarm_col = 'mean_all_false_alarm_rate' if 'mean_all_false_alarm_rate' in data.columns else 'mean_core_false_alarm_rate'
    recall_col = 'mean_all_warning_recall' if 'mean_all_warning_recall' in data.columns else 'mean_core_warning_recall'
    false_alarm = pd.to_numeric(data[false_alarm_col], errors='coerce').dropna()
    recall = pd.to_numeric(data[recall_col], errors='coerce').dropna()
    if false_alarm.empty or recall.empty:
        return (0.0, 0.08), (0.30, 0.90)
    xlim = (0.0, max(0.08, float(false_alarm.max()) + 0.014))
    ylim = (
        max(0.30, float(recall.min()) - 0.08),
        min(0.98, max(0.82, float(recall.max()) + 0.06)),
    )
    return xlim, ylim
def _export_warning_tradeoff(results_root: Path, chapter5_root: Path) -> pd.DataFrame:
    apply_academic_plot_style()
    data = pd.read_csv(results_root / 'tables' / 'warning_model_comparison_all.csv')
    data = data.copy()
    data['display_name'] = data['model_name'].map(lambda value: MODEL_LABELS.get(str(value), str(value)))
    data.to_csv(chapter5_root / 'fig5_6_warning_recall_far_tradeoff_data.csv', index=False, encoding='utf-8-sig')

    palette = get_academic_palette()
    color_map = {
        'Threshold Trend': palette['slate'],
        'LightGBM': palette['sage'],
        'LSTM': palette['coral_light'],
        'Vanilla Transformer': palette['coral'],
        'Proposed Method': palette['teal'],
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    sizes = 3600 * (pd.to_numeric(data['mean_all_warning_f1']).to_numpy(dtype=float) + 0.12)
    for idx, row in data.iterrows():
        model_name = str(row['model_name'])
        ax.scatter(
            float(row['mean_all_false_alarm_rate']),
            float(row['mean_all_warning_recall']),
            s=float(sizes[idx]),
            color=color_map.get(model_name, palette['teal']),
            marker='o',
            edgecolor='white',
            linewidth=1.6,
            alpha=0.82,
        )
    style_axis(ax, grid_axis='both')
    ax.set_xlabel('Mean FPR (five faults)')
    ax.set_ylabel('Mean Warning Recall (five faults)')
    xlim, ylim = _warning_tradeoff_axis_limits(data)
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            linestyle='none',
            label=str(row['display_name']),
            markerfacecolor=color_map.get(str(row['model_name']), palette['teal']),
            markeredgecolor='white',
            markeredgewidth=1.1,
            markersize=11,
            alpha=0.9,
        )
        for _, row in data.iterrows()
    ]
    ax.legend(handles=handles, loc='lower left', frameon=True, ncol=1)
    plt.tight_layout()
    _savefig(fig, chapter5_root / 'fig5_6_warning_recall_far_tradeoff')
    return data

def _export_warning_lead_distribution(results_root: Path, samples_root: Path, chapter5_root: Path) -> pd.DataFrame:
    apply_academic_plot_style()
    rng = np.random.default_rng(16)
    full_summary = _load_full_scale_summary(samples_root)
    test_summary = full_summary[full_summary['split'] == 'test'].iloc[0]
    per_fault = pd.read_csv(results_root / 'tables' / 'per_fault_type_metrics.csv', encoding='utf-8-sig')
    per_fault = per_fault[per_fault['fault_type'].isin(FAULT_ORDER)].set_index('fault_type')
    spread_map = {
        'sd': 8.0,
        'isc': 4.0,
        'ins': 5.0,
        'samp': 7.0,
        'conn': 3.0,
    }
    rows: list[dict[str, object]] = []
    for fault in FAULT_ORDER:
        actual_positive = int(test_summary[f'y_warn_{fault}'])
        warning_recall = float(per_fault.loc[fault, 'Warn-Recall'])
        target_mean = float(per_fault.loc[fault, 'Mean Lead Time'])
        simulated_tp_count = max(1, int(round(actual_positive * warning_recall)))
        z = rng.normal(0.0, 1.0, size=simulated_tp_count)
        z = np.clip(z, -2.35, 2.35)
        z = z - float(z.mean())
        values = target_mean + z * spread_map[fault]
        values = np.maximum(values, 1.0)
        values = values + (target_mean - float(values.mean()))
        values = np.round(values, 3)
        residual = round(target_mean * simulated_tp_count - float(values.sum()), 3)
        values[0] = round(float(values[0]) + residual, 3)
        for index, value in enumerate(values):
            rows.append(
                {
                    'fault_type': fault,
                    'fault_label': FAULT_LABELS[fault],
                    'sample_id': f'fullsim_{fault}_{index:06d}',
                    'vehicle_id': f'FULLSIM_{fault.upper()}',
                    'source_dataset': 'full_scale_simulated_test',
                    'future_first_fault': fault,
                    'lead_time_sec': float(value),
                    'actual_test_positive_count': actual_positive,
                    'calibrated_warning_recall': warning_recall,
                    'true_positive_warning_count': simulated_tp_count,
                    'target_mean_lead_time': target_mean,
                }
            )
    data = pd.DataFrame(rows).sort_values(['fault_type', 'sample_id'])
    data.to_csv(chapter5_root / 'fig5_7_warning_lead_time_distribution_data.csv', index=False, encoding='utf-8-sig')

    ordered = [FAULT_LABELS[f] for f in FAULT_ORDER]
    values = [data.loc[data['fault_label'] == label, 'lead_time_sec'].to_numpy(dtype=float) for label in ordered]
    palette = get_academic_palette()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    box = ax.boxplot(values, tick_labels=ordered, patch_artist=True, widths=0.55, showfliers=False)
    fill_colors = [palette['teal_light'], '#F7B7A3', '#D7C4B5', '#C8B6D9', '#B8D8BA']
    for patch, color in zip(box['boxes'], fill_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(palette['ink'])
        patch.set_alpha(0.86)
    for element in ['whiskers', 'caps', 'medians']:
        for line in box[element]:
            line.set_color(palette['ink'])
            line.set_linewidth(1.35)
    for idx, arr in enumerate(values, start=1):
        sample = arr if len(arr) <= 80 else rng.choice(arr, size=80, replace=False)
        jitter = rng.normal(0.0, 0.038, size=len(sample))
        ax.scatter(np.full(len(sample), idx) + jitter, sample, s=16, color=palette['teal'], alpha=0.42, edgecolor='white', linewidth=0.25)
    style_axis(ax, grid_axis='y')
    ax.set_ylabel('Lead time (s)')
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    plt.tight_layout()
    _savefig(fig, chapter5_root / 'fig5_7_warning_lead_time_distribution')
    return data


def _export_ablation_table(results_root: Path, chapter5_root: Path) -> pd.DataFrame:
    model_specs = [
        ('Full model', results_root / 'recognition' / 'main_dual_task' / 'summary.csv', results_root / 'warning' / 'main_dual_task' / 'summary.csv'),
        ('No fault-specific features', results_root / 'ablation' / 'no_fault_specific_features' / 'summary.csv', None),
        ('No expert heads', results_root / 'ablation' / 'no_expert_heads' / 'summary.csv', None),
        ('No warning task', results_root / 'ablation' / 'no_warning_task' / 'summary.csv', None),
        ('No label-quality control', results_root / 'ablation' / 'no_label_quality_control' / 'summary.csv', None),
    ]
    rows = []
    for name, path, warning_path in model_specs:
        frame = pd.read_csv(path)
        if warning_path is not None:
            warning_frame = pd.read_csv(warning_path)
            rec = frame[(frame['task_type'] == 'identification') & (frame['fault_type'].isin(FAULT_ORDER))]
            warn = warning_frame[(warning_frame['task_type'] == 'warning') & (warning_frame['fault_type'].isin(FAULT_ORDER))]
        else:
            rec = frame[(frame['task_type'] == 'identification') & (frame['fault_type'].isin(FAULT_ORDER))]
            warn = frame[(frame['task_type'] == 'warning') & (frame['fault_type'].isin(FAULT_ORDER))]
        rows.append(
            {
                '模型变体': name,
                'ID-F1': float(pd.to_numeric(rec['f1'], errors='coerce').mean()),
                'ID-Recall': float(pd.to_numeric(rec['recall'], errors='coerce').mean()),
                'Warn-F1': float(pd.to_numeric(warn['warning_f1'], errors='coerce').mean()),
                'Warn-Recall': float(pd.to_numeric(warn['warning_recall'], errors='coerce').mean()),
            }
        )
    data = pd.DataFrame(rows)
    baseline = data.loc[data['模型变体'] == 'Full model', 'ID-F1'].iloc[0]
    data['Delta ID-F1'] = baseline - data['ID-F1']
    data.to_csv(chapter5_root / 'table5_4_ablation_comparison.csv', index=False, encoding='utf-8-sig')
    data.to_csv(chapter5_root / 'table5_8_ablation_comparison.csv', index=False, encoding='utf-8-sig')
    return data




def _export_ablation_figure(results_root: Path, chapter5_root: Path, ablation: pd.DataFrame) -> pd.DataFrame:
    data = ablation[['模型变体', 'ID-F1', 'Warn-F1']].copy()
    data = data.rename(columns={'模型变体': 'variant'})
    long_rows = []
    for _, row in data.iterrows():
        long_rows.append({'variant': row['variant'], 'metric': 'ID-F1', 'value': float(row['ID-F1'])})
        long_rows.append({'variant': row['variant'], 'metric': 'Warn-F1', 'value': float(row['Warn-F1'])})
    long_data = pd.DataFrame(long_rows)
    long_data.to_csv(chapter5_root / 'fig5_8_ablation_comparison_data.csv', index=False, encoding='utf-8-sig')

    palette = get_academic_palette()
    variants = data['variant'].tolist()
    y = np.arange(len(variants))
    height = 0.34
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    bars_id = ax.barh(y - height / 2, data['ID-F1'], height=height, color=palette['teal'], label='ID-F1', edgecolor='white')
    bars_warn = ax.barh(y + height / 2, data['Warn-F1'], height=height, color=palette['coral'], label='Warn-F1', edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(variants)
    ax.invert_yaxis()
    style_axis(ax, grid_axis='x')
    ax.set_xlim(0.0, max(0.9, float(data[['ID-F1', 'Warn-F1']].max().max()) + 0.08))
    ax.legend(loc='lower right')
    for container in [bars_id, bars_warn]:
        labels = [f'{bar.get_width():.3f}' for bar in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=10)
    plt.tight_layout()
    for base in [chapter5_root / 'fig5_8_ablation_comparison', results_root / 'figures' / 'ablation' / 'fig5_ablation_bar']:
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(base.with_suffix('.png'))
        fig.savefig(base.with_suffix('.svg'))
    long_data.to_csv(results_root / 'figures' / 'ablation' / 'fig5_ablation_bar_data.csv', index=False, encoding='utf-8-sig')
    plt.close(fig)
    return long_data

def _export_real_vs_simulated(results_root: Path, real_results_root: Path, chapter5_root: Path) -> pd.DataFrame | None:
    sim_path = results_root / 'tables' / 'recognition_model_comparison_core.csv'
    real_path = real_results_root / 'tables' / 'recognition_model_comparison_core.csv'
    if not sim_path.exists() or not real_path.exists():
        return None
    sim = pd.read_csv(sim_path)[['model_name', 'mean_core_f1']].rename(columns={'mean_core_f1': '模拟ID-F1'})
    real = pd.read_csv(real_path)[['model_name', 'mean_core_f1']].rename(columns={'mean_core_f1': '真实ID-F1'})
    merged = sim.merge(real, on='model_name', how='left')
    merged['差值'] = merged['模拟ID-F1'] - merged['真实ID-F1']
    merged = merged.rename(columns={'model_name': '模型'})
    merged['模型'] = merged['模型'].map(lambda value: MODEL_LABELS.get(str(value), str(value)))
    export_frame = merged.fillna('-')
    export_frame.to_csv(chapter5_root / 'table5_real_vs_simulated_recognition.csv', index=False, encoding='utf-8-sig')
    return export_frame


def _format_markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append('-')
            elif isinstance(value, float) or isinstance(value, np.floating):
                values.append(f'{float(value):.3f}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def _write_chapter5_draft(results_root: Path, samples_root: Path, real_results_root: Path, chapter5_root: Path, draft_path: Path) -> None:
    recognition_core = pd.read_csv(results_root / 'tables' / 'recognition_model_comparison_core.csv')
    recognition_all = pd.read_csv(results_root / 'tables' / 'recognition_model_comparison_all.csv')
    warning_core = pd.read_csv(results_root / 'tables' / 'warning_model_comparison_core.csv')
    warning_all = pd.read_csv(results_root / 'tables' / 'warning_model_comparison_all.csv')
    settings = pd.read_csv(chapter5_root / 'table5_1_experiment_settings.csv')
    ablation = pd.read_csv(chapter5_root / 'table5_8_ablation_comparison.csv')
    real_vs_sim_path = chapter5_root / 'table5_real_vs_simulated_recognition.csv'
    real_vs_sim = pd.read_csv(real_vs_sim_path) if real_vs_sim_path.exists() else pd.DataFrame()
    full_summary = _load_full_scale_summary(samples_root)
    total_samples = int(full_summary['sample_count'].sum())
    train_n = int(full_summary.loc[full_summary['split'] == 'train', 'sample_count'].iloc[0])
    val_n = int(full_summary.loc[full_summary['split'] == 'val', 'sample_count'].iloc[0])
    test_n = int(full_summary.loc[full_summary['split'] == 'test', 'sample_count'].iloc[0])

    rec_core_display = recognition_core.copy()
    rec_core_display['model_name'] = rec_core_display['model_name'].map(lambda v: MODEL_LABELS.get(str(v), str(v)))
    rec_all_display = recognition_all.copy()
    rec_all_display['model_name'] = rec_all_display['model_name'].map(lambda v: MODEL_LABELS.get(str(v), str(v)))
    warn_core_display = warning_core.copy()
    warn_core_display['model_name'] = warn_core_display['model_name'].map(lambda v: MODEL_LABELS.get(str(v), str(v)))
    warn_all_display = warning_all.copy()
    warn_all_display['model_name'] = warn_all_display['model_name'].map(lambda v: MODEL_LABELS.get(str(v), str(v)))
    if not real_vs_sim.empty:
        real_vs_sim = real_vs_sim.copy()
        real_vs_sim['模型'] = real_vs_sim['模型'].map(lambda v: MODEL_LABELS.get(str(v), str(v)))

    proposed_rec = recognition_core[recognition_core['model_name'] == 'Proposed Method'].iloc[0]
    proposed_rec_all = recognition_all[recognition_all['model_name'] == 'Proposed Method'].iloc[0]
    proposed_warn = warning_core[warning_core['model_name'] == 'Proposed Method'].iloc[0]
    proposed_warn_all = warning_all[warning_all['model_name'] == 'Proposed Method'].iloc[0]
    best_baseline_rec = recognition_core[recognition_core['model_name'] != 'Proposed Method'].sort_values('mean_core_f1', ascending=False).iloc[0]
    best_baseline_rec_all = recognition_all[recognition_all['model_name'] != 'Proposed Method'].sort_values('mean_all_f1', ascending=False).iloc[0]
    best_baseline_warn = warning_core[warning_core['model_name'] != 'Proposed Method'].sort_values('mean_core_warning_f1', ascending=False).iloc[0]
    best_baseline_warn_all = warning_all[warning_all['model_name'] != 'Proposed Method'].sort_values('mean_all_warning_f1', ascending=False).iloc[0]

    settings_table = _format_markdown_table(settings, ['参数项', '设置值', '说明'])
    split_display = full_summary[['split', 'sample_count']].copy()
    split_display['split'] = split_display['split'].map({'train': '训练集', 'val': '验证集', 'test': '测试集'}).fillna(split_display['split'])
    split_table = _format_markdown_table(
        split_display.rename(columns={'split': '数据划分', 'sample_count': '样本数'}),
        ['数据划分', '样本数'],
    )
    label_rows = []
    for fault in FAULT_ORDER:
        label_rows.append(
            {
                'Fault': FAULT_LABELS[fault],
                'ID positives': int(full_summary[f'y_id_{fault}'].sum()),
                'Warning positives': int(full_summary[f'y_warn_{fault}'].sum()),
                'Test ID positives': int(full_summary.loc[full_summary['split'] == 'test', f'y_id_{fault}'].iloc[0]),
                'Test warning positives': int(full_summary.loc[full_summary['split'] == 'test', f'y_warn_{fault}'].iloc[0]),
            }
        )
    label_table = _format_markdown_table(pd.DataFrame(label_rows), ['Fault', 'ID positives', 'Warning positives', 'Test ID positives', 'Test warning positives'], ['故障类型', '识别正样本数', '预警正样本数', '测试集识别正样本数', '测试集预警正样本数'])

    rec_core_cols = ['model_name', 'mean_core_accuracy', 'mean_core_f1', 'mean_core_recall', 'mean_core_precision', 'mean_core_pr_auc', 'mean_core_roc_auc']
    rec_all_cols = ['model_name', 'mean_all_accuracy', 'mean_all_f1', 'mean_all_recall', 'mean_all_precision', 'mean_all_pr_auc', 'mean_all_roc_auc']
    rec_headers = ['模型', 'Accuracy', 'F1', 'Recall', 'Precision', 'PR-AUC', 'ROC-AUC']
    if 'mean_core_accuracy' not in rec_core_display.columns:
        rec_core_display['mean_core_accuracy'] = np.nan
    if 'mean_all_accuracy' not in rec_all_display.columns:
        rec_all_display['mean_all_accuracy'] = np.nan
    rec_core_table_frame = rec_core_display[rec_core_cols].copy()
    rec_all_table_frame = rec_all_display[rec_all_cols].copy()
    rec_core_table_frame.to_csv(chapter5_root / 'table5_5_recognition_core_fault_comparison.csv', index=False, encoding='utf-8-sig')
    rec_all_table_frame.to_csv(chapter5_root / 'table5_4_recognition_all_fault_comparison.csv', index=False, encoding='utf-8-sig')
    rec_core_table = _format_markdown_table(rec_core_table_frame, rec_core_cols, rec_headers)
    rec_all_table = _format_markdown_table(rec_all_table_frame, rec_all_cols, rec_headers)
    warn_core_cols = ['model_name', 'mean_core_warning_f1', 'mean_core_warning_recall', 'mean_core_false_alarm_rate', 'mean_core_avg_lead_time']
    warn_all_cols = ['model_name', 'mean_all_warning_f1', 'mean_all_warning_recall', 'mean_all_false_alarm_rate', 'mean_all_avg_lead_time']
    warn_headers = ['模型', 'Warning F1', 'Warning Recall', 'FPR', 'Mean Lead Time/s']
    warn_core_table_frame = warn_core_display[warn_core_cols].copy()
    warn_all_table_frame = warn_all_display[warn_all_cols].copy()
    warn_core_table_frame.to_csv(chapter5_root / 'table5_7_warning_core_fault_comparison.csv', index=False, encoding='utf-8-sig')
    warn_all_table_frame.to_csv(chapter5_root / 'table5_6_warning_all_fault_comparison.csv', index=False, encoding='utf-8-sig')
    warn_core_table = _format_markdown_table(warn_core_table_frame, warn_core_cols, warn_headers)
    warn_all_table = _format_markdown_table(warn_all_table_frame, warn_all_cols, warn_headers)
    ablation_table = _format_markdown_table(
        ablation,
        ['模型变体', 'ID-F1', 'ID-Recall', 'Warn-F1', 'Warn-Recall', 'Delta ID-F1'],
        ['模型变体', 'ID-F1', 'ID-Recall', 'Warn-F1', 'Warn-Recall', 'Delta ID-F1'],
    )
    real_vs_sim_table = ''
    if not real_vs_sim.empty:
        real_vs_sim_table = _format_markdown_table(real_vs_sim, ['模型', '模拟ID-F1', '真实ID-F1', '差值'], ['模型', '模拟 ID-F1', '真实 ID-F1', '差值'])

    draft = f'''# 第5章 实验与结果分析（模拟结果占位稿）

> 说明：本章使用 `{results_root.as_posix()}` 中的模拟结果占位，并将 `{(samples_root / 'full_scale_summary.csv').as_posix()}` 作为样本规模和标签分布依据。该稿件用于预演论文写作结构、图表排版和结果解释方式，不能替代正式实验结论。真实结果目录为 `results/round2`，后续应逐项替换并调整结论强度。

## 5.1 实验设置

本文实验围绕动力电池故障识别与故障预警两个任务展开。识别任务判断当前窗口是否已经进入某类故障状态，预警任务判断当前窗口是否将在未来窗口内进入故障状态。为避免车辆级时间窗口之间的信息泄漏，训练、验证和测试划分均以车辆为单位进行。本章按照实验研究的一般展开逻辑，先给出实验数据、参数设置和评价指标，再分别报告主模型对比、预警结果、消融实验和案例分析。

本章实验设计旨在同时验证模型预测性能与所提出框架的结构优势。首先，针对阈值规则、LightGBM、LSTM、Transformer 以及本文方法等对比模型，在统一样本划分下进行训练与验证，并利用验证集完成阈值选择和关键超参数确定，以尽量保证各模型在接近合理性能的条件下进行比较。其次，所有正式指标均以测试集结果作为最终评估依据，测试集不参与模型训练、阈值搜索和参数选择，从而降低数据泄漏风险，并更接近模型在未知车辆数据上的泛化表现。

围绕上述目标，本文设置了五类实验。第一类为全量样本统计与划分一致性检验，用于确认样本规模、车辆覆盖和五类故障标签分布；第二类为故障识别对比实验，用于比较人工规则、传统机器学习模型和时序深度模型的识别能力；第三类为故障预警对比实验，用于评估不同模型在提前发现故障风险时的 Recall、FPR 和提前量表现；第四类为双任务与消融实验，通过去除故障专属特征、专家头、预警任务和标签质量控制等模块，分析框架中各组成部分对整体性能的贡献；第五类为案例分析与结果稳定性讨论，用于结合典型故障片段解释模型输出变化，并补充总体指标难以反映的局部行为特征。

表5-1给出了本章采用的主要实验参数设置。表中列出了 batch size、epoch、学习率、阈值选择方式和 LightGBM 等关键配置，以保证不同模型在相同数据划分和相近训练条件下进行比较。由于本节数值结果仍为模拟占位，正式定稿时应根据服务器 full-data 实际运行命令对参数进行逐项核对。

表5-1 实验参数设置

{settings_table}

由表5-1可知，本章实验面向 full-data 样本包开展，而非基于随机抽样后的调试子集。`max-*` 参数仅用于小样本调试（smoke test），不参与正式对比结果。各模型的阈值均在验证集上依据 F1 进行选择，从而降低固定 0.5 阈值在类别不平衡场景下造成的少数类 Recall 低估问题。

评价指标方面，识别任务报告 Accuracy、F1、Recall、Precision、PR-AUC 和 ROC-AUC；预警任务报告 Warning F1、Warning Recall、FPR 和 Mean Lead Time。Accuracy 只作为辅助指标保留，因为本数据集中负样本占比很高，单独依赖 Accuracy 容易得到过于乐观的结论。后续分析主要围绕 F1、Recall、PR-AUC 和 FPR 展开。

## 5.2 全量样本统计与实验数据划分验证

本节样本规模统计依据来自处理后的全量样本包。全量样本总数为 {total_samples:,}，其中训练集 {train_n:,}，验证集 {val_n:,}，测试集 {test_n:,}。数据划分采用车辆级切分方式，避免同一车辆的窗口样本同时出现在训练集和测试集。表5-2给出了样本划分规模。

表5-2 全量样本划分规模

{split_table}

表5-2说明本轮实验的数据规模已经达到百万级窗口样本。训练集样本约占全量样本的一半以上，验证集和测试集也保持了较大的窗口数量，这有利于稳定评估模型在常见故障上的表现。但是，窗口数量充足并不等于每类故障样本都充足，因此还需要进一步检查五类故障的正样本分布。

表5-3给出了五类故障在全量样本和测试集中的正样本数量。该表是解释后续结果波动的基础：自放电异常和采样异常正样本较多，而突发型内短路、绝缘失效和连接异常明显稀缺。尤其是连接异常在识别标签中几乎没有稳定测试正例，因此该类指标即使在模拟结果中展示，也不应被过度解释为模型已经充分掌握该类故障机理。

表5-3 五类故障标签统计

{label_table}

图5-1进一步以柱状图展示识别标签和预警标签的正样本数量。图中横坐标为五类故障，纵坐标为正样本数。可以看到，自放电异常在识别任务中占据主要比例，采样异常在识别和预警任务中都具有较高覆盖；突发型内短路、连接异常和绝缘失效则明显处于长尾区域。这个分布决定了本文后续不能只看平均指标，还需要保留对稀缺故障的单独讨论。

图5-1 全量样本五类故障标签分布。对应文件：`{results_root.as_posix()}/figures/chapter5/fig5_1_label_distribution.png`。

## 5.3 故障识别实验结果与分析

本节比较阈值规则、LightGBM、LSTM、Transformer 和本文方法在故障识别任务上的表现。阈值规则基线体现人工规则的可解释性，LightGBM 体现统计特征建模能力，LSTM 和 Transformer 体现时序模型能力，本文方法则在共享时序编码基础上加入故障专属特征和专家头。考虑到突发型内短路和连接异常样本相对稀缺，本节将五类故障整体平均和核心三类故障平均分开报告：前者反映完整任务覆盖能力，后者反映样本更稳定故障上的主要比较结果。

表5-4 五类故障识别模型对比（模拟结果占位）

{rec_all_table}

从表5-4可以看出，本文方法在五类故障整体平均 F1 上达到 {float(proposed_rec_all['mean_all_f1']):.3f}，高于最强基线 {MODEL_LABELS.get(str(best_baseline_rec_all['model_name']), str(best_baseline_rec_all['model_name']))} 的 {float(best_baseline_rec_all['mean_all_f1']):.3f}。由于该口径包含 `isc` 和 `conn` 两类稀缺故障，其数值通常低于核心三类均值，因此更适合用于说明模型在完整五类任务上的总体覆盖能力，而不宜单独作为模型优劣的唯一依据。

表5-5 核心三类故障识别模型对比（模拟结果占位）

{rec_core_table}

从表5-5可以看出，本文方法的核心三类故障平均 F1 为 {float(proposed_rec['mean_core_f1']):.3f}，高于最强基线 {MODEL_LABELS.get(str(best_baseline_rec['model_name']), str(best_baseline_rec['model_name']))} 的 {float(best_baseline_rec['mean_core_f1']):.3f}。这种差距在模拟结果中体现为共享时序建模和故障专属建模的增益。需要注意的是，LightGBM 与 LSTM、Transformer 之间并非严格单调排序：在部分统计特征强、时序依赖较弱的故障上，LightGBM 可能优于 LSTM；在需要捕捉窗口内动态变化时，Transformer 往往更有优势。这种非严格排序比“所有深度模型必然优于传统模型”的叙述更接近真实实验情况。

图5-2将核心三类故障平均 F1 和 Recall 以柱状图形式展示。图中可以观察到，本文方法不仅提高 F1，同时保持较高 Recall，说明它不是简单通过提高阈值来换取更高 Precision。对于故障识别任务而言，Recall 的意义很重要，因为漏报故障在实际车辆安全监测中通常比少量误报更难接受。但如果后续真实结果中 Recall 提升伴随 FPR 明显恶化，则需要重新审视阈值选择和标签噪声问题。

图5-2 核心故障识别模型对比。对应文件：`{results_root.as_posix()}/figures/comparison/fig6_recognition_model_comparison.png`。

图5-3给出了五类故障识别任务的混淆率热力图。该图使用 TPR、FNR、FPR 和 TNR，而不是直接绘制 TP、FP、FN、TN 原始数量，原因是本任务负样本数量远大于正样本数量，原始 TN 会压制其他信息。图中保留五类故障，其中自放电异常和采样异常的 TPR 较高，突发型内短路、连接异常和绝缘失效的 FNR 更值得关注。这里的稀缺故障展示采用了显示层面的平滑处理，数据文件中保留了实际测试正样本数，避免把可视化效果误当作真实样本充足。

图5-3 五类故障识别混淆率热力图。对应文件：`{results_root.as_posix()}/figures/chapter5/fig5_3_recognition_confusion_heatmap.png`。

图5-4给出了核心故障识别任务的 PR 曲线。与 ROC 曲线相比，PR 曲线对少数类检测更敏感，更适合本研究的类别不平衡场景。曲线中本文方法在大部分 Recall 区间保持更高 Precision，说明模型在扩大召回范围时仍能控制误检；LightGBM、LSTM 和 Transformer 曲线之间存在局部接近，表示不同模型在不同阈值区间可能各有优势。该图是模拟曲线，不应被解读为真实逐样本排序结果，正式论文需用真实预测概率重新绘制。

图5-4 核心故障识别 PR 曲线。对应文件：`{results_root.as_posix()}/figures/chapter5/fig5_4_core_pr_curve.png`。

图5-5给出了核心故障识别任务的 ROC 曲线。ROC 曲线从 TPR 与 FPR 的角度观察不同阈值下模型区分正负样本的能力，可与表5-5中的 ROC-AUC 指标形成对应。由于类别不平衡条件下 ROC 曲线可能显得较为乐观，本文将其作为补充图，主要用于说明模型整体排序能力，核心结论仍优先参考 F1、Recall 和 PR-AUC。

图5-5 核心故障识别 ROC 曲线。对应文件：`{results_root.as_posix()}/figures/chapter5/fig5_5_core_roc_curve.png`。



## 5.4 故障预警实验结果与分析

预警任务比识别任务更困难，因为模型需要在故障发生前从弱异常信号中判断未来风险。与识别任务一致，本节将五类故障整体平均和核心三类故障平均分开报告。五类故障整体平均用于观察模型面对完整故障集合时的预警覆盖能力，核心三类故障平均用于比较样本相对稳定故障上的主要预警性能。

表5-6 五类故障预警模型对比（模拟结果占位）

{warn_all_table}

从表5-6可以看出，本文方法的五类故障整体 Warning F1 为 {float(proposed_warn_all['mean_all_warning_f1']):.3f}，高于最强预警基线 {MODEL_LABELS.get(str(best_baseline_warn_all['model_name']), str(best_baseline_warn_all['model_name']))} 的 {float(best_baseline_warn_all['mean_all_warning_f1']):.3f}。由于该口径纳入突发型内短路和连接异常等稀缺故障，整体预警结果更能反映模型在完整任务集合上的部署压力。

表5-7 核心三类故障预警模型对比（模拟结果占位）

{warn_core_table}

从表5-7可以看出，本文方法的核心三类故障 Warning F1 为 {float(proposed_warn['mean_core_warning_f1']):.3f}，高于最强预警基线 {MODEL_LABELS.get(str(best_baseline_warn['model_name']), str(best_baseline_warn['model_name']))} 的 {float(best_baseline_warn['mean_core_warning_f1']):.3f}。这一结果说明双任务学习在模拟设定下能够利用识别任务形成的故障状态表征，为预警任务提供辅助监督。不过，预警指标不能只看 Recall；如果模型通过大幅增加误报来提高 Recall，实际部署价值会下降，因此需要结合 FPR 和提前量共同分析。

图5-6展示核心三类故障预警模型的 Warning F1 和 Warning Recall 对比。阈值规则基线具有较强可解释性，但容易受固定阈值影响；LightGBM 对统计特征敏感，能捕捉一部分预警信号；LSTM 与 Transformer 能利用窗口内变化趋势。本文方法在两项指标上整体领先，模拟结果支持共享时序特征和故障专属专家头对预警任务有正向作用。

图5-6 核心故障预警模型对比。对应文件：`{results_root.as_posix()}/figures/comparison/fig7_warning_model_comparison.png`。

图5-7进一步展示 Warning Recall 与 FPR 的关系。图中每个圆点对应一个模型，点的大小表示 Warning F1，图例给出模型名称。理想模型应位于左上区域，即较低 FPR 和较高 Warning Recall。本文方法位于相对靠左且靠上的区域，说明其不是单纯通过放宽阈值增加召回，而是在误报控制和预警发现之间取得较好折中。若后续真实结果中本文方法点位右移，说明模型可能存在过度预警问题，需要重新调节 `lambda_warn` 或阈值选择策略。

图5-7 预警 Recall-FPR 权衡。对应文件：`{results_root.as_posix()}/figures/chapter5/fig5_6_warning_recall_far_tradeoff.png`。

图5-8展示五类故障正确预警样本的 Mean Lead Time 分布。自放电异常通常具有渐进变化特征，因此提前量分布相对更宽；采样异常也具备一定趋势性；突发型内短路、连接异常和绝缘失效由于样本稀缺或突发性强，提前量更集中且上界较低。该图的价值不在于证明每类故障都能稳定提前预警，而在于提醒读者：不同故障类型的可预警性存在天然差异。

图5-8 五类故障正确预警样本提前量分布。对应文件：`{results_root.as_posix()}/figures/chapter5/fig5_7_warning_lead_time_distribution.png`。

## 5.5 消融实验

为验证模型结构设计的必要性，本文在五类故障整体口径下设置四组消融实验：去除故障专属特征、去除故障专家头、去除预警任务和去除标签质量控制。表5-8给出了完整模型与四组变体的指标对比。

表5-8 消融实验结果（五类故障整体口径，模拟结果占位）

{ablation_table}

从表5-8可以看出，去除故障专属特征和去除专家头都会导致 ID-F1 下降，说明五类故障虽然共享部分时序异常模式，但仍然需要故障级差异化建模。去除预警任务后，Warn-F1 明显下降，这符合任务定义，因为该变体不再直接优化预警目标。去除标签质量控制后，ID-F1 和 Warn-F1 同时下降，说明标签噪声会削弱模型对边界样本的学习能力。这里需要批判性地看待模拟结果：消融实验应服务于结构解释，而不是人为制造每个模块都“显著有效”的结论；真实实验若某个消融项差异很小，也应该如实写成该模块贡献有限。

图5-9将完整模型和四组消融变体放在同一张图中比较，横向展示 ID-F1 与 Warn-F1。该图用于直接考察不同结构组件被移除后对整体性能的影响。图中完整模型位于最上方，四组消融均有不同程度下降，其中专家头和标签质量控制对整体性能影响较明显。

图5-9 消融实验四组变体对比。对应文件：`{results_root.as_posix()}/figures/chapter5/fig5_8_ablation_comparison.png`。

## 5.6 训练过程与案例分析

图5-10展示本文方法的训练收敛曲线。曲线分为两个阶段：第一阶段只训练识别任务，使共享时序编码器获得较稳定的故障状态表征；第二阶段进行识别与预警联合训练。训练损失和验证损失均呈下降趋势，且阶段切换后没有出现明显发散，说明两阶段训练在模拟设定下具有较好的稳定性。但 loss 图只能说明优化过程稳定，不能单独证明泛化能力，仍需结合测试集指标判断。

图5-10 双任务模型训练收敛曲线。对应文件：`{results_root.as_posix()}/figures/training/fig2_loss_curve.png`。

图5-11给出一个正确识别案例。该案例中模型得分随时间推进逐渐升高，并在故障发生前后超过决策阈值，说明模型能够利用窗口内趋势变化形成较稳定判断。案例图适合用于解释模型如何从连续信号中形成判别结果，但不应取代整体测试集统计。

图5-11 正确识别案例趋势图。对应文件：`{results_root.as_posix()}/figures/cases/fig8_isc_true_positive.png`。

图5-12给出一个漏识别案例。该案例中模型得分虽然出现波动，但未能稳定越过阈值，反映出边界样本、弱异常信号和稀缺故障仍然是模型难点。论文中保留漏识别案例是有价值的，因为它能避免实验部分只展示成功样本，也能为后续改进方向提供依据。

图5-12 漏识别案例趋势图。对应文件：`{results_root.as_posix()}/figures/cases/fig9_samp_false_negative.png`。

## 5.7 模拟结果与真实结果差异

为了避免把模拟结果误写成正式结论，本节将模拟识别结果与当前 `results/round2` 中的真实结果进行对照。表5-9显示，当前真实结果显著低于模拟占位结果，这说明真实 full-data 实验仍存在模型训练、标签噪声、类别稀缺或阈值选择方面的问题。该差异具有较强提示意义：模拟结果只能用于组织论文表达和检验展示逻辑，不能替代基于真实实验得到的研究结论。
'''
    if real_vs_sim_table:
        draft += f'''
表5-9 模拟结果与真实识别结果对照

{real_vs_sim_table}

从表5-9可以看出，模拟结果与真实结果之间仍有较大差距。因此，论文最终定稿时应优先使用真实结果；如果真实结果仍然较差，实验章节需要转向“问题诊断型”写法，而不是强行写成模型显著优越。当前模拟稿主要用于检验章节结构、图表组织和结果解释逻辑。
'''
    draft += '''
## 5.8 本章小结

本章基于真实全量样本分布和模拟性能结果，构建了实验部分的完整分析框架。相较于仅呈现单一柱状图结果，本章同时使用实验参数表、样本统计表、模型对比表、混淆率热力图、PR 曲线、Recall-FPR 权衡图、提前量分布图、消融实验表和案例图，从多个角度解释模型表现。需要强调的是，本章所有性能数值仍为模拟占位，后续必须以 `results/round2` 中真实 full-data 实验结果替换，并基于真实结果重新判断结论是否成立。
'''
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(draft, encoding='utf-8')

def export_chapter5_supplements(results_root: Path, samples_root: Path, real_results_root: Path, draft_path: Path) -> dict[str, Path]:
    apply_academic_plot_style()
    results_root = Path(results_root)
    samples_root = Path(samples_root)
    real_results_root = Path(real_results_root)
    chapter5_root = results_root / 'figures' / 'chapter5'
    chapter5_root.mkdir(parents=True, exist_ok=True)

    _export_experiment_settings(results_root, samples_root, chapter5_root)
    _export_label_distribution(samples_root, chapter5_root)
    _export_confusion_rate_heatmap(results_root, samples_root, chapter5_root)
    _export_core_pr_curve(results_root, chapter5_root)
    _export_core_roc_curve(results_root, chapter5_root)
    _export_warning_tradeoff(results_root, chapter5_root)
    _export_warning_lead_distribution(results_root, samples_root, chapter5_root)
    ablation = _export_ablation_table(results_root, chapter5_root)
    _export_ablation_figure(results_root, chapter5_root, ablation)
    _export_real_vs_simulated(results_root, real_results_root, chapter5_root)
    _write_chapter5_draft(results_root, samples_root, real_results_root, chapter5_root, draft_path)

    index = pd.DataFrame(
        [
            {'图号': '图5-1', '标题': '全量样本五类故障标签分布', '路径': str(chapter5_root / 'fig5_1_label_distribution.png')},
            {'图号': '图5-3', '标题': '五类故障识别混淆率热力图', '路径': str(chapter5_root / 'fig5_3_recognition_confusion_heatmap.png')},
            {'图号': '图5-4', '标题': '核心故障识别 PR 曲线', '路径': str(chapter5_root / 'fig5_4_core_pr_curve.png')},
            {'??': '?5-5', '??': '?????? ROC ??', '??': str(chapter5_root / 'fig5_5_core_roc_curve.png')},
            {'图号': '图5-7', '标题': '预警召回率与误报率权衡', '路径': str(chapter5_root / 'fig5_6_warning_recall_far_tradeoff.png')},
            {'图号': '图5-8', '标题': '五类故障正确预警样本提前量分布', '路径': str(chapter5_root / 'fig5_7_warning_lead_time_distribution.png')},
            {'图号': '图5-9', '标题': '消融实验四组变体对比', '路径': str(chapter5_root / 'fig5_8_ablation_comparison.png')},
        ]
    )
    index.to_csv(chapter5_root / 'chapter5_supplement_figure_index.csv', index=False, encoding='utf-8-sig')
    return {'chapter5_root': chapter5_root, 'draft_path': Path(draft_path)}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = export_chapter5_supplements(
        Path(args.results_root),
        Path(args.samples_root),
        Path(args.real_results_root),
        Path(args.draft_path),
    )
    for name, path in outputs.items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()

