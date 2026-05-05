from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import DATASET_META_ROOT
from battery_thesis.profiling import build_project_profiles, build_structured_split

DEFAULT_STRUCTURED_ROOT = Path(r'F:\Data_set')
DEFAULT_RAW_ROOT = Path(r'F:\RAW_DATA\data')
SEED = 20260407
CORE_ID_FAULTS = ['sd', 'samp', 'ins']
CORE_WARNING_FAULTS = ['sd', 'samp', 'ins']
RARE_FAULTS = {'isc': 'train_plus_eval', 'conn': 'eval_only'}


def _build_constraints_markdown(structured_profiles: object) -> str:
    id_vehicle_counts = {
        fault: int((structured_profiles[f'y_id_{fault}_positive'] > 0).sum())
        for fault in ['sd', 'isc', 'conn', 'samp', 'ins']
    }
    warn_vehicle_counts = {
        fault: int((structured_profiles[f'y_warn_{fault}_positive'] > 0).sum())
        for fault in CORE_WARNING_FAULTS
    }

    limitation_lines: list[str] = []
    for fault in CORE_ID_FAULTS:
        count = id_vehicle_counts[fault]
        if count < 3:
            limitation_lines.append(f'- 核心识别故障 {fault} 仅有 {count} 辆窗口级正例车辆，划分与评估需要谨慎解释。')
    for fault in CORE_WARNING_FAULTS:
        count = warn_vehicle_counts[fault]
        if count < 2:
            limitation_lines.append(f'- 核心预警故障 {fault} 仅有 {count} 辆窗口级预警车辆，val/test 的稳定性会受影响。')
    for fault, rule in RARE_FAULTS.items():
        count = id_vehicle_counts[fault]
        if count == 0:
            limitation_lines.append(f'- 稀缺故障 {fault} 当前没有窗口级正例车辆。')
        elif rule == 'train_plus_eval' and count < 2:
            limitation_lines.append(f'- 稀缺故障 {fault} 仅有 {count} 辆窗口级正例车辆，无法同时满足训练与评估分离。')
        elif rule == 'eval_only' and count < 1:
            limitation_lines.append(f'- 稀缺故障 {fault} 当前无法保留到评估集。')
    if not limitation_lines:
        limitation_lines.append('- 当前窗口级统计满足第1轮的核心划分约束。')

    lines = [
        '# 第1轮划分约束说明',
        '',
        '- 第1轮采用核心任务硬约束 + 稀缺故障软约束。',
        '- 核心识别任务为：sd / samp / ins。',
        '- 稀缺探索任务为：isc / conn。',
        '- train / val / test 需要覆盖核心识别正例。',
        '- val / test 需要覆盖核心预警正例。',
        '- isc 默认要求：训练集至少 1 辆 + 验证/测试至少 1 辆。',
        '- conn 默认要求：至少保留在验证集或测试集。',
        '',
        '## 数据限制说明',
        '',
        *limitation_lines,
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Build dataset metadata, structured profiles, and vehicle splits.')
    parser.add_argument('--structured-root', default=str(DEFAULT_STRUCTURED_ROOT))
    parser.add_argument('--raw-root', default=str(DEFAULT_RAW_ROOT))
    parser.add_argument('--skip-raw', action='store_true', help='Skip RAW_DATA scanning for round-one Data_set-only workflows.')
    args = parser.parse_args()

    DATASET_META_ROOT.mkdir(parents=True, exist_ok=True)
    structured_root = Path(args.structured_root)
    raw_root = Path(args.raw_root)

    manifest, structured_profiles, raw_profiles = build_project_profiles(
        structured_root=structured_root,
        raw_root=raw_root,
        include_raw=not args.skip_raw,
    )
    manifest.to_csv(DATASET_META_ROOT / 'vehicle_manifest.csv', index=False, encoding='utf-8-sig')
    structured_profiles.to_csv(DATASET_META_ROOT / 'structured_vehicle_profile.csv', index=False, encoding='utf-8-sig')
    if not raw_profiles.empty:
        raw_profiles.to_csv(DATASET_META_ROOT / 'raw_vehicle_profile.csv', index=False, encoding='utf-8-sig')

    structured_split = build_structured_split(structured_profiles, seed=SEED)
    structured_split.to_csv(DATASET_META_ROOT / 'structured_vehicle_split.csv', index=False, encoding='utf-8-sig')

    (DATASET_META_ROOT / 'selection_seed.txt').write_text(str(SEED), encoding='utf-8-sig')
    (DATASET_META_ROOT / 'selection_constraints.md').write_text(
        _build_constraints_markdown(structured_profiles),
        encoding='utf-8-sig',
    )


if __name__ == '__main__':
    main()
