from __future__ import annotations

import random
from itertools import combinations

import pandas as pd

from .config import FAULT_ORDER


DEFAULT_CORE_ID_FAULTS = ['sd', 'samp', 'ins']
DEFAULT_CORE_WARNING_FAULTS = ['sd', 'samp', 'ins']
DEFAULT_RARE_FAULT_RULES = {'isc': 'train_plus_eval', 'conn': 'eval_only'}


FAULT_INDEX = {fault: idx for idx, fault in enumerate(FAULT_ORDER)}


def _fault_mask(value: str, faults: list[str] | None = None) -> int:
    faults = faults or FAULT_ORDER
    bits = [int(bit) for bit in str(value).split('|')]
    mask = 0
    for idx, fault in enumerate(faults):
        fault_idx = FAULT_INDEX[fault]
        if fault_idx < len(bits) and bits[fault_idx]:
            mask |= 1 << idx
    return mask



def _count_mask_from_columns(row: pd.Series, prefix: str, faults: list[str]) -> int:
    mask = 0
    for idx, fault in enumerate(faults):
        if int(row.get(f'{prefix}{fault}_positive', 0) or 0) > 0:
            mask |= 1 << idx
    return mask



def _bit_count(value: int) -> int:
    return int(value.bit_count())



def _positive_count(row: pd.Series, prefix: str, faults: list[str], fallback_column: str | None = None) -> int:
    if all(f'{prefix}{fault}_positive' in row.index for fault in faults):
        return int(sum(int(row.get(f'{prefix}{fault}_positive', 0) or 0) for fault in faults))
    if fallback_column is None:
        return 0
    return _bit_count(_fault_mask(str(row.get(fallback_column, '0|0|0|0|0')), faults))



def _has_fault_positive(row: pd.Series, fault: str, prefix: str, fallback_column: str) -> bool:
    column = f'{prefix}{fault}_positive'
    if column in row.index:
        return int(row.get(column, 0) or 0) > 0
    return bool(_fault_mask(str(row.get(fallback_column, '0|0|0|0|0')), [fault]))



def _prepare_vehicle_meta(
    vehicle_meta: pd.DataFrame,
    core_id_faults: list[str],
    core_warning_faults: list[str],
    rare_fault_rules: dict[str, str],
) -> tuple[pd.DataFrame, int, int, dict[str, int]]:
    working = vehicle_meta.copy().reset_index(drop=True)
    working['core_id_mask'] = working.apply(
        lambda row: _count_mask_from_columns(row, 'y_id_', core_id_faults)
        if all(f'y_id_{fault}_positive' in working.columns for fault in core_id_faults)
        else _fault_mask(str(row.get('fault_vector', '0|0|0|0|0')), core_id_faults),
        axis=1,
    )
    working['core_warn_mask'] = working.apply(
        lambda row: _count_mask_from_columns(row, 'y_warn_', core_warning_faults)
        if all(f'y_warn_{fault}_positive' in working.columns for fault in core_warning_faults)
        else _fault_mask(str(row.get('warning_fault_vector', '0|0|0|0|0')), core_warning_faults),
        axis=1,
    )
    working['core_id_positive_total'] = working.apply(
        lambda row: _positive_count(row, 'y_id_', core_id_faults, 'fault_vector'),
        axis=1,
    )
    working['core_warn_positive_total'] = working.apply(
        lambda row: _positive_count(row, 'y_warn_', core_warning_faults, 'warning_fault_vector'),
        axis=1,
    )

    available_rare: dict[str, int] = {}
    for fault in rare_fault_rules:
        column = f'rare_{fault}_positive'
        working[column] = working.apply(
            lambda row, current_fault=fault: int(_has_fault_positive(row, current_fault, 'y_id_', 'fault_vector')),
            axis=1,
        )
        available_rare[fault] = int((working[column] > 0).sum())

    if 'window_count' not in working.columns:
        working['window_count'] = 0
    else:
        working['window_count'] = pd.to_numeric(working['window_count'], errors='coerce').fillna(0).astype(int)

    core_id_full_mask = (1 << len(core_id_faults)) - 1
    core_warn_full_mask = (1 << len(core_warning_faults)) - 1 if core_warning_faults else 0
    return working, core_id_full_mask, core_warn_full_mask, available_rare



def _subset_stat(working: pd.DataFrame, subset: tuple[int, ...], rare_faults: list[str]) -> dict[str, object]:
    subset_df = working.loc[list(subset)]
    core_id_cover = 0
    core_warn_cover = 0
    for value in subset_df['core_id_mask']:
        core_id_cover |= int(value)
    for value in subset_df['core_warn_mask']:
        core_warn_cover |= int(value)
    vehicle_ids = tuple(sorted(subset_df['vehicle_id'].astype(str).tolist()))
    rare_presence = {
        fault: bool(subset_df[f'rare_{fault}_positive'].sum() > 0)
        for fault in rare_faults
    }
    return {
        'subset': subset,
        'subset_set': frozenset(subset),
        'vehicle_ids': vehicle_ids,
        'core_id_cover': core_id_cover,
        'core_warn_cover': core_warn_cover,
        'core_id_bits': _bit_count(core_id_cover),
        'core_warn_bits': _bit_count(core_warn_cover),
        'core_id_positive_total': int(subset_df['core_id_positive_total'].sum()),
        'core_warn_positive_total': int(subset_df['core_warn_positive_total'].sum()),
        'window_total': int(subset_df['window_count'].sum()),
        'rare_presence': rare_presence,
    }



def _candidate_sort_key(
    stat: dict[str, object],
    core_id_full_mask: int,
    core_warn_full_mask: int,
    rare_fault_rules: dict[str, str],
) -> tuple[object, ...]:
    rare_eval_hits = sum(int(stat['rare_presence'].get(fault, False)) for fault in rare_fault_rules)
    return (
        -int(int(stat['core_id_cover']) == core_id_full_mask),
        -int(core_warn_full_mask == 0 or int(stat['core_warn_cover']) == core_warn_full_mask),
        -int(stat['core_id_bits']),
        -int(stat['core_warn_bits']),
        -rare_eval_hits,
        -int(stat['core_id_positive_total']),
        -int(stat['core_warn_positive_total']),
        -int(stat['window_total']),
        stat['vehicle_ids'],
    )



def _enumerate_candidate_stats(
    working: pd.DataFrame,
    subset_size: int,
    core_id_full_mask: int,
    core_warn_full_mask: int,
    rare_fault_rules: dict[str, str],
) -> list[dict[str, object]]:
    rare_faults = list(rare_fault_rules.keys())
    stats = [_subset_stat(working, subset, rare_faults) for subset in combinations(working.index.tolist(), subset_size)]
    return sorted(
        stats,
        key=lambda stat: _candidate_sort_key(stat, core_id_full_mask, core_warn_full_mask, rare_fault_rules),
    )



def _rare_assignment_score(
    train_stat: dict[str, object],
    val_stat: dict[str, object],
    test_stat: dict[str, object],
    available_rare: dict[str, int],
    rare_fault_rules: dict[str, str],
) -> tuple[int, ...]:
    eval_presence = {
        fault: bool(val_stat['rare_presence'].get(fault, False) or test_stat['rare_presence'].get(fault, False))
        for fault in rare_fault_rules
    }
    train_presence = {
        fault: bool(train_stat['rare_presence'].get(fault, False))
        for fault in rare_fault_rules
    }

    score: list[int] = []
    for fault, rule in rare_fault_rules.items():
        available_count = int(available_rare.get(fault, 0))
        if available_count <= 0:
            continue
        if rule == 'train_plus_eval':
            score.extend(
                [
                    -int(available_count >= 2 and train_presence[fault] and eval_presence[fault]),
                    -int(train_presence[fault]),
                    -int(eval_presence[fault]),
                ]
            )
        elif rule == 'eval_only':
            score.append(-int(eval_presence[fault]))
    return tuple(score)



def assign_vehicle_splits(
    vehicle_meta: pd.DataFrame,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
    prioritize_fault_coverage: bool,
    required_warning_faults: list[str] | None = None,
    core_id_faults: list[str] | None = None,
    rare_fault_rules: dict[str, str] | None = None,
) -> pd.DataFrame:
    if train_count + val_count + test_count != len(vehicle_meta):
        raise ValueError('Requested split counts must match vehicle count.')

    core_id_faults = core_id_faults or DEFAULT_CORE_ID_FAULTS
    core_warning_faults = required_warning_faults or DEFAULT_CORE_WARNING_FAULTS
    rare_fault_rules = rare_fault_rules or DEFAULT_RARE_FAULT_RULES

    working, core_id_full_mask, core_warn_full_mask, available_rare = _prepare_vehicle_meta(
        vehicle_meta,
        core_id_faults=core_id_faults,
        core_warning_faults=core_warning_faults,
        rare_fault_rules=rare_fault_rules,
    )
    working['split'] = None

    if not prioritize_fault_coverage:
        rng = random.Random(seed)
        indices = working.index.tolist()
        rng.shuffle(indices)
        test_indices = set(indices[:test_count])
        val_indices = set(indices[test_count:test_count + val_count])
        for idx in working.index:
            if idx in test_indices:
                working.at[idx, 'split'] = 'test'
            elif idx in val_indices:
                working.at[idx, 'split'] = 'val'
            else:
                working.at[idx, 'split'] = 'train'
        return working.drop(columns=['core_id_mask', 'core_warn_mask', 'core_id_positive_total', 'core_warn_positive_total', *[f'rare_{fault}_positive' for fault in rare_fault_rules]]).sort_values('vehicle_id').reset_index(drop=True)

    test_candidates = _enumerate_candidate_stats(working, test_count, core_id_full_mask, core_warn_full_mask, rare_fault_rules)
    val_candidates = _enumerate_candidate_stats(working, val_count, core_id_full_mask, core_warn_full_mask, rare_fault_rules)
    max_candidate_pairs = 800 if len(working) >= 20 else max(len(test_candidates), len(val_candidates))
    test_candidates = test_candidates[:max_candidate_pairs]
    val_candidates = val_candidates[:max_candidate_pairs]
    all_indices = frozenset(working.index.tolist())

    best_assignment: tuple[dict[str, object], dict[str, object], dict[str, object]] | None = None
    best_key: tuple[object, ...] | None = None

    for test_stat in test_candidates:
        for val_stat in val_candidates:
            if not test_stat['subset_set'].isdisjoint(val_stat['subset_set']):
                continue
            train_subset = tuple(sorted(all_indices - test_stat['subset_set'] - val_stat['subset_set']))
            train_stat = _subset_stat(working, train_subset, list(rare_fault_rules.keys()))
            assignment_key = (
                -int(int(test_stat['core_id_cover']) == core_id_full_mask),
                -int(int(val_stat['core_id_cover']) == core_id_full_mask),
                -int(int(train_stat['core_id_cover']) == core_id_full_mask),
                -int(core_warn_full_mask == 0 or int(test_stat['core_warn_cover']) == core_warn_full_mask),
                -int(core_warn_full_mask == 0 or int(val_stat['core_warn_cover']) == core_warn_full_mask),
                *_rare_assignment_score(train_stat, val_stat, test_stat, available_rare, rare_fault_rules),
                -int(train_stat['core_id_bits']),
                -int(test_stat['core_id_positive_total'] + val_stat['core_id_positive_total'] + train_stat['core_id_positive_total']),
                -int(test_stat['core_warn_positive_total'] + val_stat['core_warn_positive_total']),
                -int(test_stat['window_total'] + val_stat['window_total']),
                test_stat['vehicle_ids'],
                val_stat['vehicle_ids'],
            )
            if best_key is None or assignment_key < best_key:
                best_key = assignment_key
                best_assignment = (test_stat, val_stat, train_stat)

    if best_assignment is None:
        raise ValueError('Unable to assign vehicle splits with the requested counts.')

    test_stat, val_stat, _ = best_assignment
    for idx in working.index:
        if idx in test_stat['subset_set']:
            working.at[idx, 'split'] = 'test'
        elif idx in val_stat['subset_set']:
            working.at[idx, 'split'] = 'val'
        else:
            working.at[idx, 'split'] = 'train'

    drop_columns = ['core_id_mask', 'core_warn_mask', 'core_id_positive_total', 'core_warn_positive_total', *[f'rare_{fault}_positive' for fault in rare_fault_rules]]
    output = working.drop(columns=drop_columns).sort_values('vehicle_id').reset_index(drop=True)
    if (output['split'] == 'train').sum() != train_count or (output['split'] == 'val').sum() != val_count or (output['split'] == 'test').sum() != test_count:
        raise ValueError('Assigned split counts do not match requested counts.')
    return output
