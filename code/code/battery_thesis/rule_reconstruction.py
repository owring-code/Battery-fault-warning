from __future__ import annotations

import math

import numpy as np
import pandas as pd


ORIGINAL_VOLTAGE_GROUPS = [
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (10, 11, 12, 13, 14),
    (15, 16, 17, 18, 19),
    (20, 21, 22, 23, 24),
    (25, 26, 27, 28, 29),
    (30, 31, 32, 33, 34),
    (35, 36, 37, 38, 39),
    (40, 41, 42, 43, 44),
    (45, 46, 47, 48, 49),
    (50, 51, 52, 53, 54, 55),
    (56, 57, 58, 59, 60, 61),
    (62, 63, 64, 65, 66, 67),
    (68, 69, 70, 71, 72, 73),
    (74, 75, 76, 77, 78, 79),
    (80, 81, 82, 83, 84, 85),
    (86, 87, 88, 89, 90, 91),
]
ORIGINAL_TEMPERATURE_GROUPS = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (16, 17),
    (18, 19),
    (20, 21),
    (22, 23),
    (24, 25),
    (26, 27),
    (28, 29),
    (30, 31),
    (32, 33),
]


def _sanitize_sensor_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix
    if np.nanmax(np.abs(matrix)) > 1000:
        matrix = matrix / 1000.0
    for column_idx in range(matrix.shape[1]):
        column = matrix[:, column_idx]
        non_zero = column[column != 0]
        replacement = float(np.median(non_zero)) if non_zero.size else 0.0
        column[column == 0] = replacement
        matrix[:, column_idx] = column
    return matrix



def reconstruct_self_discharge_labels(
    voltages: np.ndarray,
    rolling_window: int = 300,
    outlier_iqr_factor: float = 3.0,
    activation_ratio: float = 0.7,
) -> np.ndarray:
    matrix = _sanitize_sensor_matrix(voltages)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0] if matrix.ndim >= 1 else 0, dtype=np.int64)

    length = matrix.shape[0]
    window = max(2, min(int(rolling_window), length))
    centered = matrix - np.median(matrix, axis=0)

    residuals = []
    valid_length = length - window + 1
    weights = np.ones(window, dtype=float) / window
    for column_idx in range(matrix.shape[1]):
        column = centered[:, column_idx]
        rolling_avg = np.convolve(column, weights, mode='valid')
        uf = np.empty(length, dtype=float)
        uf[:valid_length] = column[:valid_length] - rolling_avg
        if valid_length < length:
            remaining_avg = float(np.mean(column[valid_length - 1 :]))
            uf[valid_length:] = column[valid_length:] - remaining_avg
        residuals.append(uf)

    residual_matrix = np.stack(residuals, axis=1)
    flattened = residual_matrix.reshape(-1)
    q1 = float(np.percentile(flattened, 25))
    q3 = float(np.percentile(flattened, 75))
    iqr = q3 - q1
    lower = q1 - outlier_iqr_factor * iqr
    upper = q3 + outlier_iqr_factor * iqr

    min_per_frame = np.min(residual_matrix, axis=1)
    anomaly_matrix = np.zeros_like(residual_matrix, dtype=np.int64)
    for frame_idx in range(length):
        for cell_idx in range(matrix.shape[1]):
            value = residual_matrix[frame_idx, cell_idx]
            if (value < lower or value > upper) and value == min_per_frame[frame_idx]:
                anomaly_matrix[frame_idx, cell_idx] = 1

    per_cell_counts = anomaly_matrix.sum(axis=0)
    if per_cell_counts.max(initial=0) <= 0:
        return np.zeros(length, dtype=np.int64)
    activation_threshold = per_cell_counts.max() * activation_ratio
    active_cells = np.where(per_cell_counts > activation_threshold)[0]
    if active_cells.size == 0:
        return np.zeros(length, dtype=np.int64)
    return (anomaly_matrix[:, active_cells].sum(axis=1) > 0).astype(np.int64)



def reconstruct_sampling_labels(
    voltages: np.ndarray,
    charge_status: np.ndarray,
    window_length: int = 3,
    discharge_status: int = 3,
) -> np.ndarray:
    matrix = _sanitize_sensor_matrix(voltages)
    status = np.asarray(charge_status, dtype=int)
    labels = np.zeros(len(status), dtype=np.int64)
    if matrix.ndim != 2 or len(status) != matrix.shape[0] or matrix.shape[0] == 0:
        return labels

    discharge_indices = np.where(status == discharge_status)[0]
    if discharge_indices.size == 0:
        return labels
    discharge_matrix = matrix[discharge_indices].copy()
    rows, cols = discharge_matrix.shape
    if rows < 2 or cols < 2:
        return labels

    row_medians = np.median(discharge_matrix, axis=1, keepdims=True)
    target_matrix = discharge_matrix - row_medians
    effective_window = max(1, min(int(window_length), rows))

    distance_accumulation_matrix = []
    for start_idx in range(rows):
        window = target_matrix[start_idx : start_idx + effective_window, :]
        distance_accumulation_matrix.append(np.sum(window, axis=0))
    distance_accumulation_matrix = np.asarray(distance_accumulation_matrix, dtype=float)

    first_quartiles = []
    second_quartiles = []
    for column_idx in range(cols):
        column = distance_accumulation_matrix[:, column_idx]
        first_quartiles.append(float(np.percentile(column, 5)))
        second_quartiles.append(float(np.percentile(column, 95)))

    global_first = float(np.percentile(np.asarray(sorted(first_quartiles), dtype=float), 5))
    global_second = float(np.percentile(np.asarray(sorted(second_quartiles), dtype=float), 95))

    anomaly_matrix = np.zeros((rows, cols), dtype=np.int64)
    for row_idx in range(rows):
        for column_idx in range(cols):
            current = distance_accumulation_matrix[row_idx, column_idx]
            if current < global_first:
                has_right = column_idx < cols - 1 and distance_accumulation_matrix[row_idx, column_idx + 1] > global_second
                has_left = column_idx > 0 and distance_accumulation_matrix[row_idx, column_idx - 1] > global_second
                if has_left or has_right:
                    anomaly_matrix[row_idx, column_idx] = 1

    per_cell_counts = anomaly_matrix.sum(axis=0)
    active_cells = np.where(per_cell_counts > rows * 0.01)[0]
    if active_cells.size == 0:
        return labels

    discharge_labels = (anomaly_matrix[:, active_cells].sum(axis=1) > 0).astype(np.int64)
    labels[discharge_indices] = discharge_labels
    return labels



def reconstruct_insulation_labels(
    charge_status: np.ndarray,
    insulation_resistance: np.ndarray,
) -> np.ndarray:
    status = np.asarray(charge_status, dtype=int)
    resistance = np.nan_to_num(np.asarray(insulation_resistance, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.zeros(len(status), dtype=np.int64)
    if len(status) == 0:
        return labels

    data_range = float(resistance.max() - resistance.min())
    for idx in range(len(status)):
        if status[idx] in {1, 2}:
            if resistance[idx] < 41:
                labels[idx] = 1
        elif status[idx] in {3, 4}:
            if resistance[idx] < 205:
                labels[idx] = 1
            if idx < 2 or idx + 2 > len(status):
                continue
            if ((status[idx + 1] in {1, 2}) or (status[idx - 1] in {1, 2})) and resistance[idx] > 41:
                labels[idx] = 0

    idx = 0
    while idx < len(status):
        if labels[idx] != 1:
            idx += 1
            continue
        next_idx = idx + 1
        while next_idx < len(status):
            diff = abs(resistance[next_idx] - resistance[next_idx - 1])
            if diff < 0.003 * data_range:
                labels[next_idx] = 1
                next_idx += 1
                break
            if diff > 0.003 * data_range:
                idx = next_idx - 1
                break
            next_idx += 1
        idx = next_idx
    return labels



def _build_contiguous_groups(total: int, num_groups: int) -> list[tuple[int, ...]]:
    if total <= 0:
        return []
    groups = []
    start = 0
    base = total // num_groups
    remainder = total % num_groups
    for group_idx in range(num_groups):
        width = base + (1 if group_idx < remainder else 0)
        if width <= 0:
            continue
        stop = start + width
        groups.append(tuple(range(start, stop)))
        start = stop
    return groups



def _voltage_module_groups(num_cells: int, num_temp_groups: int) -> list[tuple[int, ...]]:
    if num_cells == 92:
        return ORIGINAL_VOLTAGE_GROUPS
    module_count = max(1, min(num_temp_groups or 1, num_cells))
    return _build_contiguous_groups(num_cells, module_count)



def _temperature_module_groups(num_temps: int, num_voltage_groups: int) -> list[tuple[int, ...]]:
    if num_temps == 34:
        return ORIGINAL_TEMPERATURE_GROUPS
    module_count = max(1, min(num_voltage_groups or 1, num_temps))
    return _build_contiguous_groups(num_temps, module_count)



def _series_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return abs(point1[1] - point2[1])



def _rand_rows(array: np.ndarray, dim_needed: int, seed: int = 20260407) -> np.ndarray:
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    rng = np.random.default_rng(seed)
    rng.shuffle(row_sequence)
    return array[row_sequence[: min(dim_needed, row_total)], :]



def _k(sample_size: int, sigma: float, z_value: float = 1.96) -> int:
    if sample_size <= 1 or sigma <= 0:
        return max(sample_size, 1)
    numerator = sample_size * (z_value ** 2) * (sigma ** 2)
    denominator = ((sample_size - 1) * (0.1 * sigma) ** 2) + (sigma ** 2) * (z_value ** 2)
    if denominator == 0:
        return sample_size
    return max(int(round(numerator / denominator)), 1)



def _distance_matrix(values: np.ndarray, observations: np.ndarray) -> np.ndarray:
    distance_matrix = np.zeros((values.shape[0], observations.shape[0]), dtype=float)
    for i in range(values.shape[0]):
        for j in range(observations.shape[0]):
            distance_matrix[i, j] = _series_distance(values[i], observations[j])
    return distance_matrix



def _nearest_identifiers(distance_matrix: np.ndarray, neighbor_count: int = 6) -> np.ndarray:
    matrix = distance_matrix.copy()
    identifiers = np.zeros((matrix.shape[0], neighbor_count), dtype=int)
    for neighbor_idx in range(neighbor_count):
        min_indices = matrix.argmin(axis=1)
        identifiers[:, neighbor_idx] = min_indices
        matrix[np.arange(matrix.shape[0]), min_indices] = np.inf
    return identifiers



def _prune_observations(identifier: np.ndarray, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if observations.shape[0] == 0:
        return np.zeros(0, dtype=float), observations
    counts = np.bincount(identifier.reshape(-1), minlength=observations.shape[0]).astype(float)
    cutoff = np.percentile(counts, 33)
    keep_mask = counts >= cutoff
    if not keep_mask.any():
        return counts, observations
    return counts[keep_mask], observations[keep_mask]



def _nearest_identifiers_from_values(
    observation_values: np.ndarray,
    candidate_values: np.ndarray,
    neighbor_count: int = 6,
    chunk_size: int = 50000,
) -> np.ndarray:
    observations = np.asarray(observation_values, dtype=float)
    candidates = np.asarray(candidate_values, dtype=float)
    if observations.size == 0 or candidates.size == 0:
        return np.zeros((observations.size, 0), dtype=int)

    k = min(max(int(neighbor_count), 1), candidates.size)
    identifiers = np.empty((observations.size, k), dtype=int)
    for start in range(0, observations.size, chunk_size):
        stop = min(start + chunk_size, observations.size)
        distances = np.abs(observations[start:stop, None] - candidates[None, :])
        identifiers[start:stop] = np.argsort(distances, axis=1, kind='stable')[:, :k]
    return identifiers



def _sdo_scores(series: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(series, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    sample_size = len(values)
    if sample_size < 3:
        return np.zeros(sample_size, dtype=float)
    observations = np.column_stack([np.arange(sample_size, dtype=float), values])
    sigma = float(np.std(observations[:, 1]))
    neighbor_count = _k(sample_size, sigma)
    sampled = _rand_rows(observations, neighbor_count)
    identifiers = _nearest_identifiers_from_values(observations[:, 1], sampled[:, 1])
    _, pruned = _prune_observations(identifiers, sampled)
    if pruned.size == 0:
        pruned = sampled
    pruned_identifiers = _nearest_identifiers_from_values(observations[:, 1], pruned[:, 1])

    average = float(np.average(observations[:, 1]))
    direction = np.sign(observations[:, 1] - average)
    neighbor_values = pruned[pruned_identifiers, 1]
    mean_distance = np.abs(observations[:, 1][:, None] - neighbor_values).mean(axis=1)
    return direction * mean_distance



def reconstruct_internal_short_labels(
    voltages: np.ndarray,
    temperatures: np.ndarray,
    temperature_threshold: float = 100.0,
    score_threshold: float = -1.0,
) -> np.ndarray:
    voltage_matrix = _sanitize_sensor_matrix(voltages)
    temperature_matrix = _sanitize_sensor_matrix(temperatures)
    length = voltage_matrix.shape[0] if voltage_matrix.ndim == 2 else 0
    labels = np.zeros(length, dtype=np.int64)
    if length == 0 or temperature_matrix.ndim != 2 or temperature_matrix.shape[0] != length:
        return labels

    voltage_groups = _voltage_module_groups(voltage_matrix.shape[1], len(ORIGINAL_TEMPERATURE_GROUPS) if temperature_matrix.shape[1] == 34 else temperature_matrix.shape[1])
    temperature_groups = _temperature_module_groups(temperature_matrix.shape[1], len(voltage_groups))
    module_count = min(len(voltage_groups), len(temperature_groups))
    if module_count == 0:
        return labels

    for module_idx in range(module_count):
        module_voltage = np.min(voltage_matrix[:, voltage_groups[module_idx]], axis=1)
        module_temperature = np.max(temperature_matrix[:, temperature_groups[module_idx]], axis=1)
        scores = _sdo_scores(module_voltage)
        module_flags = (module_temperature >= temperature_threshold) | ((scores <= score_threshold) & (module_voltage != 0))
        labels = np.maximum(labels, module_flags.astype(np.int64))
    return labels



def _delta_matrix(voltages: np.ndarray) -> np.ndarray:
    matrix = _sanitize_sensor_matrix(voltages)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    deltas = np.zeros_like(matrix, dtype=float)
    if matrix.shape[0] > 1:
        deltas[1:, :] = np.diff(matrix, axis=0) / 10.0
    return deltas.T



def _calculate_icc(v1: np.ndarray, v2: np.ndarray) -> float:
    length = len(v1)
    if length == 0:
        return 1.0
    x1 = np.asarray(v1, dtype=float)
    x2 = np.asarray(v2, dtype=float)
    mean_value = (np.sum(x1) + np.sum(x2)) / (2 * length)
    variance = (np.sum((x1 - mean_value) ** 2) + np.sum((x2 - mean_value) ** 2)) / (2 * length)
    if variance == 0:
        return 1.0
    covariance = np.sum((x1 - mean_value) * (x2 - mean_value))
    return float(covariance / (length * variance))



def reconstruct_connection_labels(
    voltages: np.ndarray,
    icc_threshold: float = 0.8,
    normalized_delta_threshold: float = 0.7,
) -> np.ndarray:
    delta_by_cell = _delta_matrix(voltages)
    if delta_by_cell.ndim != 2 or delta_by_cell.shape[0] < 2:
        return np.zeros(delta_by_cell.shape[1] if delta_by_cell.ndim == 2 else 0, dtype=np.int64)

    anomaly_counts = np.zeros(delta_by_cell.shape[0], dtype=int)
    for cell_idx in range(delta_by_cell.shape[0] - 1):
        icc = _calculate_icc(delta_by_cell[cell_idx], delta_by_cell[cell_idx + 1])
        if icc < icc_threshold:
            anomaly_counts[cell_idx] += 1
            anomaly_counts[cell_idx + 1] += 1

    suspicious_cells = np.where(anomaly_counts > 1)[0]
    if suspicious_cells.size == 0:
        suspicious_cells = np.where(anomaly_counts >= 1)[0]
    if suspicious_cells.size == 0:
        return np.zeros(delta_by_cell.shape[1], dtype=np.int64)

    max_abs = float(np.max(np.abs(delta_by_cell)))
    if max_abs == 0:
        return np.zeros(delta_by_cell.shape[1], dtype=np.int64)
    normalized = delta_by_cell / max_abs
    frame_flags = np.any(np.abs(normalized[suspicious_cells, :]) >= normalized_delta_threshold, axis=0)
    return frame_flags.astype(np.int64)



def reconstruct_structured_core_labels(frame_df: pd.DataFrame) -> dict[str, np.ndarray]:
    voltage_columns = sorted([column for column in frame_df.columns if column.startswith('U_')], key=lambda name: int(name.split('_')[1]))
    temperature_columns = sorted([column for column in frame_df.columns if column.startswith('T_')], key=lambda name: int(name.split('_')[1]))
    voltages = frame_df[voltage_columns].to_numpy(dtype=float) if voltage_columns else np.zeros((len(frame_df), 0), dtype=float)
    temperatures = frame_df[temperature_columns].to_numpy(dtype=float) if temperature_columns else np.zeros((len(frame_df), 0), dtype=float)
    charge_status = pd.to_numeric(frame_df['CHARGE_STATUS'], errors='coerce').fillna(0).astype(int).to_numpy(dtype=np.int64)
    insulation = pd.to_numeric(frame_df['INSULATION_RESISTANCE'], errors='coerce').fillna(0.0).to_numpy(dtype=float)

    return {
        'sd': reconstruct_self_discharge_labels(voltages),
        'isc': reconstruct_internal_short_labels(voltages, temperatures),
        'conn': reconstruct_connection_labels(voltages),
        'samp': reconstruct_sampling_labels(voltages, charge_status),
        'ins': reconstruct_insulation_labels(charge_status, insulation),
    }
