import numpy as np

from battery_thesis.rule_reconstruction import _distance_matrix, _sdo_scores


def _naive_distance_matrix(values: np.ndarray, observations: np.ndarray) -> np.ndarray:
    matrix = np.zeros((values.shape[0], observations.shape[0]), dtype=float)
    for i in range(values.shape[0]):
        for j in range(observations.shape[0]):
            matrix[i, j] = abs(values[i, 1] - observations[j, 1])
    return matrix


def _naive_sdo_scores(series: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(series, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    sample_size = len(values)
    if sample_size < 3:
        return np.zeros(sample_size, dtype=float)

    observations = np.column_stack([np.arange(sample_size, dtype=float), values])
    sigma = float(np.std(observations[:, 1]))
    if sample_size <= 1 or sigma <= 0:
        neighbor_count = max(sample_size, 1)
    else:
        numerator = sample_size * (1.96 ** 2) * (sigma ** 2)
        denominator = ((sample_size - 1) * (0.1 * sigma) ** 2) + (sigma ** 2) * (1.96 ** 2)
        neighbor_count = max(int(round(numerator / denominator)), 1) if denominator != 0 else sample_size

    row_sequence = np.arange(sample_size)
    rng = np.random.default_rng(20260407)
    rng.shuffle(row_sequence)
    sampled = observations[row_sequence[: min(neighbor_count, sample_size)], :]

    distance_matrix = _naive_distance_matrix(observations, sampled)
    identifiers = np.zeros((distance_matrix.shape[0], 6), dtype=int)
    matrix = distance_matrix.copy()
    for neighbor_idx in range(6):
        min_indices = matrix.argmin(axis=1)
        identifiers[:, neighbor_idx] = min_indices
        matrix[np.arange(matrix.shape[0]), min_indices] = np.inf

    counts = np.zeros(sampled.shape[0], dtype=float)
    for obs_idx in range(sampled.shape[0]):
        counts[obs_idx] = np.sum(identifiers == obs_idx)
    cutoff = np.percentile(counts, 33)
    keep_mask = counts >= cutoff
    pruned = sampled[keep_mask] if keep_mask.any() else sampled

    pruned_distance_matrix = _naive_distance_matrix(observations, pruned)
    pruned_identifiers = np.zeros((pruned_distance_matrix.shape[0], 6), dtype=int)
    matrix = pruned_distance_matrix.copy()
    for neighbor_idx in range(6):
        min_indices = matrix.argmin(axis=1)
        pruned_identifiers[:, neighbor_idx] = min_indices
        matrix[np.arange(matrix.shape[0]), min_indices] = np.inf

    average = float(np.average(observations[:, 1]))
    scores = []
    for idx in range(sample_size):
        direction = 1 if observations[idx, 1] > average else (-1 if observations[idx, 1] < average else 0)
        total = 0.0
        for neighbor_idx in range(pruned_identifiers.shape[1]):
            total += abs(observations[idx, 1] - pruned[pruned_identifiers[idx, neighbor_idx], 1])
        scores.append(direction * total / pruned_identifiers.shape[1])
    return np.asarray(scores, dtype=float)


def test_distance_matrix_matches_naive_absolute_value_gap():
    values = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, -1.0]], dtype=float)
    observations = np.array([[10.0, 2.0], [11.0, -2.0], [12.0, 5.0]], dtype=float)

    actual = _distance_matrix(values, observations)
    expected = _naive_distance_matrix(values, observations)

    np.testing.assert_allclose(actual, expected)


def test_sdo_scores_match_previous_reference_implementation():
    series = np.array([3.70, 3.69, 3.71, 3.54, 3.53, 3.52, 3.68, 3.69], dtype=float)

    actual = _sdo_scores(series)
    expected = _naive_sdo_scores(series)

    np.testing.assert_allclose(actual, expected)
