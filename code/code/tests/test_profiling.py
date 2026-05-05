from __future__ import annotations

import pandas as pd

from battery_thesis.profiling import _series_to_epoch_seconds


def test_series_to_epoch_seconds_accepts_numeric_with_sparse_missing_values():
    series = pd.Series([38.0, None, 58.0, 68.0], dtype=object)

    values = _series_to_epoch_seconds(series)

    assert values == [38, 48, 58, 68]


def test_series_to_epoch_seconds_rejects_non_numeric_non_datetime_tokens():
    series = pd.Series(['2024-01-01 00:00:00', 'bad-token', '2024-01-01 00:00:20'])

    try:
        _series_to_epoch_seconds(series)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError('Expected invalid time token to be rejected.')

    assert 'Unsupported time column format' in message
