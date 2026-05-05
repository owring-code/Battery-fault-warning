from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.export_fault_case_polished_figures import (
    build_all_cases,
    export_cases,
    write_regeneration_wrapper,
)


def _median_adjacent_abs_delta(series: pd.Series) -> float:
    return float(series.dropna().diff().abs().dropna().median())


def _max_lagged_correlation(series: pd.Series, lags: list[int]) -> float:
    values = series.dropna().to_numpy(dtype=float)
    values = values - values.mean()
    correlations = []
    for lag in lags:
        left = values[:-lag]
        right = values[lag:]
        denominator = (left @ left) ** 0.5 * (right @ right) ** 0.5
        correlations.append(float((left @ right) / denominator))
    return max(correlations)


def _direction_reversal_rate(series: pd.Series) -> float:
    deltas = series.dropna().diff().dropna()
    signs = (
        deltas.where(deltas == 0.0, deltas / deltas.abs()).replace(0.0, pd.NA).dropna()
    )
    return float((signs.diff().dropna() != 0.0).mean())


def _chunk_range_cv(series: pd.Series, chunk_size: int) -> float:
    values = series.dropna().to_numpy(dtype=float)
    ranges = []
    for start in range(0, len(values) - chunk_size + 1, chunk_size):
        chunk = values[start : start + chunk_size]
        ranges.append(float(pd.Series(chunk).quantile(0.95) - pd.Series(chunk).quantile(0.05)))
    range_series = pd.Series(ranges)
    return float(range_series.std() / range_series.mean())


def test_build_all_cases_preserves_legacy_anomaly_patterns() -> None:
    cases = build_all_cases()

    assert set(cases) == {
        "self_discharge_outlier",
        "internal_short_voltage_drop",
        "connection_differential_anomaly",
        "sampling_signal_distortion",
        "insulation_resistance_low",
    }

    for name, frame in cases.items():
        assert isinstance(frame, pd.DataFrame), name
        assert not frame.empty, name
        assert {"case_id", "series", "x", "y"}.issubset(frame.columns), name

    sd = cases["self_discharge_outlier"]
    outlier_min = sd.loc[sd["series"] == "U_57", "y"].min()
    normal_median = sd.loc[sd["series"] != "U_57", "y"].median()
    assert outlier_min < normal_median - 4.0

    isc = cases["internal_short_voltage_drop"]
    assert isc.loc[isc["series"] == "pack_voltage", "y"].max() > 4000.0
    assert isc.loc[isc["series"] == "pack_voltage", "y"].min() < 2400.0

    conn = cases["connection_differential_anomaly"]
    assert set(conn["panel"]) == {"upper_cell", "differential", "lower_cell"}
    differential = conn.loc[conn["panel"] == "differential", "y"]
    assert differential.max() - differential.min() > 1.0

    sampling = cases["sampling_signal_distortion"]
    assert sampling["y"].min() < -80.0
    assert sampling["y"].max() > 20.0
    sampling_points_per_series = sampling.groupby("series").size().min()
    assert isc.groupby("series").size().min() >= 50_000
    assert conn.groupby("series").size().min() >= 300_000
    assert isc.groupby("series").size().min() >= sampling_points_per_series
    assert conn.groupby("series").size().min() >= sampling_points_per_series
    if set(isc["role"]) == {"vectorized_reference_curve"}:
        assert isc["x"].nunique() > 500
    else:
        assert isc.sort_values("x")["x"].diff().dropna().median() == 10.0
    for _, subset in conn.groupby("series"):
        assert subset.sort_values("x")["x"].diff().dropna().median() == 10.0

    insulation = cases["insulation_resistance_low"]
    resistance = insulation.loc[insulation["series"] == "insulation_resistance", "y"]
    threshold = insulation.loc[insulation["series"] == "safety_threshold", "y"].iloc[0]
    assert resistance.min() < threshold


def test_internal_short_and_connection_cases_have_dense_high_frequency_motion() -> None:
    cases = build_all_cases()

    isc = cases["internal_short_voltage_drop"].sort_values("x")
    voltage = isc.loc[isc["series"] == "pack_voltage"]
    assert set(voltage["role"]) == {"vectorized_reference_curve"}
    assert len(voltage) > 50_000
    assert voltage["x"].nunique() > 500
    assert voltage["y"].max() > 4100.0
    assert voltage["y"].min() < 2400.0
    grouped = voltage.groupby("x")["y"]
    assert (grouped.max() - grouped.min()).max() > 700.0
    non_drop_troughs = grouped.min()
    assert int((non_drop_troughs < 3250.0).sum()) == 1
    non_fault_troughs = non_drop_troughs[
        (non_drop_troughs.index < 266000.0) | (non_drop_troughs.index > 274000.0)
    ]
    assert non_fault_troughs.min() >= 3250.0
    assert int((non_fault_troughs.round(3) == 3250.0).sum()) < 20
    non_drop_troughs = non_drop_troughs[non_drop_troughs > 3000.0]
    assert non_drop_troughs.quantile(0.75) - non_drop_troughs.quantile(0.25) > 190.0
    binned_troughs = non_drop_troughs.reset_index()
    binned_troughs["bin"] = pd.cut(
        binned_troughs["x"], bins=8, labels=False, include_lowest=True
    )
    bin_medians = binned_troughs.groupby("bin")["y"].median()
    assert bin_medians.max() - bin_medians.min() > 330.0

    conn = cases["connection_differential_anomaly"].sort_values(["series", "x"])
    for series_name, subset in conn.groupby("series"):
        voltage = subset["y"]
        delta = _median_adjacent_abs_delta(voltage)
        assert 0.004 < delta < 0.085, series_name
        rate = _direction_reversal_rate(voltage)
        assert 0.40 < rate < 0.90, series_name


def test_internal_short_export_is_drawn_as_data_line(tmp_path: Path) -> None:
    export_cases(tmp_path, selected_case="internal_short_voltage_drop")

    svg_text = (tmp_path / "internal_short_voltage_drop.svg").read_text(encoding="utf-8")
    assert "<image" not in svg_text
    assert "stroke: #00a087; stroke-opacity: 0.78; stroke-width: 0.42" in svg_text


def test_connection_case_avoids_overly_regular_periodic_waveforms() -> None:
    conn = build_all_cases()["connection_differential_anomaly"]
    candidate_lags = [900, 1100, 1800, 2600, 3200, 4200]

    for series_name, subset in conn.groupby("series"):
        first_segment = subset.sort_values("x").iloc[:120000]["y"]
        assert (
            _max_lagged_correlation(first_segment, candidate_lags) < 0.82
        ), series_name


def test_connection_case_keeps_bridge_gap_and_red_band_slightly_lower() -> None:
    conn = build_all_cases()["connection_differential_anomaly"]

    for series_name, subset in conn.groupby("series"):
        ordered = subset.sort_values("x")
        assert not ordered["y"].isna().any(), series_name
        assert ordered["x"].diff().max() > 10.0, series_name

    medians = conn.groupby("series")["y"].median()
    green_center = (medians["U_57"] + medians["U_58"]) / 2.0
    assert medians["U_59"] < green_center - 0.04


def test_connection_anomaly_band_has_smaller_red_range_than_green_cells() -> None:
    conn = build_all_cases()["connection_differential_anomaly"]
    anomaly = conn[(conn["x"] >= 1450000.0) & (conn["x"] <= 2190000.0)]
    ranges = anomaly.groupby("series")["y"].quantile(0.95) - anomaly.groupby("series")[
        "y"
    ].quantile(0.05)

    red_range = ranges["U_59"]
    assert 0.25 < red_range < 0.43
    assert ranges["U_57"] > red_range + 0.15
    assert ranges["U_58"] > red_range + 0.15


def test_write_regeneration_wrapper_mentions_target_case(tmp_path: Path) -> None:
    wrapper = tmp_path / "case.py"
    write_regeneration_wrapper(wrapper, "self_discharge_outlier")

    text = wrapper.read_text(encoding="utf-8")
    assert "self_discharge_outlier" in text
    assert "export_fault_case_polished_figures" in text


def test_exported_figures_use_plain_time_axis_and_no_case_annotations(
    tmp_path: Path,
) -> None:
    export_cases(tmp_path)

    forbidden_labels = [
        "Outlier",
        "Voltage drop",
        "Differential anomaly",
        "Signal distortion",
        "Below threshold",
        "离群现象",
        "电压突降",
        "压差异常",
        "数据失真",
        "电阻低于阈值",
        "Time interval",
    ]

    for svg_path in tmp_path.glob("*.svg"):
        text = svg_path.read_text(encoding="utf-8")
        assert "time/s" in text, svg_path.name
        for label in forbidden_labels:
            assert (
                label not in text
            ), f"{label!r} unexpectedly present in {svg_path.name}"
