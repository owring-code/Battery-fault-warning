from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from scripts.export_chapter5_simulated_supplements import (
    _export_confusion_rate_heatmap,
    _export_warning_lead_distribution,
    _export_warning_tradeoff,
)


RESULTS_ROOT = Path("results/round2_simulated_polished")
SAMPLES_ROOT = RESULTS_ROOT / "_polished_samples"
FAULT_ORDER = ["sd", "isc", "ins", "samp", "conn"]


pytestmark = pytest.mark.skipif(
    not (RESULTS_ROOT / "tables" / "per_fault_type_metrics.csv").exists(),
    reason="requires generated simulated result tables, which are not committed",
)


def test_confusion_heatmap_matches_per_fault_metric_table(
    tmp_path: Path,
) -> None:
    exported = _export_confusion_rate_heatmap(
        RESULTS_ROOT, SAMPLES_ROOT, tmp_path
    ).sort_values(["fault_type", "metric"]).reset_index(drop=True)

    metrics = pd.read_csv(RESULTS_ROOT / "tables" / "per_fault_type_metrics.csv")
    expected_rows = []
    for fault in FAULT_ORDER:
        row = metrics.loc[metrics["fault_type"] == fault].iloc[0]
        recall = float(row["ID-Recall"])
        fpr = float(row["FPR"])
        expected_rows.extend(
            [
                {"fault_type": fault, "metric": "TPR", "rate": recall},
                {"fault_type": fault, "metric": "FNR", "rate": 1.0 - recall},
                {"fault_type": fault, "metric": "FPR", "rate": fpr},
                {"fault_type": fault, "metric": "TNR", "rate": 1.0 - fpr},
            ]
        )
    expected = (
        pd.DataFrame(expected_rows)
        .sort_values(["fault_type", "metric"])
        .reset_index(drop=True)
    )

    assert_series_equal(exported["rate"], expected["rate"], check_names=False)


def test_warning_tradeoff_uses_all_five_fault_average(tmp_path: Path) -> None:
    exported = _export_warning_tradeoff(RESULTS_ROOT, tmp_path).set_index("model_key")
    expected = pd.read_csv(
        RESULTS_ROOT / "tables" / "warning_model_comparison_all.csv"
    ).set_index("model_key")

    assert "mean_all_warning_recall" in exported.columns
    assert "mean_all_false_alarm_rate" in exported.columns

    assert_series_equal(
        exported["mean_all_warning_recall"].sort_index(),
        expected["mean_all_warning_recall"].sort_index(),
        check_names=False,
    )
    assert_series_equal(
        exported["mean_all_false_alarm_rate"].sort_index(),
        expected["mean_all_false_alarm_rate"].sort_index(),
        check_names=False,
    )

    svg_text = (tmp_path / "fig5_6_warning_recall_far_tradeoff.svg").read_text(
        encoding="utf-8"
    )
    assert "Times New Roman" in svg_text


def test_warning_lead_distribution_uses_full_scale_calibrated_samples(
    tmp_path: Path,
) -> None:
    exported = _export_warning_lead_distribution(
        RESULTS_ROOT, SAMPLES_ROOT, tmp_path
    ).sort_values(["fault_type", "sample_id"]).reset_index(drop=True)

    summary = pd.read_csv(SAMPLES_ROOT / "full_scale_summary.csv")
    test_row = summary.loc[summary["split"] == "test"].iloc[0]
    metrics = pd.read_csv(RESULTS_ROOT / "tables" / "per_fault_type_metrics.csv")

    assert set(exported["fault_type"]) == set(FAULT_ORDER)
    for fault in FAULT_ORDER:
        metric_row = metrics.loc[metrics["fault_type"] == fault].iloc[0]
        expected_n = round(
            int(test_row[f"y_warn_{fault}"]) * float(metric_row["Warn-Recall"])
        )
        actual = exported.loc[exported["fault_type"] == fault, "lead_time_sec"]

        assert len(actual) == expected_n
        assert abs(actual.mean() - float(metric_row["Mean Lead Time"])) < 0.001

    svg_text = (tmp_path / "fig5_7_warning_lead_time_distribution.svg").read_text(
        encoding="utf-8"
    )
    assert "Times New Roman" in svg_text
