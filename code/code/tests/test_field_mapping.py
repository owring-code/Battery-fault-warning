from pathlib import Path

import pytest

from battery_thesis.field_mapping import (
    parse_sensor_series,
    standardize_raw_row,
    standardize_structured_row,
)


def test_parse_sensor_series_parses_tilde_delimited_values():
    assert parse_sensor_series("3.1~3.2~3.3") == [3.1, 3.2, 3.3]


def test_standardize_structured_row_maps_core_fields():
    row = {
        "TIME": "100",
        "CHARGE_STATUS": "3",
        "SPEED": "10.5",
        "SUM_VOLTAGE": "350.5",
        "SUM_CURRENT": "12.3",
        "SOC": "65",
        "INSULATION_RESISTANCE": "5000",
        "U_1": "3.51",
        "U_2": "3.52",
        "T_1": "28",
        "T_2": "29",
    }

    standardized = standardize_structured_row(row)

    assert standardized["timestamp"] == 100
    assert standardized["charge_status"] == 3
    assert standardized["speed"] == pytest.approx(10.5)
    assert standardized["sum_voltage"] == pytest.approx(350.5)
    assert standardized["voltages"] == [3.51, 3.52]
    assert standardized["temperatures"] == [28.0, 29.0]


def test_standardize_raw_row_maps_raw_columns_and_series():
    row = {
        "terminaltime": "48.0",
        "chargestatus": "3",
        "speed": "0.0",
        "totalvoltage": "349.4",
        "totalcurrent": "1.3",
        "soc": "34",
        "batteryvoltage": "3.643~3.644~3.645",
        "probetemperatures": "33.0~34.0~35.0",
        "minvoltagebattery": "3.634",
        "maxvoltagebattery": "3.645",
        "mintemperaturevalue": "31",
        "maxtemperaturevalue": "35",
    }

    standardized = standardize_raw_row(row)

    assert standardized["timestamp"] == 48
    assert standardized["charge_status"] == 3
    assert standardized["sum_voltage"] == pytest.approx(349.4)
    assert standardized["voltages"] == [3.643, 3.644, 3.645]
    assert standardized["temperatures"] == [33.0, 34.0, 35.0]
