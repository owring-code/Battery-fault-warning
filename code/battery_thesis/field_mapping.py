from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np


def parse_sensor_series(raw_value: str | None) -> list[float]:
    if raw_value is None or raw_value == "":
        return []
    return [_safe_numeric_item(item) for item in str(raw_value).split("~") if item != ""]


def normalize_sensor_sequences(sequences: Iterable[list[float] | tuple[float, ...] | np.ndarray]) -> tuple[list[list[float]], np.ndarray, int]:
    prepared: list[list[float]] = []
    for sequence in sequences:
        if sequence is None:
            prepared.append([])
            continue
        prepared.append([_safe_numeric_item(value) for value in sequence])

    non_empty_lengths = [len(sequence) for sequence in prepared if len(sequence) > 0]
    if not non_empty_lengths:
        return [[] for _ in prepared], np.zeros(len(prepared), dtype=bool), 0

    expected_length = Counter(non_empty_lengths).most_common(1)[0][0]
    normalized: list[list[float]] = []
    valid_mask = np.zeros(len(prepared), dtype=bool)
    last_valid = [0.0] * expected_length

    for index, sequence in enumerate(prepared):
        if len(sequence) == expected_length:
            clipped = np.nan_to_num(np.asarray(sequence, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
            normalized.append(clipped)
            valid_mask[index] = True
            last_valid = clipped
            continue

        if len(sequence) > expected_length:
            clipped = np.nan_to_num(np.asarray(sequence[:expected_length], dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
            normalized.append(clipped)
            valid_mask[index] = False
            last_valid = clipped
            continue

        if len(sequence) > 0:
            padded = sequence + [sequence[-1]] * (expected_length - len(sequence))
            padded = np.nan_to_num(np.asarray(padded, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
            normalized.append(padded)
            valid_mask[index] = False
            last_valid = padded
            continue

        normalized.append(list(last_valid))
        valid_mask[index] = False

    return normalized, valid_mask, expected_length


def _collect_prefixed_values(row: dict[str, str], prefix: str) -> list[float]:
    matching_keys = sorted(
        (key for key in row if key.startswith(prefix)),
        key=lambda value: int(value.split("_")[1]),
    )
    return [_safe_numeric_item(row[key]) for key in matching_keys]


def standardize_structured_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "timestamp": int(float(row["TIME"])),
        "charge_status": int(float(row["CHARGE_STATUS"])),
        "speed": _safe_numeric_item(row["SPEED"]),
        "sum_voltage": _safe_numeric_item(row["SUM_VOLTAGE"]),
        "sum_current": _safe_numeric_item(row["SUM_CURRENT"]),
        "soc": _safe_numeric_item(row["SOC"]),
        "insulation_resistance": _safe_numeric_item(row["INSULATION_RESISTANCE"]),
        "voltages": _collect_prefixed_values(row, "U_"),
        "temperatures": _collect_prefixed_values(row, "T_"),
    }


def standardize_raw_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "timestamp": int(float(row["terminaltime"])),
        "charge_status": int(float(row["chargestatus"])),
        "speed": _safe_numeric_item(row["speed"]),
        "sum_voltage": _safe_numeric_item(row["totalvoltage"]),
        "sum_current": _safe_numeric_item(row["totalcurrent"]),
        "soc": _safe_numeric_item(row["soc"]),
        "insulation_resistance": _safe_float(row.get("insulationresistance")),
        "voltages": parse_sensor_series(row.get("batteryvoltage")),
        "temperatures": parse_sensor_series(row.get("probetemperatures")),
        "min_voltage": _safe_float(row.get("minvoltagebattery")),
        "max_voltage": _safe_float(row.get("maxvoltagebattery")),
        "min_temperature": _safe_float(row.get("mintemperaturevalue")),
        "max_temperature": _safe_float(row.get("maxtemperaturevalue")),
    }


def _safe_numeric_item(value: str | float | int | None) -> float:
    number = 0.0 if value in (None, "") else float(value)
    return number if math.isfinite(number) else 0.0


def _safe_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
