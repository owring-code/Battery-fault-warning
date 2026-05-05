from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.plot_style import (
    apply_academic_plot_style,
    get_academic_palette,
    style_axis,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "deliverables" / "fault_cases_polished"
LEGACY_INTERNAL_SHORT_IMAGE = (
    PROJECT_ROOT / "assets" / "internal_short_reference.png"
)

CASE_ORDER = [
    "self_discharge_outlier",
    "internal_short_voltage_drop",
    "connection_differential_anomaly",
    "sampling_signal_distortion",
    "insulation_resistance_low",
]


def _case_frame(
    case_id: str,
    series: str,
    x: np.ndarray,
    y: np.ndarray,
    panel: str = "main",
    role: str = "signal",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": case_id,
            "panel": panel,
            "series": series,
            "role": role,
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
        }
    )


def _triangle_wave(x: np.ndarray, period: float, phase: float = 0.0) -> np.ndarray:
    cycle = np.mod(x / period + phase, 1.0)
    return 4.0 * np.abs(cycle - 0.5) - 1.0


def _sawtooth_wave(x: np.ndarray, period: float, phase: float = 0.0) -> np.ndarray:
    return 2.0 * np.mod(x / period + phase, 1.0) - 1.0


def _alternating_samples(length: int, amplitude: float) -> np.ndarray:
    signs = np.where(np.arange(length) % 2 == 0, -1.0, 1.0)
    return amplitude * signs


def build_self_discharge_case() -> pd.DataFrame:
    case_id = "self_discharge_outlier"
    x = np.arange(8, dtype=float) * 10.0
    frames: list[pd.DataFrame] = []
    cell_ids = [50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64]
    for index, cell_id in enumerate(cell_ids):
        baseline = 3625.0 + (index % 6) * 0.65
        wiggle = 0.55 * np.sin((x / 10.0 + index) * np.pi / 2.0)
        y = baseline + wiggle
        frames.append(_case_frame(case_id, f"U_{cell_id}", x, y, role="normal_cell"))
    outlier = np.array([3621.0, 3620.0, 3620.0, 3621.0, 3620.0, 3621.0, 3621.0, 3620.0])
    frames.append(_case_frame(case_id, "U_57", x, outlier, role="fault_cell"))
    return pd.concat(frames, ignore_index=True)


def build_internal_short_case() -> pd.DataFrame:
    case_id = "internal_short_voltage_drop"
    x, y = _vectorize_internal_short_reference_curve()
    return _case_frame(
        case_id, "pack_voltage", x, y, role="vectorized_reference_curve"
    )


def _vectorize_internal_short_reference_curve() -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(LEGACY_INTERNAL_SHORT_IMAGE).convert("RGB")
    pixels = np.asarray(image)
    yy, xx = np.indices(pixels.shape[:2])
    blue_mask = (
        (pixels[:, :, 2] > 130)
        & (pixels[:, :, 0] < 125)
        & (pixels[:, :, 1] < 190)
        & (pixels[:, :, 2] > pixels[:, :, 0] + 35)
        & (pixels[:, :, 2] > pixels[:, :, 1] + 15)
        & (xx >= 120)
        & (xx <= 710)
        & (yy >= 55)
        & (yy <= 505)
    )
    blue_y, blue_x = np.where(blue_mask)
    x_left = float(np.quantile(blue_x, 0.002))
    x_right = float(np.quantile(blue_x, 0.998))
    axis_top, axis_bottom = 47.0, 516.0
    data_low, data_high = 2200.0, 4225.0

    reference_x = (blue_x.astype(float) - x_left) / (x_right - x_left) * 520000.0
    reference_y = data_low + (axis_bottom - blue_y.astype(float)) / (
        axis_bottom - axis_top
    ) * (data_high - data_low)
    reference_x = np.clip(reference_x, 0.0, 520000.0)
    reference_y = _accentuate_internal_short_troughs(reference_y, reference_x)
    order = np.lexsort((reference_y, reference_x))
    return reference_x[order], reference_y[order]


def _accentuate_internal_short_troughs(
    values: np.ndarray, x_positions: np.ndarray | None = None
) -> np.ndarray:
    adjusted = np.asarray(values, dtype=float).copy()
    pivot = 3740.0
    low_mask = adjusted < pivot
    adjusted[low_mask] = pivot - 1.45 * (pivot - adjusted[low_mask])
    if x_positions is not None:
        positions = np.asarray(x_positions, dtype=float)
        broad_shift = np.interp(
            positions,
            [0.0, 60000.0, 120000.0, 190000.0, 260000.0, 330000.0, 390000.0, 455000.0, 520000.0],
            [-105.0, -160.0, -20.0, 145.0, 265.0, 65.0, -45.0, 220.0, 45.0],
        )
        depth_weight = np.clip((pivot - adjusted) / 520.0, 0.0, 1.0)
        adjusted = adjusted - broad_shift * depth_weight
        fault_mask = np.abs(positions - 268650.0) <= 460.0
        non_fault_floor = (
            3250.8
            + np.interp(
                positions,
                [0.0, 120000.0, 220000.0, 280000.0, 360000.0, 520000.0],
                [65.0, 45.0, 0.0, 0.0, 35.0, 55.0],
            )
            + 25.0 * (0.5 + 0.5 * np.sin(positions / 6900.0 + 2.1))
        )
        adjusted[~fault_mask] = np.maximum(
            adjusted[~fault_mask], non_fault_floor[~fault_mask]
        )
    return np.clip(adjusted, 2250.0, 4225.0)


def build_connection_case() -> pd.DataFrame:
    case_id = "connection_differential_anomaly"
    x = np.arange(0.0, 3200001.0, 10.0)
    gap_mask = (x > 2250000.0) & (x < 2380000.0)
    visible_mask = ~gap_mask

    def triangle_from_phase(phase: np.ndarray) -> np.ndarray:
        return 4.0 * np.abs(np.mod(phase, 1.0) - 0.5) - 1.0

    def sawtooth_from_phase(phase: np.ndarray) -> np.ndarray:
        return 2.0 * np.mod(phase, 1.0) - 1.0

    def random_profile(
        rng: np.random.Generator,
        low: float,
        high: float,
        knot_step: float,
    ) -> np.ndarray:
        knots = np.arange(x.min(), x.max() + knot_step, knot_step)
        values = rng.uniform(low, high, len(knots))
        padded_knots = np.r_[x.min() - knot_step, knots, x.max() + knot_step]
        padded_values = np.r_[values[0], values, values[-1]]
        return np.interp(x, padded_knots, padded_values)

    def dense_cell(
        seed: int,
        phase: float,
        offset: float,
        cycle_scale: float,
        texture_scale: float,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        period = random_profile(rng, 22000.0, 52000.0, 260000.0)
        amplitude = random_profile(rng, 0.80, 1.25, 180000.0)
        long_phase = np.cumsum(10.0 / period) + phase
        mid_period = random_profile(rng, 7000.0, 17000.0, 140000.0)
        mid_phase = np.cumsum(10.0 / mid_period) + phase * 1.7
        baseline = 3.78 + offset
        baseline += random_profile(rng, -0.035, 0.035, 360000.0)
        baseline += 0.018 * np.sin(2.0 * np.pi * x / 760000.0 + phase)
        cycles = (
            cycle_scale
            * amplitude
            * (
                0.225 * triangle_from_phase(long_phase)
                + 0.090 * sawtooth_from_phase(mid_phase)
            )
        )
        texture_period = random_profile(rng, 700.0, 1800.0, 90000.0)
        texture_phase = np.cumsum(10.0 / texture_period) + phase * 0.3
        small_period = random_profile(rng, 130.0, 260.0, 110000.0)
        texture = texture_scale * (
            0.026 * triangle_from_phase(texture_phase)
            + 0.014 * np.sin(2.0 * np.pi * x / small_period + phase)
        )
        y = baseline + cycles + texture + rng.normal(0.0, 0.010, size=len(x))
        needle_indices = rng.choice(len(x), size=460, replace=False)
        y[needle_indices] += rng.normal(0.0, 0.085, size=len(needle_indices))
        return y

    upper = np.clip(
        dense_cell(2026041702, 0.15, 0.06, cycle_scale=1.25, texture_scale=1.05),
        3.36,
        4.22,
    )

    differential = dense_cell(
        2026041703, 0.62, -0.08, cycle_scale=0.66, texture_scale=0.72
    )
    anomaly_mask = x >= 1250000.0
    anomaly_x = x[anomaly_mask] - 1250000.0
    differential[anomaly_mask] += 0.025 * (1.0 - np.exp(-anomaly_x / 150000.0))
    rough_mask = (x >= 1450000.0) & (x <= 2190000.0)
    rough_rng = np.random.default_rng(2026041705)
    rough_period = random_profile(rough_rng, 1800.0, 5200.0, 90000.0)
    differential[rough_mask] += 0.010 * np.sin(
        2.0 * np.pi * x[rough_mask] / rough_period[rough_mask]
    )
    differential[rough_mask] += rough_rng.normal(0.0, 0.006, size=int(rough_mask.sum()))
    differential = np.clip(differential, 3.30, 4.12)
    differential[np.argmin(np.abs(x - 1560000.0))] = 2.98
    differential[np.argmin(np.abs(x - 2710000.0))] = 4.12

    lower = np.clip(
        dense_cell(2026041704, 0.38, 0.02, cycle_scale=1.22, texture_scale=1.05),
        3.34,
        4.20,
    )

    return pd.concat(
        [
            _case_frame(
                case_id,
                "U_58",
                x[visible_mask],
                upper[visible_mask],
                panel="upper_cell",
                role="cell_voltage",
            ),
            _case_frame(
                case_id,
                "U_59",
                x[visible_mask],
                differential[visible_mask],
                panel="differential",
                role="fault_signal",
            ),
            _case_frame(
                case_id,
                "U_57",
                x[visible_mask],
                lower[visible_mask],
                panel="lower_cell",
                role="cell_voltage",
            ),
        ],
        ignore_index=True,
    )


def build_sampling_case() -> pd.DataFrame:
    case_id = "sampling_signal_distortion"
    x = np.arange(0.0, 470001.0, 500.0)
    frames: list[pd.DataFrame] = []
    phases = {"U_28": 0.0, "U_27": 1.4, "U_29": 2.2}
    offsets = {"U_28": -7.0, "U_27": 4.0, "U_29": 7.0}
    for series, phase in phases.items():
        y = (
            offsets[series]
            + 9.0 * np.sin(x / 900.0 + phase)
            + 7.0 * np.sin(x / 370.0 + phase / 2.0)
        )
        y += 3.0 * np.sin(x / 95.0 + phase)
        role = "fault_signal" if series == "U_28" else "neighbor_cell"
        frames.append(_case_frame(case_id, series, x, y, role=role))

    data = pd.concat(frames, ignore_index=True)
    spike_points = {185000.0: -47.0, 410000.0: -98.0}
    for spike_x, spike_y in spike_points.items():
        idx = (data["series"] == "U_28") & (np.isclose(data["x"], spike_x))
        data.loc[idx, "y"] = spike_y
    high_idx = (data["series"] == "U_29") & (np.isclose(data["x"], 332500.0))
    data.loc[high_idx, "y"] = 31.0
    return data


def build_insulation_case() -> pd.DataFrame:
    case_id = "insulation_resistance_low"
    x = np.arange(262520.0, 262781.0, 2.0)
    y = 0.765 + 0.0015 * np.sin((x - x.min()) / 2.7)
    drop_center = 262568.0
    drop_mask = (x >= drop_center - 4.0) & (x <= drop_center + 4.0)
    y[drop_mask] = np.interp(
        x[drop_mask],
        [drop_center - 4.0, drop_center, drop_center + 4.0],
        [0.76, 0.0, 0.63],
    )
    recovery_mask = (x > drop_center + 4.0) & (x < drop_center + 20.0)
    y[recovery_mask] = 0.63 + 0.13 * (
        1.0 - np.exp(-(x[recovery_mask] - drop_center - 4.0) / 3.5)
    )
    noisy_mask = (x >= 262655.0) & (x <= 262690.0)
    y[noisy_mask] = 0.84 + 0.025 * np.sin((x[noisy_mask] - 262655.0) / 2.2)
    y[np.argmin(np.abs(x - 262674.0))] = 0.72
    y[np.argmin(np.abs(x - 262682.0))] = 0.73
    y[-8:-1] = np.linspace(0.78, 0.87, 7)

    resistance = _case_frame(case_id, "insulation_resistance", x, y, role="resistance")
    threshold = _case_frame(
        case_id, "safety_threshold", x, np.full_like(x, 0.20), role="threshold"
    )
    return pd.concat([resistance, threshold], ignore_index=True)


def build_all_cases() -> dict[str, pd.DataFrame]:
    return {
        "self_discharge_outlier": build_self_discharge_case(),
        "internal_short_voltage_drop": build_internal_short_case(),
        "connection_differential_anomaly": build_connection_case(),
        "sampling_signal_distortion": build_sampling_case(),
        "insulation_resistance_low": build_insulation_case(),
    }


def _save_figure(fig: plt.Figure, output_root: Path, case_id: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / f"{case_id}.png")
    fig.savefig(output_root / f"{case_id}.svg")
    plt.close(fig)


def _plain_x_axis(ax) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)


def _annotation_color() -> str:
    return get_academic_palette()["coral"]


def plot_self_discharge(frame: pd.DataFrame, output_root: Path) -> None:
    palette = get_academic_palette()
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for series_name, subset in frame.groupby("series"):
        role = str(subset["role"].iloc[0])
        if role == "fault_cell":
            ax.plot(
                subset["x"],
                subset["y"],
                color=palette["coral"],
                linewidth=2.4,
                label=series_name,
            )
        else:
            ax.plot(
                subset["x"],
                subset["y"],
                color=palette["teal_light"],
                linewidth=1.1,
                alpha=0.72,
            )
    style_axis(ax, grid_axis="both")
    ax.set_xlabel("time/s")
    ax.set_ylabel("Cell voltage / mV")
    ax.set_ylim(3619.5, 3644.0)
    normal_proxy = Line2D(
        [0], [0], color=palette["teal_light"], linewidth=1.4, label="Normal cells"
    )
    fault_proxy = Line2D([0], [0], color=palette["coral"], linewidth=2.4, label="U_57")
    ax.legend(handles=[normal_proxy, fault_proxy], loc="upper right")
    plt.tight_layout()
    _save_figure(fig, output_root, "self_discharge_outlier")


def plot_internal_short(frame: pd.DataFrame, output_root: Path) -> None:
    palette = get_academic_palette()
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    if set(frame["role"]) == {"vectorized_reference_curve"}:
        curve = frame.sort_values(["x", "y"])
        ax.plot(
            curve["x"],
            curve["y"],
            color=palette["teal"],
            linewidth=0.42,
            alpha=0.78,
            label="Pack voltage",
        )
    else:
        ax.plot(
            frame["x"],
            frame["y"],
            color=palette["teal"],
            linewidth=0.48,
            alpha=0.92,
            label="Pack voltage",
        )
    style_axis(ax, grid_axis="both")
    ax.set_xlabel("time/s")
    _plain_x_axis(ax)
    ax.set_ylabel("Voltage / mV")
    ax.set_ylim(2200.0, 4180.0)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save_figure(fig, output_root, "internal_short_voltage_drop")


def plot_connection(frame: pd.DataFrame, output_root: Path) -> None:
    palette = get_academic_palette()
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 6.2), sharex=True)
    panel_order = ["upper_cell", "differential", "lower_cell"]
    panel_labels = {
        "upper_cell": "U_58",
        "differential": "U_59",
        "lower_cell": "U_57",
    }
    for ax, panel in zip(axes, panel_order):
        subset = frame[frame["panel"] == panel]
        color = palette["coral"] if panel == "differential" else palette["teal"]
        ax.plot(
            subset["x"],
            subset["y"],
            color=color,
            linewidth=0.44,
            alpha=0.93,
            label=panel_labels[panel],
            rasterized=True,
        )
        style_axis(ax, grid_axis="y")
        ax.set_ylabel("Voltage / mV")
        ax.legend(loc="lower left")
        ax.set_ylim(2.85, 4.45)
    axes[-1].set_xlabel("time/s")
    _plain_x_axis(axes[-1])
    plt.tight_layout()
    _save_figure(fig, output_root, "connection_differential_anomaly")


def plot_sampling(frame: pd.DataFrame, output_root: Path) -> None:
    palette = get_academic_palette()
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    colors = {
        "U_28": palette["coral"],
        "U_27": palette["teal"],
        "U_29": palette["slate"],
    }
    for series_name, subset in frame.groupby("series"):
        ax.plot(
            subset["x"],
            subset["y"],
            color=colors[series_name],
            linewidth=1.2,
            label=series_name,
        )
    style_axis(ax, grid_axis="both")
    ax.set_xlabel("time/s")
    _plain_x_axis(ax)
    ax.set_ylabel("Voltage / mV")
    ax.set_ylim(-105.0, 35.0)
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save_figure(fig, output_root, "sampling_signal_distortion")


def plot_insulation(frame: pd.DataFrame, output_root: Path) -> None:
    palette = get_academic_palette()
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    resistance = frame[frame["series"] == "insulation_resistance"]
    threshold = frame[frame["series"] == "safety_threshold"]
    ax.plot(
        resistance["x"],
        resistance["y"],
        color=palette["teal"],
        linewidth=2.0,
        label="Insulation resistance",
    )
    ax.plot(
        threshold["x"],
        threshold["y"],
        color=palette["coral"],
        linestyle="--",
        linewidth=2.0,
        label="Safety threshold",
    )
    style_axis(ax, grid_axis="both")
    ax.set_xlabel("time/s")
    _plain_x_axis(ax)
    ax.set_ylabel("Insulation resistance")
    ax.set_ylim(-0.2, 1.0)
    ax.legend(loc="lower left")
    plt.tight_layout()
    _save_figure(fig, output_root, "insulation_resistance_low")


PLOTTERS: dict[str, Callable[[pd.DataFrame, Path], None]] = {
    "self_discharge_outlier": plot_self_discharge,
    "internal_short_voltage_drop": plot_internal_short,
    "connection_differential_anomaly": plot_connection,
    "sampling_signal_distortion": plot_sampling,
    "insulation_resistance_low": plot_insulation,
}


def write_regeneration_wrapper(path: Path, case_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from __future__ import annotations\n\n"
        "from scripts.export_fault_case_polished_figures import main\n\n\n"
        "if __name__ == '__main__':\n"
        f"    main(['--case', '{case_id}'])\n",
        encoding="utf-8",
    )


def export_cases(output_root: Path, selected_case: str | None = None) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    apply_academic_plot_style()
    cases = build_all_cases()
    exported: list[Path] = []
    selected = [selected_case] if selected_case else CASE_ORDER
    for case_id in selected:
        frame = cases[case_id]
        data_path = output_root / f"{case_id}_data.csv"
        frame.to_csv(data_path, index=False, encoding="utf-8-sig")
        PLOTTERS[case_id](frame, output_root)
        script_path = output_root / f"{case_id}.py"
        write_regeneration_wrapper(script_path, case_id)
        exported.extend(
            [
                data_path,
                output_root / f"{case_id}.png",
                output_root / f"{case_id}.svg",
                script_path,
            ]
        )
    return exported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export polished legacy fault-case figures."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case", choices=CASE_ORDER)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    export_cases(output_root, args.case)


if __name__ == "__main__":
    main()
