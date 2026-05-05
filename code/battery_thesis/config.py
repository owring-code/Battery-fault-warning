from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
RESULTS_ROOT = PROJECT_ROOT / "results"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PLOTTING_ROOT = SCRIPTS_ROOT / "plotting"

RAW_LABELS_ROOT = ARTIFACTS_ROOT / "labels" / "raw_dataset"
STRUCTURED_LABELS_ROOT = ARTIFACTS_ROOT / "labels" / "structured_dataset"
FINAL_LABELS_ROOT = ARTIFACTS_ROOT / "labels" / "final"
SAMPLES_ROOT = ARTIFACTS_ROOT / "samples"
TENSORS_ROOT = ARTIFACTS_ROOT / "tensors"

DATASET_META_ROOT = ARTIFACTS_ROOT / "dataset_meta"
DELIVERABLES_ROOT = PROJECT_ROOT / "deliverables"

FRAME_INTERVAL_SECONDS = 10
MAX_SEGMENT_GAP_SECONDS = 20
WINDOW_SIZE_FRAMES = 30
STEP_SIZE_FRAMES = 3
FORECAST_HORIZON_SECONDS = 60
FORECAST_HORIZON_FRAMES = FORECAST_HORIZON_SECONDS // FRAME_INTERVAL_SECONDS

FAULT_ORDER = ["sd", "isc", "conn", "samp", "ins"]
FAULT_LABEL_COLUMNS = {
    "sd": "label_final_sd",
    "isc": "label_final_isc",
    "conn": "label_final_conn",
    "samp": "label_final_samp",
    "ins": "label_final_ins",
}

FAULT_NAME_MAP = {
    "sd": "abnormal_self_discharge",
    "isc": "sudden_short_circuit",
    "conn": "connection_exception",
    "samp": "sampling_exception",
    "ins": "failure_of_insulation",
}


def ensure_project_directories() -> list[Path]:
    paths = [
        ARTIFACTS_ROOT,
        RESULTS_ROOT,
        PLOTTING_ROOT,
        RAW_LABELS_ROOT / "per_vehicle",
        RAW_LABELS_ROOT / "events",
        STRUCTURED_LABELS_ROOT / "per_vehicle",
        STRUCTURED_LABELS_ROOT / "events",
        FINAL_LABELS_ROOT / "per_vehicle",
        FINAL_LABELS_ROOT / "events",
        SAMPLES_ROOT,
        TENSORS_ROOT,
        DATASET_META_ROOT,
        DELIVERABLES_ROOT,
        RESULTS_ROOT / "recognition",
        RESULTS_ROOT / "warning",
        RESULTS_ROOT / "ablation",
        RESULTS_ROOT / "external_validation",
        RESULTS_ROOT / "figures",
        RESULTS_ROOT / "tables",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths
