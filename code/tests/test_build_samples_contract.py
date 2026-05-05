from pathlib import Path

from scripts.build_samples import resolve_label_path
from battery_thesis.config import FINAL_LABELS_ROOT


def test_build_samples_reads_labels_from_final_root():
    path = resolve_label_path('WCVT1000000000099', 'structured_dataset')
    assert path == FINAL_LABELS_ROOT / 'per_vehicle' / 'WCVT1000000000099_labels.csv'
