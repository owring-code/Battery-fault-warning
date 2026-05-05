from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_thesis.config import ensure_project_directories
from battery_thesis.samples import save_sample_artifacts
from scripts.build_samples import (
    DEFAULT_RAW_ROOT,
    collect_external_validation_sample_artifacts,
    parse_selected_vehicle_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build RAW_DATA-only external validation samples without Data_set source files.')
    parser.add_argument('--samples-root', required=True)
    parser.add_argument('--raw-root', default=str(DEFAULT_RAW_ROOT))
    parser.add_argument('--vehicle-ids', default=None, help='Optional comma-separated RAW vehicle ids for partial verification runs.')
    parser.add_argument('--label-kind', choices=['final', 'raw'], default='final')
    parser.add_argument('--ignore-quality-flag', action='store_true')
    args = parser.parse_args()

    ensure_project_directories()
    output_root = Path(args.samples_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_vehicle_ids = parse_selected_vehicle_ids(args.vehicle_ids)

    samples_master, features_all, tensors, split_summary, selection_summary = collect_external_validation_sample_artifacts(
        label_kind=args.label_kind,
        ignore_quality_flag=args.ignore_quality_flag,
        raw_root=Path(args.raw_root),
        selected_vehicle_ids=selected_vehicle_ids,
    )
    if samples_master.empty:
        raise ValueError('No RAW_DATA external validation samples were materialized. Check labels/raw source paths and vehicle ids.')

    paths = save_sample_artifacts(output_root, samples_master, features_all, tensors)
    split_summary.to_csv(output_root / 'external_validation_summary.csv', index=False, encoding='utf-8-sig')
    selection_summary.to_csv(output_root / 'external_validation_selection.csv', index=False, encoding='utf-8-sig')
    notes = '\n'.join([
        '# External Validation Build Notes',
        '',
        f'- label_kind: {args.label_kind}',
        f'- ignore_quality_flag: {args.ignore_quality_flag}',
        '- source_dataset: raw_dataset only',
        '- split: external_test only',
        '- This bundle is intended to pair with an existing round2 training sample bundle.',
    ])
    (output_root / 'external_validation_notes.md').write_text(notes + '\n', encoding='utf-8')

    print(f"Saved samples to: {paths['samples_master']}")
    print(f"Saved features to: {paths['features_all']}")
    print(f"Saved tensor pack to: {paths['dataset_pack']}")
    print(f'Sample count: {len(samples_master)}')


if __name__ == '__main__':
    main()
