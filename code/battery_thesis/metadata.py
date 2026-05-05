from __future__ import annotations

from pathlib import Path

import pandas as pd


def scan_project_datasets(structured_root: Path, raw_root: Path | None) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for csv_path in sorted(structured_root.glob('*.csv')):
        records.append(
            {
                'vehicle_id': csv_path.stem,
                'source_dataset': 'structured_dataset',
                'source_file': str(csv_path),
                'is_extracted': True,
                'entry_kind': 'csv',
            }
        )

    if raw_root is not None and raw_root.exists():
        for entry in sorted(raw_root.iterdir()):
            if entry.is_dir():
                csv_files = sorted(entry.glob('*.csv'))
                source_file = str(csv_files[0]) if csv_files else str(entry)
                records.append(
                    {
                        'vehicle_id': entry.name,
                        'source_dataset': 'raw_dataset',
                        'source_file': source_file,
                        'is_extracted': bool(csv_files),
                        'entry_kind': 'directory',
                    }
                )
            elif entry.suffix.lower() == '.csv':
                records.append(
                    {
                        'vehicle_id': entry.stem,
                        'source_dataset': 'raw_dataset',
                        'source_file': str(entry),
                        'is_extracted': True,
                        'entry_kind': 'csv',
                    }
                )
            elif entry.suffix.lower() == '.rar':
                records.append(
                    {
                        'vehicle_id': entry.stem,
                        'source_dataset': 'raw_dataset',
                        'source_file': str(entry),
                        'is_extracted': False,
                        'entry_kind': 'archive',
                    }
                )

    return pd.DataFrame(records)
