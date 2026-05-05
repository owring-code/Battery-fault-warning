from pathlib import Path

import pandas as pd

from battery_thesis.metadata import scan_project_datasets


def test_scan_project_datasets_classifies_structured_and_raw_sources(tmp_path: Path):
    structured_root = tmp_path / 'Data_set'
    raw_root = tmp_path / 'RAW_DATA'
    structured_root.mkdir()
    raw_root.mkdir()

    (structured_root / 'WCVT1000000000099.csv').write_text('TIME\n1\n', encoding='utf-8')
    (raw_root / 'vin1').mkdir()
    (raw_root / 'vin1' / 'vin1.csv').write_text('terminaltime\n48\n', encoding='utf-8')
    (raw_root / 'vin2.rar').write_text('placeholder', encoding='utf-8')

    scanned = scan_project_datasets(structured_root=structured_root, raw_root=raw_root)

    assert set(scanned['source_dataset']) == {'structured_dataset', 'raw_dataset'}
    raw_rows = scanned[scanned['source_dataset'] == 'raw_dataset']
    assert raw_rows['is_extracted'].sum() == 1
    assert scanned['source_file'].str.endswith('.csv').sum() == 2



def test_scan_project_datasets_supports_flat_raw_csv_layout(tmp_path: Path):
    structured_root = tmp_path / 'Data_set'
    raw_root = tmp_path / 'RAW_DATA'
    structured_root.mkdir()
    raw_root.mkdir()

    (structured_root / 'WCVT1000000000099.csv').write_text('TIME\n1\n', encoding='utf-8')
    (raw_root / 'vin_flat.csv').write_text('terminaltime\n48\n', encoding='utf-8')

    scanned = scan_project_datasets(structured_root=structured_root, raw_root=raw_root)

    raw_rows = scanned[scanned['source_dataset'] == 'raw_dataset']
    assert raw_rows['vehicle_id'].tolist() == ['vin_flat']
    assert raw_rows['is_extracted'].tolist() == [True]
    assert raw_rows['entry_kind'].tolist() == ['csv']
    assert raw_rows['source_file'].str.endswith('vin_flat.csv').all()
