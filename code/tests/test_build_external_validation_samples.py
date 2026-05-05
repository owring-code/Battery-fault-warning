from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.build_samples as build_samples


FAULTS = ['sd', 'isc', 'conn', 'samp', 'ins']


def _write_raw_csv(csv_path: Path, rows: int = 40) -> None:
    records = []
    for idx in range(rows):
        voltage_values = [3.70, 3.69, 3.68, 3.67, 3.66]
        records.append(
            {
                'terminaltime': 1 + idx * 10,
                'soc': 50,
                'speed': 0.0,
                'chargestatus': 3,
                'totalvoltage': sum(voltage_values),
                'totalcurrent': 0.0,
                'batteryvoltage': '~'.join(str(value) for value in voltage_values),
                'probetemperatures': '30~31',
                'insulationresistance': 5000.0,
            }
        )
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding='utf-8')


def _write_raw_labels(label_path: Path, rows: int = 40) -> None:
    data = {
        'timestamp': [1 + idx * 10 for idx in range(rows)],
        'frame_index': list(range(rows)),
        'segment_id': [1] * rows,
        'quality_flag': ['ok'] * rows,
    }
    for fault in FAULTS:
        data[f'label_raw_{fault}'] = [0] * rows
        data[f'label_final_{fault}'] = [0] * rows
    pd.DataFrame(data).to_csv(label_path, index=False, encoding='utf-8-sig')


def test_collect_external_validation_sample_artifacts_uses_raw_rows_only(tmp_path, monkeypatch):
    labels_root = tmp_path / 'labels' / 'final'
    per_vehicle = labels_root / 'per_vehicle'
    per_vehicle.mkdir(parents=True)
    raw_root = tmp_path / 'raw'
    raw_root.mkdir()

    pd.DataFrame(
        [
            {'vehicle_id': 'raw_flat', 'source_dataset': 'raw_dataset', 'total_rows': 40},
            {'vehicle_id': 'WCVT100', 'source_dataset': 'structured_dataset', 'total_rows': 40},
        ]
    ).to_csv(labels_root / 'summary.csv', index=False, encoding='utf-8-sig')

    _write_raw_csv(raw_root / 'raw_flat.csv')
    _write_raw_labels(per_vehicle / 'raw_flat_labels.csv')

    monkeypatch.setattr(build_samples, 'FINAL_LABELS_ROOT', labels_root)

    samples_master, features_all, tensors, split_summary, selection_summary = build_samples.collect_external_validation_sample_artifacts(
        label_kind='final',
        ignore_quality_flag=False,
        raw_root=raw_root,
        selected_vehicle_ids=None,
    )

    assert not samples_master.empty
    assert set(samples_master['source_dataset']) == {'raw_dataset'}
    assert set(samples_master['split']) == {'external_test'}
    assert len(features_all) == len(samples_master)
    assert tensors['X_seq'].shape[0] == len(samples_master)
    assert split_summary['split'].tolist() == ['external_test']
    assert selection_summary['vehicle_id'].tolist() == ['raw_flat']
