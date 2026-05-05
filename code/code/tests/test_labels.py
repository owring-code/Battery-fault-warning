from battery_thesis.labels import (
    derive_identification_label,
    derive_warning_label,
    extract_events,
    merge_short_gaps,
)


def test_merge_short_gaps_closes_small_internal_breaks():
    merged = merge_short_gaps([1, 1, 0, 1, 1], max_gap=1)

    assert merged == [1, 1, 1, 1, 1]


def test_extract_events_returns_start_end_and_duration():
    events = extract_events([0, 1, 1, 0, 1], frame_interval_sec=10)

    assert events == [
        {"event_id": 1, "start_frame": 1, "end_frame": 2, "duration_frames": 2, "duration_seconds": 20},
        {"event_id": 2, "start_frame": 4, "end_frame": 4, "duration_frames": 1, "duration_seconds": 10},
    ]


def test_identification_label_uses_last_three_frame_majority():
    assert derive_identification_label([0, 1, 1]) == 1
    assert derive_identification_label([0, 0, 1]) == 0


def test_warning_label_detects_first_fault_within_horizon():
    label, lead_time, first_index = derive_warning_label(
        current_identification_label=0,
        future_labels=[0, 0, 1, 1],
        frame_interval_sec=10,
    )

    assert label == 1
    assert lead_time == 30
    assert first_index == 2


def test_warning_label_excludes_current_positive_state():
    label, lead_time, first_index = derive_warning_label(
        current_identification_label=1,
        future_labels=[0, 1, 1],
        frame_interval_sec=10,
    )

    assert label == 0
    assert lead_time is None
    assert first_index is None
