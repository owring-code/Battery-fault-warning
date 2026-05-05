from __future__ import annotations

from typing import Iterable


def merge_short_gaps(labels: list[int], max_gap: int) -> list[int]:
    merged = labels[:]
    start = 0
    while start < len(merged):
        if merged[start] != 0:
            start += 1
            continue
        end = start
        while end < len(merged) and merged[end] == 0:
            end += 1
        gap_size = end - start
        has_left = start > 0 and merged[start - 1] == 1
        has_right = end < len(merged) and merged[end] == 1
        if has_left and has_right and gap_size <= max_gap:
            for idx in range(start, end):
                merged[idx] = 1
        start = end
    return merged


def extract_events(labels: list[int], frame_interval_sec: int) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    event_id = 0
    in_event = False
    start_idx = 0
    for idx, label in enumerate(labels):
        if label == 1 and not in_event:
            in_event = True
            start_idx = idx
        if in_event and (label == 0):
            event_id += 1
            end_idx = idx - 1
            events.append(_build_event(event_id, start_idx, end_idx, frame_interval_sec))
            in_event = False
    if in_event:
        event_id += 1
        events.append(_build_event(event_id, start_idx, len(labels) - 1, frame_interval_sec))
    return events


def derive_identification_label(last_three_labels: list[int]) -> int:
    return int(sum(last_three_labels[-3:]) >= 2)


def derive_warning_label(
    current_identification_label: int,
    future_labels: list[int],
    frame_interval_sec: int,
) -> tuple[int, int | None, int | None]:
    if current_identification_label == 1:
        return 0, None, None
    for idx, label in enumerate(future_labels):
        if label == 1:
            return 1, (idx + 1) * frame_interval_sec, idx
    return 0, None, None


def _build_event(event_id: int, start_idx: int, end_idx: int, frame_interval_sec: int) -> dict[str, int]:
    duration_frames = end_idx - start_idx + 1
    return {
        "event_id": event_id,
        "start_frame": start_idx,
        "end_frame": end_idx,
        "duration_frames": duration_frames,
        "duration_seconds": duration_frames * frame_interval_sec,
    }
