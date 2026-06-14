import json
import os
import re
from collections import Counter

import pytest


SEGMENTS_JSON = "/workspace/segments.json"
INPUT_WAV = "/workspace/input.wav"

LABEL_REGEX = re.compile(r"^[A-Z]$")


@pytest.fixture(scope="module")
def segments():
    assert os.path.isfile(SEGMENTS_JSON), (
        f"Expected output file {SEGMENTS_JSON} to exist after the task completes."
    )
    with open(SEGMENTS_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{SEGMENTS_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, list) and len(data) > 0, (
        f"{SEGMENTS_JSON} must be a non-empty JSON array, got: {type(data).__name__} "
        f"with length {len(data) if hasattr(data, '__len__') else 'N/A'}."
    )
    return data


@pytest.fixture(scope="module")
def audio_duration():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    duration = float(len(y)) / float(sr)
    assert duration > 0, f"Reference audio duration must be positive, got {duration}."
    return duration


def test_each_segment_has_required_schema(segments):
    for idx, seg in enumerate(segments):
        assert isinstance(seg, dict), (
            f"Segment {idx} is not a JSON object, got: {type(seg).__name__}."
        )
        for key in ("start", "end", "label"):
            assert key in seg, (
                f"Segment {idx} is missing required key '{key}': {seg!r}."
            )
        assert isinstance(seg["start"], (int, float)) and not isinstance(seg["start"], bool), (
            f"Segment {idx} 'start' must be numeric, got: {seg['start']!r}."
        )
        assert isinstance(seg["end"], (int, float)) and not isinstance(seg["end"], bool), (
            f"Segment {idx} 'end' must be numeric, got: {seg['end']!r}."
        )
        assert isinstance(seg["label"], str) and len(seg["label"]) > 0, (
            f"Segment {idx} 'label' must be a non-empty string, got: {seg['label']!r}."
        )
        assert float(seg["start"]) < float(seg["end"]), (
            f"Segment {idx} must satisfy start < end, got start={seg['start']}, end={seg['end']}."
        )


def test_segment_labels_match_regex(segments):
    for idx, seg in enumerate(segments):
        label = seg["label"]
        assert LABEL_REGEX.match(label), (
            f"Segment {idx} label '{label}' does not match regex ^[A-Z]$ "
            f"(must be a single uppercase letter A-Z)."
        )


def test_segments_sorted_by_start(segments):
    starts = [float(seg["start"]) for seg in segments]
    assert starts == sorted(starts), (
        f"Segments must be sorted by 'start' in ascending order; got starts={starts}."
    )


def test_segments_cover_audio_without_significant_gaps_or_overlaps(segments, audio_duration):
    sorted_segs = sorted(segments, key=lambda s: float(s["start"]))

    first_start = float(sorted_segs[0]["start"])
    assert first_start <= 0.3 + 1e-6, (
        f"First segment must start within 0.3s of 0, got start={first_start}."
    )

    last_end = float(sorted_segs[-1]["end"])
    assert abs(last_end - audio_duration) <= 0.5 + 1e-6, (
        f"Last segment end must be within 0.5s of audio duration {audio_duration}, "
        f"got end={last_end}."
    )

    for i in range(len(sorted_segs) - 1):
        end_i = float(sorted_segs[i]["end"])
        start_next = float(sorted_segs[i + 1]["start"])
        gap = start_next - end_i
        assert gap <= 0.3 + 1e-6, (
            f"Gap between segment {i} (end={end_i}) and segment {i+1} (start={start_next}) "
            f"exceeds 0.3s: gap={gap}."
        )
        overlap = end_i - start_next
        assert overlap <= 0.05 + 1e-6, (
            f"Overlap between segment {i} (end={end_i}) and segment {i+1} (start={start_next}) "
            f"exceeds 0.05s: overlap={overlap}."
        )


def test_each_segment_duration_above_threshold(segments):
    for idx, seg in enumerate(segments):
        duration = float(seg["end"]) - float(seg["start"])
        assert duration > 0.5, (
            f"Segment {idx} duration must be > 0.5s, got {duration} "
            f"(start={seg['start']}, end={seg['end']})."
        )


def test_minimum_segment_and_label_counts(segments):
    assert len(segments) >= 3, (
        f"Expected at least 3 segments in the output, got {len(segments)}."
    )
    unique_labels = {seg["label"] for seg in segments}
    assert len(unique_labels) >= 2, (
        f"Expected at least 2 distinct labels, got {len(unique_labels)}: {sorted(unique_labels)}."
    )


def test_at_least_one_label_is_reused(segments):
    counts = Counter(seg["label"] for seg in segments)
    reused = [lbl for lbl, c in counts.items() if c >= 2]
    assert len(reused) >= 1, (
        f"Expected at least one label to appear in more than one segment "
        f"(proves repetition recovery, not pure temporal partitioning); "
        f"got label counts: {dict(counts)}."
    )
