import json
import os
import re

import pytest


CHORDS_JSON = "/workspace/chords.json"
INPUT_WAV = "/workspace/input.wav"

CHORD_REGEX = re.compile(r"^[A-G]#?:(maj|min)$")

ALLOWED_CHORDS = {
    "C:maj", "C#:maj", "D:maj", "D#:maj", "E:maj", "F:maj",
    "F#:maj", "G:maj", "G#:maj", "A:maj", "A#:maj", "B:maj",
    "C:min", "C#:min", "D:min", "D#:min", "E:min", "F:min",
    "F#:min", "G:min", "G#:min", "A:min", "A#:min", "B:min",
}


@pytest.fixture(scope="module")
def segments():
    assert os.path.isfile(CHORDS_JSON), (
        f"Expected output file {CHORDS_JSON} to exist after the task completes."
    )
    with open(CHORDS_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{CHORDS_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, list) and len(data) > 0, (
        f"{CHORDS_JSON} must be a non-empty JSON array, got: {type(data).__name__} "
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
        for key in ("start", "end", "chord"):
            assert key in seg, (
                f"Segment {idx} is missing required key '{key}': {seg!r}."
            )
        assert isinstance(seg["start"], (int, float)) and not isinstance(seg["start"], bool), (
            f"Segment {idx} 'start' must be numeric, got: {seg['start']!r}."
        )
        assert isinstance(seg["end"], (int, float)) and not isinstance(seg["end"], bool), (
            f"Segment {idx} 'end' must be numeric, got: {seg['end']!r}."
        )
        assert isinstance(seg["chord"], str), (
            f"Segment {idx} 'chord' must be a string, got: {seg['chord']!r}."
        )
        assert float(seg["start"]) < float(seg["end"]), (
            f"Segment {idx} must satisfy start < end, got start={seg['start']}, end={seg['end']}."
        )


def test_chord_labels_match_regex_and_allowed_set(segments):
    for idx, seg in enumerate(segments):
        chord = seg["chord"]
        assert CHORD_REGEX.match(chord), (
            f"Segment {idx} chord label '{chord}' does not match regex ^[A-G]#?:(maj|min)$."
        )
        assert chord in ALLOWED_CHORDS, (
            f"Segment {idx} chord label '{chord}' is not in the 24-allowed-label set."
        )


def test_segments_cover_audio_without_significant_gaps_or_overlaps(segments, audio_duration):
    sorted_segs = sorted(segments, key=lambda s: float(s["start"]))

    first_start = float(sorted_segs[0]["start"])
    assert first_start <= 0.2 + 1e-6, (
        f"First segment must start within 0.2s of 0, got start={first_start}."
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
        assert gap <= 0.2 + 1e-6, (
            f"Gap between segment {i} (end={end_i}) and segment {i+1} (start={start_next}) "
            f"exceeds 0.2s: gap={gap}."
        )
        overlap = end_i - start_next
        assert overlap <= 0.05 + 1e-6, (
            f"Overlap between segment {i} (end={end_i}) and segment {i+1} (start={start_next}) "
            f"exceeds 0.05s: overlap={overlap}."
        )


def test_each_segment_duration_above_threshold(segments):
    for idx, seg in enumerate(segments):
        duration = float(seg["end"]) - float(seg["start"])
        assert duration > 0.1, (
            f"Segment {idx} duration must be > 0.1s, got {duration} "
            f"(start={seg['start']}, end={seg['end']})."
        )


def test_at_least_two_distinct_chord_labels(segments):
    unique_chords = {seg["chord"] for seg in segments}
    assert len(unique_chords) >= 2, (
        f"Expected at least 2 distinct chord labels in the decoded output, "
        f"got {len(unique_chords)}: {sorted(unique_chords)}."
    )
