import json
import math
import os

import pytest


MELODY_JSON = "/workspace/melody.json"
INPUT_WAV = "/workspace/input.wav"

MIDI_MIN = 21
MIDI_MAX = 108


@pytest.fixture(scope="module")
def melody():
    assert os.path.isfile(MELODY_JSON), (
        f"Expected output file {MELODY_JSON} to exist after the task completes."
    )
    with open(MELODY_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{MELODY_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, list), (
        f"{MELODY_JSON} must be a JSON array, got: {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def audio_duration():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    duration = float(len(y)) / float(sr)
    assert duration > 0, f"Reference audio duration must be positive, got {duration}."
    return duration


def test_melody_has_minimum_length(melody):
    assert len(melody) >= 5, (
        f"Expected at least 5 notes in melody.json, got {len(melody)}."
    )


def test_each_note_has_required_schema(melody):
    for idx, note in enumerate(melody):
        assert isinstance(note, dict), (
            f"Note {idx} is not a JSON object, got: {type(note).__name__}."
        )
        for key in ("start", "end", "midi_note", "mean_f0_hz"):
            assert key in note, (
                f"Note {idx} is missing required key '{key}': {note!r}."
            )
        assert isinstance(note["start"], (int, float)) and not isinstance(note["start"], bool), (
            f"Note {idx} 'start' must be numeric, got: {note['start']!r}."
        )
        assert isinstance(note["end"], (int, float)) and not isinstance(note["end"], bool), (
            f"Note {idx} 'end' must be numeric, got: {note['end']!r}."
        )
        assert isinstance(note["midi_note"], int) and not isinstance(note["midi_note"], bool), (
            f"Note {idx} 'midi_note' must be an int, got: {note['midi_note']!r} "
            f"(type {type(note['midi_note']).__name__})."
        )
        assert isinstance(note["mean_f0_hz"], (int, float)) and not isinstance(note["mean_f0_hz"], bool), (
            f"Note {idx} 'mean_f0_hz' must be numeric, got: {note['mean_f0_hz']!r}."
        )
        assert float(note["mean_f0_hz"]) > 0, (
            f"Note {idx} 'mean_f0_hz' must be positive, got: {note['mean_f0_hz']}."
        )


def test_midi_note_in_piano_range(melody):
    for idx, note in enumerate(melody):
        midi_note = note["midi_note"]
        assert MIDI_MIN <= midi_note <= MIDI_MAX, (
            f"Note {idx} midi_note={midi_note} is outside the allowed range "
            f"[{MIDI_MIN}, {MIDI_MAX}]."
        )


def test_notes_sorted_and_non_overlapping(melody):
    sorted_notes = sorted(melody, key=lambda n: float(n["start"]))
    for idx, (a, b) in enumerate(zip(melody, sorted_notes)):
        assert float(a["start"]) == float(b["start"]), (
            f"Notes are not sorted by start; element {idx} has start={a['start']} "
            f"but sorted position has start={b['start']}."
        )
    for i in range(len(melody) - 1):
        end_i = float(melody[i]["end"])
        start_next = float(melody[i + 1]["start"])
        assert start_next >= end_i - 1e-6, (
            f"Notes {i} and {i+1} overlap: end={end_i}, next start={start_next}."
        )


def test_each_note_duration_above_threshold(melody):
    for idx, note in enumerate(melody):
        duration = float(note["end"]) - float(note["start"])
        assert duration > 0.05, (
            f"Note {idx} duration must be > 0.05s, got {duration} "
            f"(start={note['start']}, end={note['end']})."
        )


def test_note_timestamps_within_audio_bounds(melody, audio_duration):
    upper = audio_duration + 0.05
    for idx, note in enumerate(melody):
        start = float(note["start"])
        end = float(note["end"])
        assert 0 <= start <= upper, (
            f"Note {idx} start={start} outside [0, {upper}] (audio_duration={audio_duration})."
        )
        assert 0 <= end <= upper, (
            f"Note {idx} end={end} outside [0, {upper}] (audio_duration={audio_duration})."
        )


def test_mean_f0_within_50_cents_of_midi(melody):
    import librosa

    for idx, note in enumerate(melody):
        midi_note = note["midi_note"]
        mean_f0 = float(note["mean_f0_hz"])
        ref_hz = float(librosa.midi_to_hz(midi_note))
        assert ref_hz > 0, (
            f"Reference frequency for midi_note={midi_note} is non-positive: {ref_hz}."
        )
        cents = abs(1200.0 * math.log2(mean_f0 / ref_hz))
        assert cents <= 50.0 + 1e-6, (
            f"Note {idx} mean_f0_hz={mean_f0} deviates {cents:.2f} cents from "
            f"midi_to_hz({midi_note})={ref_hz:.3f}; allowed <= 50 cents."
        )


def test_distinct_pitch_count(melody):
    distinct = {note["midi_note"] for note in melody}
    assert len(distinct) >= 4, (
        f"Expected at least 4 distinct midi_note values, got {len(distinct)}: "
        f"{sorted(distinct)}."
    )
