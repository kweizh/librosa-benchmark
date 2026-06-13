import json
import os
import wave

import pytest


WORKSPACE = "/workspace"
NOTES_JSON = os.path.join(WORKSPACE, "notes.json")
INPUT_WAV = os.path.join(WORKSPACE, "input.wav")

# Ground truth baked into the Dockerfile-generated input.wav.
GROUND_TRUTH_PITCHES = [60, 64, 67, 72, 67]
GROUND_TRUTH_COUNT = len(GROUND_TRUTH_PITCHES)
NOTE_DURATION_SEC = 1.0
GAP_SEC = 0.2
EXPECTED_TOTAL_DURATION_SEC = (
    GROUND_TRUTH_COUNT * NOTE_DURATION_SEC
    + (GROUND_TRUTH_COUNT - 1) * GAP_SEC
)


def _audio_duration_seconds() -> float:
    with wave.open(INPUT_WAV, "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


@pytest.fixture(scope="module")
def notes():
    assert os.path.isfile(NOTES_JSON), (
        f"Expected output file {NOTES_JSON} to exist after the executor finishes."
    )
    with open(NOTES_JSON, "r") as f:
        data = json.load(f)
    assert isinstance(data, list), (
        f"{NOTES_JSON} must contain a JSON list at the top level; got {type(data).__name__}."
    )
    return data


def test_notes_json_exists_and_is_list(notes):
    assert isinstance(notes, list), (
        f"Top-level JSON value in {NOTES_JSON} must be a list."
    )
    assert len(notes) > 0, f"{NOTES_JSON} must contain at least one detected note."


def test_each_note_has_required_keys_and_types(notes):
    required = {"onset_sec", "offset_sec", "pitch_midi", "velocity"}
    for i, entry in enumerate(notes):
        assert isinstance(entry, dict), (
            f"Note at index {i} must be a JSON object; got {type(entry).__name__}."
        )
        assert set(entry.keys()) >= required, (
            f"Note at index {i} is missing required keys; expected {required}, got {set(entry.keys())}."
        )

        onset = entry["onset_sec"]
        offset = entry["offset_sec"]
        pitch = entry["pitch_midi"]
        velocity = entry["velocity"]

        assert isinstance(onset, float) and not isinstance(onset, bool), (
            f"Note at index {i}: onset_sec must be a float, got {type(onset).__name__}."
        )
        assert isinstance(offset, float) and not isinstance(offset, bool), (
            f"Note at index {i}: offset_sec must be a float, got {type(offset).__name__}."
        )
        assert isinstance(pitch, int) and not isinstance(pitch, bool), (
            f"Note at index {i}: pitch_midi must be an int, got {type(pitch).__name__}."
        )
        assert isinstance(velocity, int) and not isinstance(velocity, bool), (
            f"Note at index {i}: velocity must be an int, got {type(velocity).__name__}."
        )

        assert 0 <= pitch <= 127, (
            f"Note at index {i}: pitch_midi={pitch} is out of MIDI range [0, 127]."
        )
        assert 1 <= velocity <= 127, (
            f"Note at index {i}: velocity={velocity} is out of allowed range [1, 127]."
        )


def test_notes_are_strictly_ordered_by_onset(notes):
    for i in range(1, len(notes)):
        prev = notes[i - 1]["onset_sec"]
        cur = notes[i]["onset_sec"]
        assert cur > prev, (
            f"Notes must be strictly increasing by onset_sec; "
            f"got onset[{i - 1}]={prev} and onset[{i}]={cur}."
        )


def test_each_note_has_positive_duration(notes):
    for i, entry in enumerate(notes):
        assert entry["offset_sec"] > entry["onset_sec"], (
            f"Note at index {i}: offset_sec ({entry['offset_sec']}) must be strictly greater than "
            f"onset_sec ({entry['onset_sec']})."
        )


def test_offsets_within_audio_duration(notes):
    duration = _audio_duration_seconds()
    tolerance = 0.1
    for i, entry in enumerate(notes):
        assert entry["offset_sec"] <= duration + tolerance, (
            f"Note at index {i}: offset_sec ({entry['offset_sec']}) exceeds audio duration "
            f"({duration:.3f}s) by more than the tolerance of {tolerance}s."
        )


def test_detected_note_count_close_to_ground_truth(notes):
    diff = abs(len(notes) - GROUND_TRUTH_COUNT)
    assert diff <= 1, (
        f"Expected the number of detected notes to be within 1 of "
        f"{GROUND_TRUTH_COUNT}, got {len(notes)}."
    )


def test_pitch_accuracy_against_ground_truth(notes):
    # Compare the first min(len(notes), GROUND_TRUTH_COUNT) detected notes
    # against the ground-truth sequence in order.
    pairs = list(zip(notes, GROUND_TRUTH_PITCHES))
    correct = 0
    for entry, gt in pairs:
        if abs(int(entry["pitch_midi"]) - gt) <= 1:
            correct += 1
    assert correct >= 4, (
        f"Expected at least 4 of the detected notes to have pitch_midi within 1 semitone "
        f"of the corresponding ground-truth pitch {GROUND_TRUTH_PITCHES}; got {correct} matches "
        f"from detected pitches {[entry['pitch_midi'] for entry in notes]}."
    )
