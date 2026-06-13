import json
import os

import pytest

WORKSPACE_DIR = "/workspace"
INPUT_WAV = os.path.join(WORKSPACE_DIR, "input.wav")
BEATS_JSON = os.path.join(WORKSPACE_DIR, "beats.json")

GROUND_TRUTH_TEMPO_BPM = 120.0
TEMPO_TOLERANCE = 0.10
IBI_TOLERANCE = 0.15


@pytest.fixture(scope="module")
def beats_data():
    assert os.path.isfile(BEATS_JSON), (
        f"Expected output file {BEATS_JSON} was not produced by the task."
    )
    with open(BEATS_JSON, "r") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{BEATS_JSON} is not valid JSON: {exc}"
        ) from exc
    return data


@pytest.fixture(scope="module")
def audio_duration():
    import librosa

    duration = float(librosa.get_duration(path=INPUT_WAV))
    assert duration > 0.0, (
        f"Could not determine a positive duration for {INPUT_WAV}."
    )
    return duration


def test_output_file_exists():
    assert os.path.isfile(BEATS_JSON), (
        f"Expected output file {BEATS_JSON} does not exist."
    )


def test_output_schema(beats_data):
    assert isinstance(beats_data, dict), (
        f"{BEATS_JSON} must contain a JSON object at the top level."
    )
    assert "global_tempo_bpm" in beats_data, (
        f"{BEATS_JSON} is missing the required key 'global_tempo_bpm'."
    )
    assert "beat_times_sec" in beats_data, (
        f"{BEATS_JSON} is missing the required key 'beat_times_sec'."
    )
    assert isinstance(beats_data["global_tempo_bpm"], (int, float)) and not isinstance(
        beats_data["global_tempo_bpm"], bool
    ), "'global_tempo_bpm' must be a JSON number."
    assert isinstance(beats_data["beat_times_sec"], list), (
        "'beat_times_sec' must be a JSON array."
    )
    for i, t in enumerate(beats_data["beat_times_sec"]):
        assert isinstance(t, (int, float)) and not isinstance(t, bool), (
            f"beat_times_sec[{i}] must be a JSON number, got {type(t).__name__}."
        )


def test_global_tempo_in_valid_range(beats_data):
    import math

    tempo = float(beats_data["global_tempo_bpm"])
    assert math.isfinite(tempo), "global_tempo_bpm must be a finite number."
    assert 30.0 <= tempo <= 300.0, (
        f"global_tempo_bpm {tempo} is not in the valid range [30, 300]."
    )


def test_beat_times_nonempty_and_strictly_increasing(beats_data, audio_duration):
    beats = [float(t) for t in beats_data["beat_times_sec"]]
    assert len(beats) > 0, "beat_times_sec must be non-empty."
    upper = audio_duration + 0.5
    for i, t in enumerate(beats):
        assert 0.0 <= t <= upper, (
            f"beat_times_sec[{i}]={t} is outside [0, duration+0.5]={upper}."
        )
    for i in range(1, len(beats)):
        assert beats[i] > beats[i - 1], (
            "beat_times_sec must be strictly increasing; "
            f"index {i} has {beats[i]} <= previous {beats[i - 1]}."
        )


def test_global_tempo_within_tolerance_of_ground_truth(beats_data):
    tempo = float(beats_data["global_tempo_bpm"])
    rel_err = abs(tempo - GROUND_TRUTH_TEMPO_BPM) / GROUND_TRUTH_TEMPO_BPM
    assert rel_err <= TEMPO_TOLERANCE, (
        f"global_tempo_bpm {tempo:.3f} is more than "
        f"{TEMPO_TOLERANCE * 100:.0f}% away from the ground-truth tempo "
        f"{GROUND_TRUTH_TEMPO_BPM:.3f} BPM (relative error {rel_err:.3f})."
    )


def test_median_ibi_matches_global_tempo(beats_data):
    import statistics

    beats = [float(t) for t in beats_data["beat_times_sec"]]
    assert len(beats) >= 2, (
        "Need at least two beats to compute an inter-beat interval."
    )
    intervals = [beats[i] - beats[i - 1] for i in range(1, len(beats))]
    median_ibi = statistics.median(intervals)
    assert median_ibi > 0.0, (
        f"Median inter-beat interval must be positive, got {median_ibi}."
    )
    ibi_tempo = 60.0 / median_ibi
    tempo = float(beats_data["global_tempo_bpm"])
    rel_err = abs(ibi_tempo - tempo) / tempo
    assert rel_err <= IBI_TOLERANCE, (
        f"Tempo implied by median IBI ({ibi_tempo:.3f} BPM) differs from "
        f"global_tempo_bpm ({tempo:.3f} BPM) by more than "
        f"{IBI_TOLERANCE * 100:.0f}% (relative error {rel_err:.3f})."
    )
