import json
import os

import pytest


HITS_JSON = "/workspace/hits.json"
INPUT_WAV = "/workspace/input.wav"

REQUIRED_KEYS = {"time_seconds", "grid_index", "velocity", "raw_time_seconds"}


def _is_real_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


@pytest.fixture(scope="module")
def raw_payload():
    assert os.path.isfile(HITS_JSON), (
        f"Expected output file {HITS_JSON} to exist after the task completes."
    )
    with open(HITS_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{HITS_JSON} is not valid JSON: {exc}")
    return data


@pytest.fixture(scope="module")
def hits(raw_payload):
    if isinstance(raw_payload, list):
        return raw_payload
    if isinstance(raw_payload, dict):
        assert "hits" in raw_payload and isinstance(raw_payload["hits"], list), (
            f"{HITS_JSON} is a JSON object but does not contain a 'hits' array; "
            f"top-level keys: {sorted(raw_payload.keys())}."
        )
        return raw_payload["hits"]
    raise AssertionError(
        f"{HITS_JSON} must be a JSON array of hits or a JSON object with a 'hits' "
        f"array, got: {type(raw_payload).__name__}."
    )


@pytest.fixture(scope="module")
def audio_duration():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    duration = float(len(y)) / float(sr)
    assert duration > 0, f"Reference audio duration must be positive, got {duration}."
    return duration


@pytest.fixture(scope="module")
def estimated_tempo(raw_payload):
    if isinstance(raw_payload, dict):
        metadata = raw_payload.get("_metadata")
        if isinstance(metadata, dict) and "estimated_tempo" in metadata:
            candidate = metadata["estimated_tempo"]
            assert _is_real_number(candidate), (
                f"_metadata.estimated_tempo must be numeric, got: {candidate!r} "
                f"(type {type(candidate).__name__})."
            )
            return float(candidate)

    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    _, y_percussive = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(
        y=y_percussive, sr=sr, hop_length=512
    )
    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=512
    )
    try:
        tempo_scalar = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"Failed to coerce librosa.beat.beat_track tempo {tempo!r} to a scalar: {exc}"
        )
    return tempo_scalar


def test_hits_list_minimum_length(hits):
    assert len(hits) >= 5, (
        f"Expected at least 5 hits in /workspace/hits.json, got {len(hits)}."
    )


def test_each_hit_has_required_schema(hits):
    for idx, hit in enumerate(hits):
        assert isinstance(hit, dict), (
            f"Hit {idx} is not a JSON object, got: {type(hit).__name__}."
        )
        assert set(hit.keys()) == REQUIRED_KEYS, (
            f"Hit {idx} has keys {sorted(hit.keys())}; expected exactly "
            f"{sorted(REQUIRED_KEYS)}."
        )
        assert _is_real_number(hit["time_seconds"]), (
            f"Hit {idx} 'time_seconds' must be numeric, got: {hit['time_seconds']!r}."
        )
        assert _is_real_number(hit["raw_time_seconds"]), (
            f"Hit {idx} 'raw_time_seconds' must be numeric, got: "
            f"{hit['raw_time_seconds']!r}."
        )
        assert _is_integer(hit["grid_index"]), (
            f"Hit {idx} 'grid_index' must be an integer, got: {hit['grid_index']!r} "
            f"(type {type(hit['grid_index']).__name__})."
        )
        assert hit["grid_index"] >= 0, (
            f"Hit {idx} 'grid_index' must be non-negative, got {hit['grid_index']}."
        )
        assert _is_real_number(hit["velocity"]), (
            f"Hit {idx} 'velocity' must be numeric, got: {hit['velocity']!r}."
        )
        velocity = float(hit["velocity"])
        assert 0.0 < velocity <= 1.0, (
            f"Hit {idx} 'velocity' must be in (0.0, 1.0], got {velocity}."
        )


def test_time_fields_within_audio_bounds(hits, audio_duration):
    upper = audio_duration + 1e-3
    for idx, hit in enumerate(hits):
        t = float(hit["time_seconds"])
        raw = float(hit["raw_time_seconds"])
        assert 0.0 <= t <= upper, (
            f"Hit {idx} 'time_seconds'={t} is outside [0, {upper}] "
            f"(audio_duration={audio_duration})."
        )
        assert 0.0 <= raw <= upper, (
            f"Hit {idx} 'raw_time_seconds'={raw} is outside [0, {upper}] "
            f"(audio_duration={audio_duration})."
        )


def test_time_seconds_non_decreasing(hits):
    times = [float(h["time_seconds"]) for h in hits]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1] - 1e-9, (
            f"'time_seconds' must be non-decreasing; hit {i-1}={times[i-1]} "
            f"> hit {i}={times[i]}."
        )


def test_grid_index_non_decreasing(hits):
    indices = [int(h["grid_index"]) for h in hits]
    for i in range(1, len(indices)):
        assert indices[i] >= indices[i - 1], (
            f"'grid_index' must be non-decreasing; hit {i-1}={indices[i-1]} "
            f"> hit {i}={indices[i]}."
        )


def test_estimated_tempo_in_expected_range(estimated_tempo):
    assert 40.0 <= estimated_tempo <= 240.0, (
        f"Estimated tempo {estimated_tempo} BPM is outside [40, 240]."
    )


def test_snapped_time_within_grid_tolerance(hits, estimated_tempo):
    step_seconds = 60.0 / float(estimated_tempo) / 4.0
    tolerance = step_seconds / 2.0 + 1e-3
    for idx, hit in enumerate(hits):
        t = float(hit["time_seconds"])
        raw = float(hit["raw_time_seconds"])
        delta = abs(t - raw)
        assert delta <= tolerance, (
            f"Hit {idx} snap distance |time_seconds - raw_time_seconds|={delta} "
            f"exceeds tolerance {tolerance} (step={step_seconds}, "
            f"tempo={estimated_tempo})."
        )
