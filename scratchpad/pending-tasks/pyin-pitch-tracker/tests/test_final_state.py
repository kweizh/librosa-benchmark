import json
import math
import os

import pytest


PITCH_TRACK_JSON = "/workspace/pitch_track.json"
INPUT_WAV = "/workspace/input.wav"

REQUIRED_KEYS = {"time", "f0_hz", "voiced", "voiced_prob"}


def _is_real_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@pytest.fixture(scope="module")
def records():
    assert os.path.isfile(PITCH_TRACK_JSON), (
        f"Expected output file {PITCH_TRACK_JSON} to exist after the task completes."
    )
    with open(PITCH_TRACK_JSON, "r") as fh:
        raw = fh.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{PITCH_TRACK_JSON} is not valid JSON: {exc}"
        )
    assert isinstance(data, list) and len(data) > 0, (
        f"{PITCH_TRACK_JSON} must be a non-empty JSON array, got: {type(data).__name__} "
        f"with length {len(data) if hasattr(data, '__len__') else 'N/A'}."
    )
    return data


@pytest.fixture(scope="module")
def reference_pyin():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    fmin = float(librosa.note_to_hz("C2"))
    fmax = float(librosa.note_to_hz("C7"))
    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    duration = float(len(y)) / float(sr)
    return {
        "n_frames": int(f0.shape[-1]),
        "duration": duration,
        "fmin": fmin,
        "fmax": fmax,
        "sr": int(sr),
    }


def test_each_record_has_required_schema(records):
    for idx, rec in enumerate(records):
        assert isinstance(rec, dict), (
            f"Record {idx} is not a JSON object, got: {type(rec).__name__}."
        )
        missing = REQUIRED_KEYS - set(rec.keys())
        assert not missing, (
            f"Record {idx} is missing required keys {sorted(missing)}: {rec!r}."
        )
        assert _is_real_number(rec["time"]), (
            f"Record {idx} 'time' must be numeric, got: {rec['time']!r}."
        )
        assert math.isfinite(float(rec["time"])), (
            f"Record {idx} 'time' must be finite, got: {rec['time']!r}."
        )
        assert isinstance(rec["voiced"], bool), (
            f"Record {idx} 'voiced' must be a JSON boolean, got: {rec['voiced']!r}."
        )
        assert _is_real_number(rec["voiced_prob"]), (
            f"Record {idx} 'voiced_prob' must be numeric, got: {rec['voiced_prob']!r}."
        )
        prob = float(rec["voiced_prob"])
        assert math.isfinite(prob), (
            f"Record {idx} 'voiced_prob' must be finite, got: {rec['voiced_prob']!r}."
        )
        assert 0.0 <= prob <= 1.0, (
            f"Record {idx} 'voiced_prob' must be in [0.0, 1.0], got: {prob}."
        )
        f0 = rec["f0_hz"]
        if f0 is not None:
            assert _is_real_number(f0), (
                f"Record {idx} 'f0_hz' must be null or numeric, got: {f0!r}."
            )


def test_frame_count_matches_pyin_output(records, reference_pyin):
    expected = reference_pyin["n_frames"]
    assert len(records) == expected, (
        f"Expected {expected} frames from librosa.pyin (C2..C7, default kwargs), "
        f"but {PITCH_TRACK_JSON} has {len(records)} records."
    )


def test_time_values_are_strictly_monotonic(records):
    for i in range(len(records) - 1):
        t_now = float(records[i]["time"])
        t_next = float(records[i + 1]["time"])
        assert t_next > t_now, (
            f"Time values must be strictly increasing: records[{i}].time={t_now} "
            f"is not less than records[{i + 1}].time={t_next}."
        )


def test_last_time_close_to_audio_duration(records, reference_pyin):
    last_time = float(records[-1]["time"])
    duration = reference_pyin["duration"]
    delta = abs(last_time - duration)
    assert delta <= 0.1 + 1e-6, (
        f"Last frame time {last_time}s must be within 0.1s of audio duration {duration}s, "
        f"delta={delta}s."
    )


def test_unvoiced_frames_have_null_f0(records):
    for idx, rec in enumerate(records):
        if rec["voiced"] is False:
            assert rec["f0_hz"] is None, (
                f"Record {idx} has voiced=false but f0_hz is not null: {rec['f0_hz']!r}."
            )


def test_voiced_frames_have_in_range_finite_f0(records, reference_pyin):
    fmin = reference_pyin["fmin"]
    fmax = reference_pyin["fmax"]
    for idx, rec in enumerate(records):
        if rec["voiced"] is True:
            f0 = rec["f0_hz"]
            assert f0 is not None, (
                f"Record {idx} has voiced=true but f0_hz is null."
            )
            assert _is_real_number(f0), (
                f"Record {idx} has voiced=true but f0_hz is not numeric: {f0!r}."
            )
            f0_val = float(f0)
            assert math.isfinite(f0_val), (
                f"Record {idx} has voiced=true but f0_hz is not finite: {f0_val}."
            )
            assert fmin - 1e-6 <= f0_val <= fmax + 1e-6, (
                f"Record {idx} f0_hz={f0_val} is outside [C2={fmin}, C7={fmax}]."
            )


def test_voiced_fraction_above_threshold(records):
    total = len(records)
    voiced_count = sum(1 for rec in records if rec["voiced"] is True)
    fraction = voiced_count / total
    assert fraction >= 0.30, (
        f"At least 30% of frames must be voiced, got {fraction:.3f} "
        f"({voiced_count}/{total})."
    )


def test_unvoiced_fraction_above_threshold(records):
    total = len(records)
    unvoiced_count = sum(1 for rec in records if rec["voiced"] is False)
    fraction = unvoiced_count / total
    assert fraction >= 0.05, (
        f"At least 5% of frames must be unvoiced, got {fraction:.3f} "
        f"({unvoiced_count}/{total})."
    )
