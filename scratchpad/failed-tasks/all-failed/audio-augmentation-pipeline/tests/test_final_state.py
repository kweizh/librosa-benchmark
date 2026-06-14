import json
import math
import os

import numpy as np
import pytest


WORKSPACE = "/workspace"
INPUT_WAV = os.path.join(WORKSPACE, "input.wav")
AUGMENTED_WAV = os.path.join(WORKSPACE, "augmented.wav")
AUG_META_JSON = os.path.join(WORKSPACE, "aug_meta.json")

REQUIRED_KEYS = (
    "sample_rate",
    "input_duration_seconds",
    "after_pitch_shift_seconds",
    "after_time_stretch_seconds",
    "after_trim_seconds",
    "trim_indices",
    "n_steps",
    "rate",
    "top_db",
)


@pytest.fixture(scope="module")
def meta():
    assert os.path.isfile(AUG_META_JSON), (
        f"Expected metadata file {AUG_META_JSON} to exist after the task completes."
    )
    with open(AUG_META_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{AUG_META_JSON} is not valid JSON: {exc}")
    assert isinstance(data, dict), (
        f"{AUG_META_JSON} must contain a JSON object, got: {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def input_audio():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    assert y.ndim == 1 and y.size > 0, (
        f"Expected non-empty mono input from {INPUT_WAV}, got ndim={y.ndim}, size={y.size}."
    )
    return y, int(sr)


@pytest.fixture(scope="module")
def augmented_audio():
    import librosa

    assert os.path.isfile(AUGMENTED_WAV), (
        f"Expected augmented output file {AUGMENTED_WAV} to exist after the task completes."
    )
    y, sr = librosa.load(AUGMENTED_WAV, sr=None, mono=True)
    assert y.ndim == 1 and y.size > 0, (
        f"Expected non-empty mono augmented audio at {AUGMENTED_WAV}, got ndim={y.ndim}, size={y.size}."
    )
    return y, int(sr)


def test_meta_has_required_keys(meta):
    for key in REQUIRED_KEYS:
        assert key in meta, (
            f"Metadata file {AUG_META_JSON} is missing required key '{key}'."
        )


def test_meta_param_values(meta):
    assert isinstance(meta["sample_rate"], int) and not isinstance(meta["sample_rate"], bool), (
        f"'sample_rate' must be an integer, got: {meta['sample_rate']!r} ({type(meta['sample_rate']).__name__})."
    )
    assert meta["sample_rate"] > 0, (
        f"'sample_rate' must be positive, got: {meta['sample_rate']}."
    )
    assert float(meta["n_steps"]) == pytest.approx(3.0), (
        f"'n_steps' must equal 3.0, got: {meta['n_steps']}."
    )
    assert float(meta["rate"]) == pytest.approx(0.85), (
        f"'rate' must equal 0.85, got: {meta['rate']}."
    )
    assert float(meta["top_db"]) == pytest.approx(40), (
        f"'top_db' must equal 40, got: {meta['top_db']}."
    )


def test_meta_durations_are_positive_floats(meta):
    for key in (
        "input_duration_seconds",
        "after_pitch_shift_seconds",
        "after_time_stretch_seconds",
        "after_trim_seconds",
    ):
        value = meta[key]
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"'{key}' must be numeric, got: {value!r} ({type(value).__name__})."
        )
        assert float(value) > 0, (
            f"'{key}' must be positive, got: {value}."
        )


def test_trim_indices_schema(meta):
    indices = meta["trim_indices"]
    assert isinstance(indices, list) and len(indices) == 2, (
        f"'trim_indices' must be a 2-element list, got: {indices!r}."
    )
    start, end = indices
    assert isinstance(start, int) and not isinstance(start, bool), (
        f"'trim_indices[0]' must be an integer, got: {start!r}."
    )
    assert isinstance(end, int) and not isinstance(end, bool), (
        f"'trim_indices[1]' must be an integer, got: {end!r}."
    )
    assert 0 <= start < end, (
        f"'trim_indices' must satisfy 0 <= start < end, got: start={start}, end={end}."
    )
    upper = round(float(meta["after_time_stretch_seconds"]) * float(meta["sample_rate"]))
    assert end <= upper, (
        f"'trim_indices[1]' ({end}) must be <= round(after_time_stretch_seconds * sample_rate) ({upper})."
    )


def test_sample_rate_matches_input(meta, input_audio):
    _, sr_in = input_audio
    assert meta["sample_rate"] == sr_in, (
        f"Metadata sample_rate ({meta['sample_rate']}) must equal input sample rate ({sr_in})."
    )


def test_input_duration_matches_reference(meta, input_audio):
    y_in, sr_in = input_audio
    duration = float(len(y_in)) / float(sr_in)
    rel_err = abs(float(meta["input_duration_seconds"]) - duration) / duration
    assert rel_err <= 0.01, (
        f"Reported input_duration_seconds ({meta['input_duration_seconds']}) deviates "
        f"from measured input duration ({duration}) by more than 1% (rel_err={rel_err:.4f})."
    )


def test_augmented_wav_is_mono_and_matches_sample_rate(meta, augmented_audio):
    y_out, sr_out = augmented_audio
    assert y_out.ndim == 1, (
        f"Augmented WAV must be mono (1-D), got ndim={y_out.ndim}."
    )
    assert sr_out == meta["sample_rate"], (
        f"Augmented WAV sample rate ({sr_out}) must match metadata sample_rate ({meta['sample_rate']})."
    )


def test_augmented_sample_count_matches_after_trim_seconds(meta, augmented_audio):
    y_out, sr_out = augmented_audio
    expected = round(float(meta["after_trim_seconds"]) * float(sr_out))
    assert abs(len(y_out) - expected) <= 1, (
        f"Augmented WAV sample count ({len(y_out)}) must equal "
        f"round(after_trim_seconds * sample_rate) ({expected}) within 1 sample."
    )


def test_pitch_shift_preserves_length(meta):
    input_dur = float(meta["input_duration_seconds"])
    ps_dur = float(meta["after_pitch_shift_seconds"])
    rel_err = abs(ps_dur - input_dur) / input_dur
    assert rel_err <= 0.02, (
        f"Pitch shift should preserve length within 2%; got input={input_dur}, "
        f"after_pitch_shift={ps_dur}, rel_err={rel_err:.4f}."
    )


def test_time_stretch_ratio(meta):
    ps_dur = float(meta["after_pitch_shift_seconds"])
    ts_dur = float(meta["after_time_stretch_seconds"])
    expected = ps_dur / 0.85
    rel_err = abs(ts_dur - expected) / ts_dur
    assert rel_err <= 0.05, (
        f"Time stretch with rate=0.85 should produce duration ~ after_pitch_shift / 0.85 "
        f"({expected}); got after_time_stretch={ts_dur}, rel_err={rel_err:.4f}."
    )


def test_trim_did_not_lengthen_signal(meta):
    ts_dur = float(meta["after_time_stretch_seconds"])
    trim_dur = float(meta["after_trim_seconds"])
    assert trim_dur <= ts_dur + 1e-6, (
        f"after_trim_seconds ({trim_dur}) must be <= after_time_stretch_seconds ({ts_dur})."
    )


def _median_voiced_f0(y, sr):
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        y,
        sr=sr,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    f0 = np.asarray(f0, dtype=float)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    mask = voiced_flag & np.isfinite(f0)
    voiced_f0 = f0[mask]
    assert voiced_f0.size > 0, (
        "pyin produced no voiced frames; cannot compute median F0 for verification."
    )
    return float(np.median(voiced_f0))


def test_f0_shifted_by_three_semitones(input_audio, augmented_audio):
    y_in, sr_in = input_audio
    y_out, sr_out = augmented_audio

    median_in = _median_voiced_f0(y_in, sr_in)
    median_out = _median_voiced_f0(y_out, sr_out)
    assert median_in > 0, f"Median input F0 must be positive, got {median_in}."

    ratio = median_out / median_in
    lower = 2 ** (2.5 / 12)
    upper = 2 ** (3.5 / 12)
    assert lower <= ratio <= upper, (
        f"Median F0 ratio (augmented/input) must lie in [{lower:.4f}, {upper:.4f}] "
        f"(i.e., +3 semitones \u00b10.5 semitone), got ratio={ratio:.4f} "
        f"(median_in={median_in:.2f} Hz, median_out={median_out:.2f} Hz)."
    )
