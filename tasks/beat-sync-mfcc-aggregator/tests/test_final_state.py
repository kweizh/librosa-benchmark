import json
import os

import numpy as np
import pytest


WORKSPACE = "/workspace"
INPUT_WAV = "/workspace/input.wav"
BEAT_MFCC_NPZ = "/workspace/beat_mfcc.npz"
BEATS_JSON = "/workspace/beats.json"


@pytest.fixture(scope="module")
def meta():
    assert os.path.isfile(BEATS_JSON), (
        f"Expected {BEATS_JSON} to exist after the task completes."
    )
    with open(BEATS_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{BEATS_JSON} is not valid JSON: {exc}")
    assert isinstance(data, dict), (
        f"{BEATS_JSON} must be a JSON object, got: {type(data).__name__}."
    )
    for key in ("tempo_bpm", "beat_times_seconds", "hop_length", "sample_rate", "n_mfcc"):
        assert key in data, f"{BEATS_JSON} is missing required key '{key}'."
    return data


@pytest.fixture(scope="module")
def mfcc_sync():
    assert os.path.isfile(BEAT_MFCC_NPZ), (
        f"Expected {BEAT_MFCC_NPZ} to exist after the task completes."
    )
    with np.load(BEAT_MFCC_NPZ) as npz:
        assert "mfcc_sync" in npz.files, (
            f"{BEAT_MFCC_NPZ} must contain a 'mfcc_sync' array, got keys: {npz.files}."
        )
        arr = np.array(npz["mfcc_sync"])
    return arr


@pytest.fixture(scope="module")
def audio():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    duration = float(len(y)) / float(sr)
    assert duration > 0, f"Reference audio duration must be positive, got {duration}."
    return y, int(sr), duration


def test_meta_schema_and_types(meta):
    tempo = meta["tempo_bpm"]
    assert isinstance(tempo, (int, float)) and not isinstance(tempo, bool), (
        f"tempo_bpm must be numeric, got: {tempo!r}."
    )
    assert 40.0 <= float(tempo) <= 240.0, (
        f"tempo_bpm must be in [40.0, 240.0], got {tempo}."
    )

    beats = meta["beat_times_seconds"]
    assert isinstance(beats, list) and len(beats) > 0, (
        f"beat_times_seconds must be a non-empty list, got: {beats!r}."
    )
    for i, t in enumerate(beats):
        assert isinstance(t, (int, float)) and not isinstance(t, bool), (
            f"beat_times_seconds[{i}] must be numeric, got: {t!r}."
        )

    assert isinstance(meta["hop_length"], int) and meta["hop_length"] > 0, (
        f"hop_length must be a positive integer, got: {meta['hop_length']!r}."
    )
    assert isinstance(meta["sample_rate"], int) and meta["sample_rate"] > 0, (
        f"sample_rate must be a positive integer, got: {meta['sample_rate']!r}."
    )
    assert meta["n_mfcc"] == 20, (
        f"n_mfcc must equal 20, got: {meta['n_mfcc']!r}."
    )


def test_beat_times_strictly_increasing(meta):
    beats = [float(t) for t in meta["beat_times_seconds"]]
    for i in range(len(beats) - 1):
        assert beats[i] < beats[i + 1], (
            f"beat_times_seconds must be strictly increasing; "
            f"index {i} ({beats[i]}) is not < index {i+1} ({beats[i+1]})."
        )


def test_beat_times_within_audio_bounds(meta, audio):
    _, _, duration = audio
    beats = [float(t) for t in meta["beat_times_seconds"]]
    upper = duration + 1e-3
    for i, t in enumerate(beats):
        assert 0.0 <= t <= upper, (
            f"beat_times_seconds[{i}]={t} is outside [0, audio_duration+1e-3]={upper}."
        )


def test_mfcc_sync_shape(mfcc_sync):
    assert mfcc_sync.ndim == 2, (
        f"mfcc_sync must be 2-D, got shape {mfcc_sync.shape}."
    )
    assert mfcc_sync.shape[0] == 20, (
        f"mfcc_sync.shape[0] must equal 20, got {mfcc_sync.shape[0]}."
    )
    assert mfcc_sync.shape[1] >= 1, (
        f"mfcc_sync.shape[1] must be >= 1, got {mfcc_sync.shape[1]}."
    )


def test_mfcc_sync_column_count_matches_boundary_convention(meta, mfcc_sync):
    n_cols = int(mfcc_sync.shape[1])
    n_beats = len(meta["beat_times_seconds"])
    accepted = {n_beats - 1, n_beats}
    assert n_cols in accepted, (
        f"mfcc_sync.shape[1]={n_cols} must equal len(beat_times_seconds) - 1 "
        f"({n_beats - 1}) OR len(beat_times_seconds) ({n_beats})."
    )


def test_mfcc_sync_matches_recomputed_median(meta, mfcc_sync, audio):
    import librosa

    y, sr_actual, _ = audio
    hop_length = int(meta["hop_length"])
    sample_rate = int(meta["sample_rate"])
    assert sample_rate == int(sr_actual), (
        f"Recorded sample_rate ({sample_rate}) must equal the audio's native "
        f"sample rate ({sr_actual})."
    )

    beat_times = np.array([float(t) for t in meta["beat_times_seconds"]], dtype=np.float64)
    beat_frames = librosa.time_to_frames(
        beat_times, sr=sample_rate, hop_length=hop_length
    )
    beat_frames = np.asarray(beat_frames, dtype=int)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sample_rate, n_mfcc=20, hop_length=hop_length
    )

    expected = None
    for pad in (True, False):
        try:
            candidate = librosa.util.sync(
                mfcc, beat_frames, aggregate=np.median, pad=pad
            )
        except Exception:
            continue
        if candidate.shape == mfcc_sync.shape:
            expected = np.asarray(candidate)
            break

    assert expected is not None, (
        f"Could not reproduce mfcc_sync shape {mfcc_sync.shape} via "
        f"librosa.util.sync with pad in {{True, False}}; check the beat boundary "
        f"convention recorded in beats.json."
    )

    n_cols = mfcc_sync.shape[1]
    diffs = np.abs(np.asarray(mfcc_sync, dtype=np.float64) - expected.astype(np.float64))
    col_max_diff = diffs.max(axis=0)
    matching = int(np.sum(col_max_diff <= 1e-3))
    ratio = matching / float(n_cols)
    assert ratio >= 0.8, (
        f"Only {matching}/{n_cols} ({ratio:.2%}) columns of mfcc_sync match the "
        f"recomputed per-interval median within 1e-3; require >= 80%."
    )
