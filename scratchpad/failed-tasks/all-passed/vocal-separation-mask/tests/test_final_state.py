import os

import numpy as np
import pytest
import soundfile as sf


INPUT_WAV = "/workspace/input.wav"
VOCAL_WAV = "/workspace/vocal.wav"
ACCOMP_WAV = "/workspace/accompaniment.wav"


def _to_mono_1d(y):
    """Reduce any 2-D soundfile output (frames, channels) to mono 1-D."""
    arr = np.asarray(y)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # soundfile returns (frames, channels)
        if arr.shape[1] == 1:
            return arr[:, 0]
        # average channels if the output is unexpectedly multi-channel
        return arr.mean(axis=1)
    raise AssertionError(f"Unexpected output array shape: {arr.shape}")


def _align_length(y, n):
    """Pad or trim by at most 1 sample to match the reference length n."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    out = np.zeros(n, dtype=np.float64)
    out[: len(y)] = y
    return out


@pytest.fixture(scope="module")
def reference():
    import librosa

    y_ref, sr_ref = librosa.load(INPUT_WAV, sr=None, mono=True)
    assert y_ref.size > 0, f"Reference audio at {INPUT_WAV} is empty."
    assert sr_ref > 0, f"Reference sample rate at {INPUT_WAV} is invalid: {sr_ref}."
    return {
        "y": np.asarray(y_ref, dtype=np.float64),
        "sr": int(sr_ref),
        "n": int(len(y_ref)),
    }


@pytest.fixture(scope="module")
def vocal_output():
    assert os.path.isfile(VOCAL_WAV), (
        f"Expected vocal output file {VOCAL_WAV} to exist after the task completes."
    )
    try:
        y, sr = sf.read(VOCAL_WAV, always_2d=False)
    except Exception as exc:
        raise AssertionError(f"Failed to read {VOCAL_WAV} with soundfile: {exc}")
    y = _to_mono_1d(y)
    return {"y": np.asarray(y, dtype=np.float64), "sr": int(sr)}


@pytest.fixture(scope="module")
def accomp_output():
    assert os.path.isfile(ACCOMP_WAV), (
        f"Expected accompaniment output file {ACCOMP_WAV} to exist after the task completes."
    )
    try:
        y, sr = sf.read(ACCOMP_WAV, always_2d=False)
    except Exception as exc:
        raise AssertionError(f"Failed to read {ACCOMP_WAV} with soundfile: {exc}")
    y = _to_mono_1d(y)
    return {"y": np.asarray(y, dtype=np.float64), "sr": int(sr)}


def test_vocal_sample_rate_matches_input(reference, vocal_output):
    assert vocal_output["sr"] == reference["sr"], (
        f"Vocal sample rate {vocal_output['sr']} does not match input sample rate "
        f"{reference['sr']}."
    )


def test_accomp_sample_rate_matches_input(reference, accomp_output):
    assert accomp_output["sr"] == reference["sr"], (
        f"Accompaniment sample rate {accomp_output['sr']} does not match input sample rate "
        f"{reference['sr']}."
    )


def test_vocal_length_matches_input(reference, vocal_output):
    n_ref = reference["n"]
    n_out = len(vocal_output["y"])
    assert abs(n_out - n_ref) <= 1, (
        f"Vocal output length {n_out} differs from input length {n_ref} by more than 1 sample."
    )


def test_accomp_length_matches_input(reference, accomp_output):
    n_ref = reference["n"]
    n_out = len(accomp_output["y"])
    assert abs(n_out - n_ref) <= 1, (
        f"Accompaniment output length {n_out} differs from input length {n_ref} by more than 1 sample."
    )


def test_vocal_is_not_silent(vocal_output):
    rms = float(np.sqrt(np.mean(vocal_output["y"] ** 2)))
    assert rms > 1e-4, (
        f"Vocal output RMS energy {rms} is below 1e-4; output appears silent."
    )


def test_accomp_is_not_silent(accomp_output):
    rms = float(np.sqrt(np.mean(accomp_output["y"] ** 2)))
    assert rms > 1e-4, (
        f"Accompaniment output RMS energy {rms} is below 1e-4; output appears silent."
    )


def test_vocal_is_not_bitwise_copy_of_input(reference, vocal_output):
    y_ref = reference["y"]
    y_out = vocal_output["y"]
    if len(y_out) == len(y_ref) and np.array_equal(y_out, y_ref):
        raise AssertionError(
            "Vocal output is a bitwise copy of the input mixture; no separation was performed."
        )


def test_accomp_is_not_bitwise_copy_of_input(reference, accomp_output):
    y_ref = reference["y"]
    y_out = accomp_output["y"]
    if len(y_out) == len(y_ref) and np.array_equal(y_out, y_ref):
        raise AssertionError(
            "Accompaniment output is a bitwise copy of the input mixture; no separation was performed."
        )


def test_complementary_masks_sum_reconstructs_input(reference, vocal_output, accomp_output):
    n_ref = reference["n"]
    y_ref = reference["y"]
    y_v = _align_length(vocal_output["y"], n_ref)
    y_a = _align_length(accomp_output["y"], n_ref)
    y_sum = y_v + y_a
    mae = float(np.mean(np.abs(y_sum - y_ref)))
    assert mae <= 5e-2, (
        f"vocal + accompaniment does not reconstruct the input mixture: "
        f"mean-absolute-error {mae} exceeds 5e-2 (soft masks must be complementary)."
    )


def test_separation_reduces_vocal_accomp_similarity(reference, vocal_output, accomp_output):
    import librosa

    n_ref = reference["n"]
    y_ref = reference["y"].astype(np.float32)
    y_v = _align_length(vocal_output["y"], n_ref).astype(np.float32)
    y_a = _align_length(accomp_output["y"], n_ref).astype(np.float32)

    m_ref = np.abs(librosa.stft(y_ref, n_fft=2048, hop_length=512)).flatten()
    m_v = np.abs(librosa.stft(y_v, n_fft=2048, hop_length=512)).flatten()
    m_a = np.abs(librosa.stft(y_a, n_fft=2048, hop_length=512)).flatten()

    def _cos(u, v):
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        assert nu > 0 and nv > 0, (
            "Cannot compute cosine similarity: a magnitude STFT has zero norm."
        )
        return float(np.dot(u, v) / (nu * nv))

    cos_va = _cos(m_v, m_a)
    cos_ra = _cos(m_ref, m_a)
    assert cos_va < cos_ra, (
        f"Expected the vocal track to be LESS similar to the accompaniment than the "
        f"original mixture is, but got cos(vocal, accomp)={cos_va} >= "
        f"cos(input, accomp)={cos_ra}. This indicates no real separation occurred."
    )
