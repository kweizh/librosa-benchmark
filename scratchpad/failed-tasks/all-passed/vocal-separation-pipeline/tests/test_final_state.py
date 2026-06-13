import os

import numpy as np
import pytest
import soundfile as sf


WORKSPACE = "/workspace"
INPUT_PATH = os.path.join(WORKSPACE, "input.wav")
FOREGROUND_PATH = os.path.join(WORKSPACE, "foreground.wav")
BACKGROUND_PATH = os.path.join(WORKSPACE, "background.wav")

LENGTH_TOLERANCE_SAMPLES = 1024
RECON_COS_SIM_MIN = 0.85
RECON_NMAE_MAX = 0.05
FG_BG_COS_SIM_MAX = 0.95
MIN_RMS = 1e-4


def _read_wav_channels_first(path):
    """Read a WAV file and return (data, sr) with data in shape (channels, n_samples)."""
    data, sr = sf.read(path, always_2d=True)
    # soundfile returns shape (n_samples, channels); convert to (channels, n_samples)
    data = np.asarray(data, dtype=np.float64).T
    return data, sr


def _to_mono(data):
    """Average across channels to obtain a 1-D mono waveform."""
    if data.ndim == 1:
        return data
    return data.mean(axis=0)


def _crop_or_pad(arr, target_len):
    """Crop or zero-pad along the last axis so the last dimension == target_len."""
    cur = arr.shape[-1]
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[..., :target_len]
    pad_shape = list(arr.shape)
    pad_shape[-1] = target_len - cur
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.fixture(scope="module")
def input_audio():
    assert os.path.isfile(INPUT_PATH), f"Input file {INPUT_PATH} is missing."
    data, sr = _read_wav_channels_first(INPUT_PATH)
    return data, sr


@pytest.fixture(scope="module")
def foreground_audio():
    assert os.path.isfile(FOREGROUND_PATH), (
        f"Foreground output {FOREGROUND_PATH} was not created."
    )
    data, sr = _read_wav_channels_first(FOREGROUND_PATH)
    return data, sr


@pytest.fixture(scope="module")
def background_audio():
    assert os.path.isfile(BACKGROUND_PATH), (
        f"Background output {BACKGROUND_PATH} was not created."
    )
    data, sr = _read_wav_channels_first(BACKGROUND_PATH)
    return data, sr


def test_foreground_file_exists_and_readable():
    assert os.path.isfile(FOREGROUND_PATH), (
        f"Expected foreground output file at {FOREGROUND_PATH}."
    )
    try:
        sf.read(FOREGROUND_PATH)
    except Exception as exc:
        pytest.fail(f"Foreground WAV {FOREGROUND_PATH} is not readable: {exc}")


def test_background_file_exists_and_readable():
    assert os.path.isfile(BACKGROUND_PATH), (
        f"Expected background output file at {BACKGROUND_PATH}."
    )
    try:
        sf.read(BACKGROUND_PATH)
    except Exception as exc:
        pytest.fail(f"Background WAV {BACKGROUND_PATH} is not readable: {exc}")


def test_sample_rate_matches_input(input_audio, foreground_audio, background_audio):
    _, sr_in = input_audio
    _, sr_fg = foreground_audio
    _, sr_bg = background_audio
    assert sr_fg == sr_in, (
        f"Foreground sample rate {sr_fg} does not match input {sr_in}."
    )
    assert sr_bg == sr_in, (
        f"Background sample rate {sr_bg} does not match input {sr_in}."
    )


def test_channel_count_matches_input(input_audio, foreground_audio, background_audio):
    in_data, _ = input_audio
    fg_data, _ = foreground_audio
    bg_data, _ = background_audio
    in_ch = in_data.shape[0]
    fg_ch = fg_data.shape[0]
    bg_ch = bg_data.shape[0]
    assert fg_ch == in_ch, (
        f"Foreground channel count {fg_ch} does not match input {in_ch}."
    )
    assert bg_ch == in_ch, (
        f"Background channel count {bg_ch} does not match input {in_ch}."
    )


def test_length_within_tolerance(input_audio, foreground_audio, background_audio):
    in_data, _ = input_audio
    fg_data, _ = foreground_audio
    bg_data, _ = background_audio
    n_in = in_data.shape[-1]
    n_fg = fg_data.shape[-1]
    n_bg = bg_data.shape[-1]
    assert abs(n_fg - n_in) <= LENGTH_TOLERANCE_SAMPLES, (
        f"Foreground length {n_fg} differs from input length {n_in} "
        f"by more than {LENGTH_TOLERANCE_SAMPLES} samples."
    )
    assert abs(n_bg - n_in) <= LENGTH_TOLERANCE_SAMPLES, (
        f"Background length {n_bg} differs from input length {n_in} "
        f"by more than {LENGTH_TOLERANCE_SAMPLES} samples."
    )


def test_additive_reconstruction(input_audio, foreground_audio, background_audio):
    in_data, _ = input_audio
    fg_data, _ = foreground_audio
    bg_data, _ = background_audio
    n_in = in_data.shape[-1]
    fg_aligned = _crop_or_pad(fg_data, n_in)
    bg_aligned = _crop_or_pad(bg_data, n_in)

    in_mono = _to_mono(in_data)
    fg_mono = _to_mono(fg_aligned)
    bg_mono = _to_mono(bg_aligned)
    recon = fg_mono + bg_mono

    cos_sim = _cosine_similarity(recon, in_mono)
    assert cos_sim >= RECON_COS_SIM_MIN, (
        f"Cosine similarity between (foreground + background) and the input "
        f"is {cos_sim:.4f}, expected >= {RECON_COS_SIM_MIN}."
    )

    denom = float(np.max(np.abs(in_mono))) + 1e-8
    nmae = float(np.mean(np.abs(recon - in_mono)) / denom)
    assert nmae < RECON_NMAE_MAX, (
        f"Normalized mean absolute reconstruction error is {nmae:.4f}, "
        f"expected < {RECON_NMAE_MAX}."
    )


def test_foreground_and_background_are_distinct(foreground_audio, background_audio):
    fg_data, _ = foreground_audio
    bg_data, _ = background_audio
    n = min(fg_data.shape[-1], bg_data.shape[-1])
    fg_mono = _to_mono(fg_data[..., :n])
    bg_mono = _to_mono(bg_data[..., :n])
    cos_sim = _cosine_similarity(fg_mono, bg_mono)
    assert cos_sim < FG_BG_COS_SIM_MAX, (
        f"Foreground and background are too similar "
        f"(cosine similarity = {cos_sim:.4f}, expected < {FG_BG_COS_SIM_MAX}). "
        f"They likely were not actually separated."
    )


def test_foreground_not_silent(foreground_audio):
    fg_data, _ = foreground_audio
    rms = float(np.sqrt(np.mean(np.square(fg_data))))
    assert rms > MIN_RMS, (
        f"Foreground RMS energy {rms:.6f} is below {MIN_RMS}; output appears silent."
    )


def test_background_not_silent(background_audio):
    bg_data, _ = background_audio
    rms = float(np.sqrt(np.mean(np.square(bg_data))))
    assert rms > MIN_RMS, (
        f"Background RMS energy {rms:.6f} is below {MIN_RMS}; output appears silent."
    )
