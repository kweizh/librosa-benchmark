import os

import numpy as np
import pytest


WORKSPACE = "/workspace"
TARGET_PATH = os.path.join(WORKSPACE, "target.wav")
SOURCE_PATH = os.path.join(WORKSPACE, "source.wav")
MOSAIC_PATH = os.path.join(WORKSPACE, "mosaic.wav")


def _load_mono(path: str):
    import soundfile as sf

    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), int(sr)


@pytest.fixture(scope="module")
def target_audio():
    assert os.path.isfile(TARGET_PATH), f"Target file {TARGET_PATH} missing."
    return _load_mono(TARGET_PATH)


@pytest.fixture(scope="module")
def source_audio():
    assert os.path.isfile(SOURCE_PATH), f"Source file {SOURCE_PATH} missing."
    return _load_mono(SOURCE_PATH)


@pytest.fixture(scope="module")
def mosaic_audio():
    assert os.path.isfile(MOSAIC_PATH), (
        f"Mosaic file {MOSAIC_PATH} was not produced by the agent."
    )
    return _load_mono(MOSAIC_PATH)


def test_mosaic_file_exists_and_is_readable():
    assert os.path.isfile(MOSAIC_PATH), (
        f"Expected mosaic output at {MOSAIC_PATH}, but the file is missing."
    )
    import soundfile as sf

    info = sf.info(MOSAIC_PATH)
    assert info.frames > 0, (
        f"Mosaic file {MOSAIC_PATH} is empty or unreadable: {info}."
    )


def test_mosaic_sample_rate_matches_target(target_audio, mosaic_audio):
    _, target_sr = target_audio
    _, mosaic_sr = mosaic_audio
    assert mosaic_sr == target_sr, (
        f"Mosaic sample rate {mosaic_sr} does not match target sample rate {target_sr}."
    )


def test_mosaic_length_close_to_target(target_audio, mosaic_audio):
    target, _ = target_audio
    mosaic, _ = mosaic_audio
    diff = abs(len(mosaic) - len(target))
    assert diff <= 4096, (
        f"Mosaic length {len(mosaic)} differs from target length {len(target)} "
        f"by {diff} samples (>4096 allowed)."
    )


def test_mosaic_peak_amplitude_in_range(mosaic_audio):
    mosaic, _ = mosaic_audio
    peak = float(np.max(np.abs(mosaic))) if mosaic.size > 0 else 0.0
    assert 0.05 <= peak <= 1.0, (
        f"Mosaic peak amplitude {peak:.4f} is out of the required [0.05, 1.0] range."
    )


def test_mosaic_is_closer_to_target_than_source(
    target_audio, source_audio, mosaic_audio
):
    import librosa

    target, sr = target_audio
    source, _ = source_audio
    mosaic, _ = mosaic_audio

    mfcc_target = librosa.feature.mfcc(y=target, sr=sr)
    mfcc_source = librosa.feature.mfcc(y=source, sr=sr)
    mfcc_mosaic = librosa.feature.mfcc(y=mosaic, sr=sr)

    n_frames = min(
        mfcc_target.shape[1], mfcc_source.shape[1], mfcc_mosaic.shape[1]
    )
    assert n_frames > 0, "Unable to compute MFCC frames for similarity check."

    t = mfcc_target[:, :n_frames]
    s = mfcc_source[:, :n_frames]
    m = mfcc_mosaic[:, :n_frames]

    dist_source = float(np.mean(np.linalg.norm(s - t, axis=0)))
    dist_mosaic = float(np.mean(np.linalg.norm(m - t, axis=0)))

    assert dist_source > 0.0, (
        f"Baseline source-vs-target MFCC distance is {dist_source}; cannot evaluate "
        "mosaic similarity."
    )
    assert dist_mosaic <= 1.3 * dist_source, (
        f"Mosaic MFCC distance to target ({dist_mosaic:.3f}) must be at most "
        f"1.3x the source-to-target distance ({dist_source:.3f}); the mosaic does not "
        "appear meaningfully closer to the target timbre than the raw source."
    )
