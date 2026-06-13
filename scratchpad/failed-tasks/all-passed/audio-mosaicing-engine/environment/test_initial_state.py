import os

import pytest


WORKSPACE = "/workspace"
TARGET_PATH = os.path.join(WORKSPACE, "target.wav")
SOURCE_PATH = os.path.join(WORKSPACE, "source.wav")


def test_librosa_importable():
    try:
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover - failure is the assertion
        pytest.fail(f"librosa must be importable in the task environment: {exc}")


def test_soundfile_importable():
    try:
        import soundfile  # noqa: F401
    except Exception as exc:  # pragma: no cover - failure is the assertion
        pytest.fail(f"soundfile must be importable in the task environment: {exc}")


def test_workspace_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} must exist."


def test_target_wav_exists():
    assert os.path.isfile(TARGET_PATH), f"Target audio file {TARGET_PATH} must exist."


def test_source_wav_exists():
    assert os.path.isfile(SOURCE_PATH), f"Source audio file {SOURCE_PATH} must exist."


def test_target_wav_is_22050_mono():
    import soundfile as sf

    data, sr = sf.read(TARGET_PATH, always_2d=False)
    assert sr == 22050, f"Target sample rate must be 22050 Hz, got {sr}."
    assert data.ndim == 1, f"Target must be mono, got shape {data.shape}."


def test_source_wav_is_22050_mono():
    import soundfile as sf

    data, sr = sf.read(SOURCE_PATH, always_2d=False)
    assert sr == 22050, f"Source sample rate must be 22050 Hz, got {sr}."
    assert data.ndim == 1, f"Source must be mono, got shape {data.shape}."


def test_mosaic_not_yet_created():
    mosaic_path = os.path.join(WORKSPACE, "mosaic.wav")
    assert not os.path.exists(mosaic_path), (
        f"Mosaic output {mosaic_path} must not exist before the agent runs."
    )
