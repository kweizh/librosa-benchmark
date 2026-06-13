import os

import pytest


WORKSPACE = "/workspace"
INPUT_PATH = os.path.join(WORKSPACE, "input.wav")


def test_librosa_importable():
    try:
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import librosa: {exc}")


def test_soundfile_importable():
    try:
        import soundfile  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import soundfile: {exc}")


def test_numpy_importable():
    try:
        import numpy  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import numpy: {exc}")


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), (
        f"Expected workspace directory {WORKSPACE} to exist before the task starts."
    )


def test_input_wav_exists():
    assert os.path.isfile(INPUT_PATH), (
        f"Expected input audio file {INPUT_PATH} to exist before the task starts."
    )


def test_input_wav_is_readable():
    import soundfile as sf

    try:
        with sf.SoundFile(INPUT_PATH) as f:
            assert f.samplerate > 0, (
                f"Input file {INPUT_PATH} has a non-positive sample rate."
            )
            assert f.channels >= 1, (
                f"Input file {INPUT_PATH} has zero channels."
            )
            assert len(f) > 0, (
                f"Input file {INPUT_PATH} contains no audio samples."
            )
    except Exception as exc:
        pytest.fail(f"Failed to read input WAV {INPUT_PATH}: {exc}")
