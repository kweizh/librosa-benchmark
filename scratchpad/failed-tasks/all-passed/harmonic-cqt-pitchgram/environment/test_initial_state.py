import os

import pytest


WORKSPACE = "/workspace"
INPUT_WAV = "/workspace/input.wav"


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_input_wav_exists():
    assert os.path.isfile(INPUT_WAV), (
        f"Input audio file {INPUT_WAV} does not exist; it must be baked before evaluation."
    )


def test_input_wav_is_non_empty():
    assert os.path.getsize(INPUT_WAV) > 0, (
        f"Input audio file {INPUT_WAV} is empty; the bootstrap step must populate it."
    )


def test_librosa_importable():
    try:
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover - import-time failure path
        pytest.fail(f"librosa is not importable in the environment: {exc!r}")


def test_numpy_importable():
    try:
        import numpy  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"numpy is not importable in the environment: {exc!r}")


def test_soundfile_importable():
    try:
        import soundfile  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"soundfile is not importable in the environment: {exc!r}")


def test_input_wav_loadable_as_audio():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    assert y.ndim == 1 and y.size > 0, (
        f"Input WAV at {INPUT_WAV} did not load as a non-empty mono waveform; got shape={y.shape}."
    )
    assert sr > 0, f"Input WAV at {INPUT_WAV} reported a non-positive sample rate: {sr}."


def test_hcqt_output_not_present_initially():
    output_path = "/workspace/hcqt.npy"
    assert not os.path.exists(output_path), (
        f"Expected {output_path} to be absent before evaluation; the executor must create it."
    )
