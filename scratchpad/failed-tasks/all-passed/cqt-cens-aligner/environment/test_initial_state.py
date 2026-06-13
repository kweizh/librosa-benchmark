import os

import pytest

WORKSPACE = "/workspace"
INPUT_WAV = os.path.join(WORKSPACE, "input.wav")


def test_librosa_importable():
    try:
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"librosa is not importable: {exc!r}")


def test_numpy_importable():
    try:
        import numpy  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"numpy is not importable: {exc!r}")


def test_matplotlib_importable():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"matplotlib is not importable: {exc!r}")


def test_pil_importable():
    try:
        import PIL.Image  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Pillow (PIL) is not importable: {exc!r}")


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_input_wav_exists():
    assert os.path.isfile(INPUT_WAV), f"Input audio file {INPUT_WAV} does not exist."


def test_input_wav_non_empty():
    assert os.path.getsize(INPUT_WAV) > 1024, (
        f"Input audio file {INPUT_WAV} is unexpectedly small."
    )


def test_input_wav_is_loadable():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    assert y.ndim == 1 and y.size > 0, "Loaded audio waveform is empty."
    assert sr > 0, "Loaded audio sample rate is not positive."
