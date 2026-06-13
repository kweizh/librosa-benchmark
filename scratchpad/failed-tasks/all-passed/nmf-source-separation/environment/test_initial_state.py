import os

import pytest


WORKSPACE_DIR = "/workspace"
INPUT_AUDIO = os.path.join(WORKSPACE_DIR, "input.wav")


def test_librosa_importable():
    try:
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"librosa is not importable in the task environment: {exc}")


def test_soundfile_importable():
    try:
        import soundfile  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"soundfile is not importable in the task environment: {exc}")


def test_sklearn_decomposition_importable():
    try:
        from sklearn.decomposition import NMF  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(
            f"sklearn.decomposition.NMF is not importable in the task environment: {exc}"
        )


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE_DIR), (
        f"Workspace directory {WORKSPACE_DIR} does not exist."
    )


def test_input_audio_exists():
    assert os.path.isfile(INPUT_AUDIO), (
        f"Input audio file {INPUT_AUDIO} does not exist."
    )


def test_input_audio_is_valid_wav():
    import soundfile as sf

    try:
        data, sr = sf.read(INPUT_AUDIO)
    except Exception as exc:
        pytest.fail(f"Could not read input audio {INPUT_AUDIO}: {exc}")

    assert sr > 0, f"Input audio at {INPUT_AUDIO} has invalid sample rate {sr}."
    assert len(data) > 0, f"Input audio at {INPUT_AUDIO} is empty."


def test_component_outputs_not_yet_present():
    # The four component WAV files are created by the executor; they must not
    # exist in the initial state.
    for i in range(4):
        path = os.path.join(WORKSPACE_DIR, f"component_{i}.wav")
        assert not os.path.exists(path), (
            f"Output {path} must not exist before the executor runs."
        )
