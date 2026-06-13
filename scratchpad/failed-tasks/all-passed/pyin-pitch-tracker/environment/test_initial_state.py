import os

import pytest


WORKSPACE = "/workspace"
INPUT_WAV = os.path.join(WORKSPACE, "input.wav")


def test_librosa_importable():
    try:
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover - import diagnostic only
        pytest.fail(f"Expected librosa to be importable, but got: {exc!r}")


def test_librosa_version_is_0_11_0():
    import librosa

    assert librosa.__version__ == "0.11.0", (
        f"Expected librosa==0.11.0 in the environment, got {librosa.__version__}."
    )


def test_librosa_pyin_callable():
    import librosa

    assert callable(getattr(librosa, "pyin", None)), (
        "Expected librosa.pyin to be available and callable in this environment."
    )


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_input_wav_exists():
    assert os.path.isfile(INPUT_WAV), (
        f"Expected the synthesized sine-sweep input at {INPUT_WAV}."
    )


def test_input_wav_is_readable_audio():
    import soundfile as sf

    info = sf.info(INPUT_WAV)
    assert info.samplerate == 22050, (
        f"Expected {INPUT_WAV} to be sampled at 22050 Hz, got {info.samplerate}."
    )
    assert info.channels == 1, (
        f"Expected {INPUT_WAV} to be mono, got {info.channels} channels."
    )
    assert info.frames > 0, f"Expected {INPUT_WAV} to contain audio samples."


def test_output_csv_not_present_yet():
    output_csv = os.path.join(WORKSPACE, "pitch.csv")
    assert not os.path.exists(output_csv), (
        f"{output_csv} must not exist before the agent runs the task."
    )
