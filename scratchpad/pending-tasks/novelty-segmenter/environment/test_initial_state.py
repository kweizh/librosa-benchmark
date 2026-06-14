import importlib
import os


WORKSPACE_DIR = "/workspace"
INPUT_AUDIO_PATH = "/workspace/input.wav"


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE_DIR), (
        f"Workspace directory {WORKSPACE_DIR} does not exist."
    )


def test_librosa_importable():
    try:
        librosa = importlib.import_module("librosa")
    except Exception as exc:  # pragma: no cover - import error path
        raise AssertionError(f"Failed to import librosa: {exc}") from exc
    assert hasattr(librosa, "load"), "librosa.load is not available."


def test_numpy_importable():
    try:
        importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"Failed to import numpy: {exc}") from exc


def test_soundfile_importable():
    try:
        importlib.import_module("soundfile")
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"Failed to import soundfile: {exc}") from exc


def test_input_audio_present():
    assert os.path.isfile(INPUT_AUDIO_PATH), (
        f"Expected input audio file at {INPUT_AUDIO_PATH} to be provisioned by the environment."
    )
    size = os.path.getsize(INPUT_AUDIO_PATH)
    assert size > 0, f"Input audio file {INPUT_AUDIO_PATH} is empty."


def test_input_audio_is_a_valid_waveform():
    import soundfile as sf

    info = sf.info(INPUT_AUDIO_PATH)
    assert info.samplerate == 22050, (
        f"Expected /workspace/input.wav to have sample rate 22050 Hz, got {info.samplerate}."
    )
    assert info.channels == 1, (
        f"Expected /workspace/input.wav to be mono, got {info.channels} channels."
    )
    # The test fixture builds a ~15 s waveform (3 sections of 5 s each).
    assert info.frames >= 14 * info.samplerate, (
        f"Expected /workspace/input.wav to be at least ~14 s long, got {info.frames / info.samplerate:.2f} s."
    )
