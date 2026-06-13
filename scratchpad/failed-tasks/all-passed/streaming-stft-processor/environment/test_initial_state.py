import importlib
import os

import pytest


WORKSPACE = "/workspace"
INPUT_PATH = os.path.join(WORKSPACE, "input.wav")
OUTPUT_PATH = os.path.join(WORKSPACE, "rms_stream.csv")


def test_librosa_importable():
    try:
        librosa = importlib.import_module("librosa")
    except ImportError as exc:  # pragma: no cover - defensive guard
        pytest.fail(f"librosa is not importable in the task environment: {exc}")
    assert hasattr(librosa, "stream"), "librosa.stream is required but not available."


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_input_wav_exists():
    assert os.path.isfile(INPUT_PATH), (
        f"Expected input audio at {INPUT_PATH}, but the file is missing."
    )


def test_input_wav_is_22050_mono():
    import soundfile as sf

    with sf.SoundFile(INPUT_PATH) as f:
        sr = f.samplerate
        channels = f.channels
        frames = f.frames
    assert sr == 22050, f"Expected sample rate 22050 Hz, got {sr} Hz."
    assert channels == 1, f"Expected mono input, got {channels} channels."
    assert frames > 22050, (
        "Input WAV is too short; expected at least one second of audio."
    )


def test_output_csv_not_yet_created():
    assert not os.path.exists(OUTPUT_PATH), (
        f"{OUTPUT_PATH} must not exist before the executor runs the task."
    )
