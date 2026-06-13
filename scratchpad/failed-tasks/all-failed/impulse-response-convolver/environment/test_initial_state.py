import os

import numpy as np
import pytest
import soundfile as sf


WORKSPACE = "/workspace"
DRY_PATH = os.path.join(WORKSPACE, "dry.wav")
IR_PATH = os.path.join(WORKSPACE, "ir.wav")
WET_PATH = os.path.join(WORKSPACE, "wet.wav")


def test_librosa_importable():
    """The target library must be importable in the evaluation environment."""
    import librosa  # noqa: F401

    assert hasattr(librosa, "load"), "librosa.load is not available."
    assert hasattr(librosa, "resample"), "librosa.resample is not available."


def test_scipy_signal_available():
    """scipy.signal.fftconvolve must be available for the reference convolution."""
    from scipy.signal import fftconvolve  # noqa: F401


def test_soundfile_available():
    """soundfile must be available for reading/writing WAV files."""
    import soundfile  # noqa: F401


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_dry_wav_exists_and_readable():
    assert os.path.isfile(DRY_PATH), f"Dry input {DRY_PATH} does not exist."
    data, sr = sf.read(DRY_PATH, always_2d=False)
    assert sr > 0, "Dry sample rate must be positive."
    assert data.size > 0, "Dry signal must be non-empty."


def test_ir_wav_exists_and_readable():
    assert os.path.isfile(IR_PATH), f"Impulse response {IR_PATH} does not exist."
    data, sr = sf.read(IR_PATH, always_2d=False)
    assert sr > 0, "IR sample rate must be positive."
    assert data.size > 0, "IR signal must be non-empty."


def test_dry_and_ir_have_different_sample_rates():
    """The dry signal and IR are intentionally generated at different sample rates
    so the agent must resample the IR before convolving."""
    _, sr_dry = sf.read(DRY_PATH, always_2d=False)
    _, sr_ir = sf.read(IR_PATH, always_2d=False)
    assert sr_dry != sr_ir, (
        f"Initial state expects different sample rates for dry and IR, "
        f"but both are {sr_dry} Hz."
    )


def test_wet_output_not_yet_present():
    """The agent is responsible for producing /workspace/wet.wav."""
    assert not os.path.exists(WET_PATH), (
        f"Wet output {WET_PATH} should not exist before the task runs."
    )
