import importlib
import os

import pytest

WORKSPACE = "/workspace"
REFERENCE_WAV = os.path.join(WORKSPACE, "reference.wav")
PERFORMANCE_WAV = os.path.join(WORKSPACE, "performance.wav")
ALIGNMENT_JSON = os.path.join(WORKSPACE, "alignment.json")


def test_librosa_importable():
    """librosa must be installed and importable in the task environment."""
    librosa = importlib.import_module("librosa")
    assert hasattr(librosa, "load"), "librosa.load is not available in the installed librosa."


def test_librosa_version_is_pinned():
    """The research plan pins librosa to v0.11.x."""
    librosa = importlib.import_module("librosa")
    version = getattr(librosa, "__version__", "")
    assert version.startswith("0.11."), (
        f"Expected librosa 0.11.x to be installed, found {version!r}."
    )


def test_soundfile_importable():
    """soundfile is required for WAV I/O in this task."""
    sf = importlib.import_module("soundfile")
    assert hasattr(sf, "read"), "soundfile.read is not available."


def test_numpy_importable():
    """numpy is required to manipulate audio buffers and the DTW warping path."""
    np = importlib.import_module("numpy")
    assert hasattr(np, "ndarray"), "numpy.ndarray is not available."


def test_workspace_directory_exists():
    """The task explicitly uses /workspace as the project path."""
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_reference_wav_exists():
    """The reference audio file must be prepared by the Dockerfile before the agent runs."""
    assert os.path.isfile(REFERENCE_WAV), (
        f"Reference audio file {REFERENCE_WAV} is missing from the initial environment."
    )


def test_performance_wav_exists():
    """The performance audio file must be prepared by the Dockerfile before the agent runs."""
    assert os.path.isfile(PERFORMANCE_WAV), (
        f"Performance audio file {PERFORMANCE_WAV} is missing from the initial environment."
    )


def test_reference_wav_is_readable_audio():
    """The reference WAV must be a non-empty mono audio file."""
    sf = importlib.import_module("soundfile")
    data, sr = sf.read(REFERENCE_WAV)
    assert sr > 0, f"Reference sample rate {sr} is invalid."
    assert getattr(data, "size", 0) > 0, "Reference audio file contains no samples."


def test_performance_wav_is_readable_audio():
    """The performance WAV must be a non-empty mono audio file."""
    sf = importlib.import_module("soundfile")
    data, sr = sf.read(PERFORMANCE_WAV)
    assert sr > 0, f"Performance sample rate {sr} is invalid."
    assert getattr(data, "size", 0) > 0, "Performance audio file contains no samples."


def test_alignment_output_not_yet_created():
    """The executor must produce /workspace/alignment.json; it must NOT exist beforehand."""
    assert not os.path.exists(ALIGNMENT_JSON), (
        f"Unexpected pre-existing alignment output at {ALIGNMENT_JSON}."
    )
