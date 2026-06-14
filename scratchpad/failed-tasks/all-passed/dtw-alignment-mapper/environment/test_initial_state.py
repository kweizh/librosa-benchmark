import importlib
import os


WORKSPACE = "/workspace"
REFERENCE_WAV = "/workspace/reference.wav"
COMPARISON_WAV = "/workspace/comparison.wav"
ALIGNMENT_JSON = "/workspace/alignment.json"


def test_librosa_importable():
    try:
        importlib.import_module("librosa")
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"librosa is not importable in the task environment: {exc}"
        )


def test_workspace_dir_exists():
    assert os.path.isdir(WORKSPACE), (
        f"Expected workspace directory {WORKSPACE} to exist before the task starts."
    )


def test_reference_wav_exists():
    assert os.path.isfile(REFERENCE_WAV), (
        f"Expected reference audio file {REFERENCE_WAV} to be present before the task starts."
    )


def test_comparison_wav_exists():
    assert os.path.isfile(COMPARISON_WAV), (
        f"Expected comparison audio file {COMPARISON_WAV} to be present before the task starts."
    )


def test_reference_wav_is_readable_audio():
    librosa = importlib.import_module("librosa")
    try:
        y, sr = librosa.load(REFERENCE_WAV, sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"Failed to load {REFERENCE_WAV} with librosa.load: {exc}"
        )
    assert getattr(y, "size", 0) > 0, (
        f"Loaded waveform from {REFERENCE_WAV} is empty; expected non-empty audio."
    )
    assert sr and sr > 0, (
        f"Loaded sample rate from {REFERENCE_WAV} is invalid: {sr!r}."
    )


def test_comparison_wav_is_readable_audio():
    librosa = importlib.import_module("librosa")
    try:
        y, sr = librosa.load(COMPARISON_WAV, sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"Failed to load {COMPARISON_WAV} with librosa.load: {exc}"
        )
    assert getattr(y, "size", 0) > 0, (
        f"Loaded waveform from {COMPARISON_WAV} is empty; expected non-empty audio."
    )
    assert sr and sr > 0, (
        f"Loaded sample rate from {COMPARISON_WAV} is invalid: {sr!r}."
    )


def test_alignment_json_not_yet_created():
    assert not os.path.exists(ALIGNMENT_JSON), (
        f"Expected {ALIGNMENT_JSON} to NOT exist before the task starts; the agent must create it."
    )
