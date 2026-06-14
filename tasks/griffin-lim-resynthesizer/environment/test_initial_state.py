import importlib
import os


WORKSPACE = "/workspace"
INPUT_WAV = "/workspace/input.wav"
RECONSTRUCTED_WAV = "/workspace/reconstructed.wav"
METRICS_JSON = "/workspace/metrics.json"


def test_librosa_importable():
    try:
        importlib.import_module("librosa")
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"librosa is not importable in the task environment: {exc}"
        )


def test_soundfile_importable():
    try:
        importlib.import_module("soundfile")
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"soundfile is not importable in the task environment: {exc}"
        )


def test_workspace_dir_exists():
    assert os.path.isdir(WORKSPACE), (
        f"Expected workspace directory {WORKSPACE} to exist before the task starts."
    )


def test_input_wav_exists():
    assert os.path.isfile(INPUT_WAV), (
        f"Expected input audio file {INPUT_WAV} to be present before the task starts."
    )


def test_input_wav_is_readable_mono_22050():
    librosa = importlib.import_module("librosa")
    try:
        y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"Failed to load {INPUT_WAV} with librosa.load: {exc}"
        )
    assert getattr(y, "size", 0) > 0, (
        f"Loaded waveform from {INPUT_WAV} is empty; expected non-empty audio."
    )
    assert sr == 22050, (
        f"Expected input sample rate to be 22050 Hz, got {sr!r}."
    )
    # Expect roughly ~5 seconds of audio (1s to 30s tolerated to remain robust).
    duration = float(len(y)) / float(sr)
    assert 1.0 <= duration <= 30.0, (
        f"Expected input duration to be between 1s and 30s, got {duration:.3f}s."
    )


def test_reconstructed_wav_not_yet_created():
    assert not os.path.exists(RECONSTRUCTED_WAV), (
        f"Expected {RECONSTRUCTED_WAV} to NOT exist before the task starts; "
        "the agent must create it."
    )


def test_metrics_json_not_yet_created():
    assert not os.path.exists(METRICS_JSON), (
        f"Expected {METRICS_JSON} to NOT exist before the task starts; "
        "the agent must create it."
    )
