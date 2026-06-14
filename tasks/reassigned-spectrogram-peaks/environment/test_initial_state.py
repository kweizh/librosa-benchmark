import os
import importlib


WORKSPACE = "/workspace"
INPUT_WAV = "/workspace/input.wav"


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


def test_input_wav_exists():
    assert os.path.isfile(INPUT_WAV), (
        f"Expected input audio file {INPUT_WAV} to be present before the task starts."
    )


def test_input_wav_is_readable_audio():
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
        f"Expected sample rate of 22050 Hz for the prepared input, got {sr!r}."
    )
    duration = float(y.size) / float(sr)
    assert 3.0 <= duration <= 10.0, (
        f"Expected input duration in the 3-10s range, got {duration:.3f}s."
    )


def test_peaks_json_not_yet_created():
    peaks_path = os.path.join(WORKSPACE, "peaks.json")
    assert not os.path.exists(peaks_path), (
        f"Expected {peaks_path} to NOT exist before the task starts; the agent must create it."
    )
