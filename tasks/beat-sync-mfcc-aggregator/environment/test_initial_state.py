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


def test_numpy_importable():
    try:
        importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"numpy is not importable in the task environment: {exc}"
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
    assert sr and sr > 0, (
        f"Loaded sample rate from {INPUT_WAV} is invalid: {sr!r}."
    )


def test_beat_mfcc_npz_not_yet_created():
    out_path = os.path.join(WORKSPACE, "beat_mfcc.npz")
    assert not os.path.exists(out_path), (
        f"Expected {out_path} to NOT exist before the task starts; the agent must create it."
    )


def test_beats_json_not_yet_created():
    out_path = os.path.join(WORKSPACE, "beats.json")
    assert not os.path.exists(out_path), (
        f"Expected {out_path} to NOT exist before the task starts; the agent must create it."
    )
