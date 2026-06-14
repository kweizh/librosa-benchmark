import importlib
import os


WORKSPACE = "/workspace"
INPUT_WAV = "/workspace/input.wav"


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


def test_input_wav_is_readable_mono_audio():
    librosa = importlib.import_module("librosa")
    try:
        y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"Failed to load {INPUT_WAV} with librosa.load: {exc}"
        )
    assert getattr(y, "ndim", 0) == 1, (
        f"Expected input waveform from {INPUT_WAV} to be mono (1-D), got ndim={getattr(y, 'ndim', None)}."
    )
    assert getattr(y, "size", 0) > 0, (
        f"Loaded waveform from {INPUT_WAV} is empty; expected non-empty audio."
    )
    assert sr and sr > 0, (
        f"Loaded sample rate from {INPUT_WAV} is invalid: {sr!r}."
    )


def test_augmented_wav_not_yet_created():
    augmented_path = os.path.join(WORKSPACE, "augmented.wav")
    assert not os.path.exists(augmented_path), (
        f"Expected {augmented_path} to NOT exist before the task starts; the agent must create it."
    )


def test_aug_meta_json_not_yet_created():
    meta_path = os.path.join(WORKSPACE, "aug_meta.json")
    assert not os.path.exists(meta_path), (
        f"Expected {meta_path} to NOT exist before the task starts; the agent must create it."
    )
