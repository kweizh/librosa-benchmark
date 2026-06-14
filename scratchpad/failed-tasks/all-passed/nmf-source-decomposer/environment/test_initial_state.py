import importlib
import os


WORKSPACE = "/workspace"
MIXTURE_WAV = "/workspace/mixture.wav"


def test_librosa_importable():
    try:
        importlib.import_module("librosa")
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"librosa is not importable in the task environment: {exc}"
        )


def test_sklearn_importable():
    try:
        importlib.import_module("sklearn.decomposition")
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"sklearn.decomposition is not importable in the task environment: {exc}"
        )


def test_workspace_dir_exists():
    assert os.path.isdir(WORKSPACE), (
        f"Expected workspace directory {WORKSPACE} to exist before the task starts."
    )


def test_mixture_wav_exists():
    assert os.path.isfile(MIXTURE_WAV), (
        f"Expected input mixture audio file {MIXTURE_WAV} to be present before the task starts."
    )


def test_mixture_wav_is_readable_mono_audio():
    librosa = importlib.import_module("librosa")
    try:
        y, sr = librosa.load(MIXTURE_WAV, sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"Failed to load {MIXTURE_WAV} with librosa.load: {exc}"
        )
    assert getattr(y, "ndim", 0) == 1, (
        f"Loaded mixture from {MIXTURE_WAV} must be mono (1-D), got ndim={getattr(y, 'ndim', None)}."
    )
    assert getattr(y, "size", 0) > 0, (
        f"Loaded waveform from {MIXTURE_WAV} is empty; expected non-empty audio."
    )
    assert sr == 22050, (
        f"Expected sample rate of {MIXTURE_WAV} to be 22050 Hz, got {sr!r}."
    )


def test_component_outputs_not_yet_created():
    for i in range(4):
        path = os.path.join(WORKSPACE, f"component_{i}.wav")
        assert not os.path.exists(path), (
            f"Expected {path} to NOT exist before the task starts; the agent must create it."
        )
    components_json = os.path.join(WORKSPACE, "components.json")
    assert not os.path.exists(components_json), (
        f"Expected {components_json} to NOT exist before the task starts; the agent must create it."
    )
