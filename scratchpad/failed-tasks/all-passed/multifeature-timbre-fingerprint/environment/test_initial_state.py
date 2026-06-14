import importlib
import json
import os


WORKSPACE = "/workspace"
INPUT_WAV = "/workspace/input.wav"
CENTROIDS_JSON = "/workspace/centroids.json"
FEATURES_JSON = "/workspace/features.json"

EXPECTED_LABELS = {"rock", "classical", "jazz"}


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
        f"Expected sample rate 22050 for {INPUT_WAV}, got {sr!r}."
    )


def test_centroids_json_exists_and_has_three_labels():
    assert os.path.isfile(CENTROIDS_JSON), (
        f"Expected centroids file {CENTROIDS_JSON} to be present before the task starts."
    )
    with open(CENTROIDS_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{CENTROIDS_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, dict), (
        f"{CENTROIDS_JSON} must be a JSON object, got: {type(data).__name__}."
    )
    assert set(data.keys()) == EXPECTED_LABELS, (
        f"Expected centroid keys {sorted(EXPECTED_LABELS)} in {CENTROIDS_JSON}, "
        f"got: {sorted(data.keys())}."
    )
    for label, vector in data.items():
        assert isinstance(vector, list), (
            f"Centroid for label {label!r} must be a JSON array, got: {type(vector).__name__}."
        )
        assert len(vector) == 72, (
            f"Centroid for label {label!r} must have length 72, got {len(vector)}."
        )
        for idx, value in enumerate(vector):
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"Centroid {label!r}[{idx}] must be numeric, got: {value!r}."
            )


def test_features_json_not_yet_created():
    assert not os.path.exists(FEATURES_JSON), (
        f"Expected {FEATURES_JSON} to NOT exist before the task starts; the agent must create it."
    )
