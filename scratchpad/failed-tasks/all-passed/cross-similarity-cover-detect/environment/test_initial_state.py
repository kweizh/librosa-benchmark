import json
import os
import importlib


WORKSPACE = "/workspace"
TRACK_A = "/workspace/track_a.wav"
TRACK_B = "/workspace/track_b.wav"
GROUND_TRUTH = "/workspace/ground_truth.json"
OUTPUT_JSON = "/workspace/cover_decision.json"


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


def test_track_a_exists():
    assert os.path.isfile(TRACK_A), (
        f"Expected fixture audio {TRACK_A} to be present before the task starts."
    )


def test_track_b_exists():
    assert os.path.isfile(TRACK_B), (
        f"Expected fixture audio {TRACK_B} to be present before the task starts."
    )


def test_tracks_are_readable_audio():
    librosa = importlib.import_module("librosa")
    for path in (TRACK_A, TRACK_B):
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
        except Exception as exc:  # pragma: no cover - failure path
            raise AssertionError(
                f"Failed to load {path} with librosa.load: {exc}"
            )
        assert getattr(y, "size", 0) > 0, (
            f"Loaded waveform from {path} is empty; expected non-empty audio."
        )
        assert sr and sr > 0, (
            f"Loaded sample rate from {path} is invalid: {sr!r}."
        )


def test_ground_truth_file_exists_and_valid():
    assert os.path.isfile(GROUND_TRUTH), (
        f"Expected ground-truth file {GROUND_TRUTH} to be present before the task starts."
    )
    with open(GROUND_TRUTH, "r") as fh:
        data = json.load(fh)
    assert isinstance(data, dict), (
        f"{GROUND_TRUTH} must be a JSON object, got: {type(data).__name__}."
    )
    assert "is_cover" in data, (
        f"{GROUND_TRUTH} must contain key 'is_cover'."
    )
    assert isinstance(data["is_cover"], bool), (
        f"{GROUND_TRUTH} 'is_cover' must be a boolean, got: {data['is_cover']!r}."
    )


def test_output_not_yet_created():
    assert not os.path.exists(OUTPUT_JSON), (
        f"Expected {OUTPUT_JSON} to NOT exist before the task starts; the agent must create it."
    )
