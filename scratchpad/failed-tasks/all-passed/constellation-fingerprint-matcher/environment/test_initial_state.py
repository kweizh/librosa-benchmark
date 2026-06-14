import importlib
import os


WORKSPACE = "/workspace"
REFERENCE_WAV = "/workspace/reference.wav"
QUERY_WAV = "/workspace/query.wav"
MATCH_JSON = "/workspace/match.json"


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


def test_query_wav_exists():
    assert os.path.isfile(QUERY_WAV), (
        f"Expected query audio file {QUERY_WAV} to be present before the task starts."
    )


def test_reference_wav_is_readable_mono_22050():
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
    assert sr == 22050, (
        f"Reference audio sample rate must be 22050 Hz, got {sr}."
    )
    duration = float(len(y)) / float(sr)
    assert duration >= 20.0, (
        f"Reference audio duration must be at least 20 seconds, got {duration:.3f}s."
    )


def test_query_wav_is_readable_mono_22050():
    librosa = importlib.import_module("librosa")
    try:
        y, sr = librosa.load(QUERY_WAV, sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"Failed to load {QUERY_WAV} with librosa.load: {exc}"
        )
    assert getattr(y, "size", 0) > 0, (
        f"Loaded waveform from {QUERY_WAV} is empty; expected non-empty audio."
    )
    assert sr == 22050, (
        f"Query audio sample rate must be 22050 Hz, got {sr}."
    )
    duration = float(len(y)) / float(sr)
    assert duration >= 4.0, (
        f"Query audio duration must be at least 4 seconds, got {duration:.3f}s."
    )


def test_query_is_shorter_than_reference():
    librosa = importlib.import_module("librosa")
    y_ref, sr_ref = librosa.load(REFERENCE_WAV, sr=None, mono=True)
    y_qry, sr_qry = librosa.load(QUERY_WAV, sr=None, mono=True)
    ref_dur = float(len(y_ref)) / float(sr_ref)
    qry_dur = float(len(y_qry)) / float(sr_qry)
    assert qry_dur < ref_dur, (
        f"Query duration ({qry_dur:.3f}s) must be strictly less than reference "
        f"duration ({ref_dur:.3f}s)."
    )


def test_match_json_not_yet_created():
    assert not os.path.exists(MATCH_JSON), (
        f"Expected {MATCH_JSON} to NOT exist before the task starts; the agent must create it."
    )


def test_ground_truth_not_visible_in_workspace():
    gt_path = os.path.join(WORKSPACE, "ground_truth.json")
    assert not os.path.exists(gt_path), (
        f"Ground truth file {gt_path} must NOT be visible in the workspace during task execution."
    )
