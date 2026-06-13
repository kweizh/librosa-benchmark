import json
import os
import wave

WORKSPACE = "/workspace"
INPUT_WAV = os.path.join(WORKSPACE, "input.wav")
SEGMENTS_JSON = os.path.join(WORKSPACE, "segments.json")


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_librosa_importable():
    import librosa  # noqa: F401

    assert hasattr(
        librosa, "__version__"
    ), "librosa is importable but has no __version__ attribute."


def test_librosa_version_is_pinned():
    import librosa

    assert librosa.__version__.startswith(
        "0.11."
    ), f"Expected librosa 0.11.x, got {librosa.__version__}."


def test_numpy_importable():
    import numpy  # noqa: F401


def test_scipy_importable():
    import scipy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import scipy.sparse.csgraph  # noqa: F401


def test_sklearn_importable():
    import sklearn  # noqa: F401
    import sklearn.cluster  # noqa: F401


def test_input_audio_present():
    assert os.path.isfile(
        INPUT_WAV
    ), f"Pre-baked input audio file {INPUT_WAV} is missing from the image."


def test_input_audio_is_valid_wav():
    with wave.open(INPUT_WAV, "rb") as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
    assert n_frames > 0, f"{INPUT_WAV} contains zero audio frames."
    assert framerate > 0, f"{INPUT_WAV} has an invalid sample rate ({framerate})."
    duration = n_frames / float(framerate)
    assert duration > 10.0, (
        f"{INPUT_WAV} is too short for a structural segmentation task "
        f"(duration={duration:.2f}s)."
    )


def test_input_audio_loadable_with_librosa():
    import librosa

    duration = librosa.get_duration(path=INPUT_WAV)
    assert duration > 10.0, (
        f"librosa reported an unexpectedly short duration for {INPUT_WAV}: "
        f"{duration:.2f}s."
    )


def test_segments_output_not_yet_created():
    assert not os.path.exists(SEGMENTS_JSON), (
        f"Output artifact {SEGMENTS_JSON} must not exist before the agent runs; "
        "found a stale file in the initial state."
    )


def test_segments_output_directory_is_writable():
    probe = os.path.join(WORKSPACE, ".write_probe")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        with open(probe) as f:
            assert (
                f.read() == "ok"
            ), f"Probe file {probe} did not round-trip its contents."
    finally:
        if os.path.exists(probe):
            os.remove(probe)


def test_segments_json_schema_shape_documented():
    # Smoke test that json import works and the expected schema shape is a list of
    # dicts with float keys "start" and "end".
    example = [{"start": 0.0, "end": 1.0}]
    serialized = json.dumps(example)
    parsed = json.loads(serialized)
    assert isinstance(parsed, list) and parsed[0].keys() == {"start", "end"}
