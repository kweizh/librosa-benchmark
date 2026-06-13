import json
import os
import wave

WORKSPACE_DIR = "/workspace"
INPUT_WAV = os.path.join(WORKSPACE_DIR, "input.wav")
BEATS_JSON = os.path.join(WORKSPACE_DIR, "beats.json")


def test_librosa_importable():
    import librosa  # noqa: F401

    assert hasattr(librosa, "beat"), "librosa.beat submodule should be available."
    assert hasattr(librosa.beat, "plp"), (
        "librosa.beat.plp must exist for the PLP tempo tracker task."
    )


def test_librosa_feature_tempo_available():
    import librosa.feature

    assert hasattr(librosa.feature, "tempo"), (
        "librosa.feature.tempo must be available in librosa 0.11.0."
    )


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE_DIR), (
        f"Workspace directory {WORKSPACE_DIR} must exist before the task starts."
    )


def test_input_wav_exists():
    assert os.path.isfile(INPUT_WAV), (
        f"Input audio file {INPUT_WAV} must be present before the task starts."
    )


def test_input_wav_is_readable():
    with wave.open(INPUT_WAV, "rb") as wav:
        n_frames = wav.getnframes()
        framerate = wav.getframerate()
    assert n_frames > 0, f"Input WAV {INPUT_WAV} must contain audio samples."
    assert framerate > 0, f"Input WAV {INPUT_WAV} must have a valid sample rate."


def test_beats_json_not_yet_produced():
    # The agent is responsible for creating beats.json; it must not exist yet.
    assert not os.path.exists(BEATS_JSON), (
        f"{BEATS_JSON} should not exist before the task runs."
    )
    # Sanity: json module is importable in the verifier environment.
    json.dumps({"ok": True})
