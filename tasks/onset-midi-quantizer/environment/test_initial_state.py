import os
import wave


WORKSPACE = "/workspace"
INPUT_WAV = os.path.join(WORKSPACE, "input.wav")


def test_librosa_importable():
    import librosa  # noqa: F401

    assert hasattr(librosa, "load"), "librosa.load is missing from the installed librosa package."
    assert hasattr(librosa, "pyin"), "librosa.pyin is missing from the installed librosa package."
    assert hasattr(librosa, "hz_to_midi"), "librosa.hz_to_midi is missing from the installed librosa package."
    assert hasattr(librosa.onset, "onset_detect"), "librosa.onset.onset_detect is missing from the installed librosa package."


def test_workspace_dir_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_input_wav_exists():
    assert os.path.isfile(INPUT_WAV), f"Input audio file {INPUT_WAV} does not exist."


def test_input_wav_is_readable_wav():
    with wave.open(INPUT_WAV, "rb") as wav:
        assert wav.getnchannels() >= 1, "Input WAV must have at least one channel."
        assert wav.getframerate() > 0, "Input WAV must have a positive sample rate."
        assert wav.getnframes() > 0, "Input WAV must contain at least one audio frame."
