import os
import glob


WORKSPACE = "/workspace"
INPUTS_DIR = "/workspace/inputs"

EXPECTED_INPUT_FILES = [
    "quiet_mono_16k.wav",
    "loud_mono_44100.wav",
    "pink_stereo_48k.wav",
    "tone_mono_22050.wav",
]


def test_librosa_importable():
    import librosa  # noqa: F401
    assert hasattr(librosa, "feature"), "librosa.feature submodule must be available."


def test_soundfile_importable():
    import soundfile  # noqa: F401


def test_numpy_importable():
    import numpy  # noqa: F401


def test_workspace_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_inputs_dir_exists():
    assert os.path.isdir(INPUTS_DIR), f"Inputs directory {INPUTS_DIR} does not exist."


def test_inputs_dir_contains_wavs():
    wavs = sorted(glob.glob(os.path.join(INPUTS_DIR, "*.wav")))
    assert len(wavs) >= 4, (
        f"Expected at least 4 prebuilt WAV files in {INPUTS_DIR}, found {len(wavs)}."
    )


def test_expected_input_files_present():
    for name in EXPECTED_INPUT_FILES:
        path = os.path.join(INPUTS_DIR, name)
        assert os.path.isfile(path), f"Required input file {path} is missing."


def test_input_files_are_valid_wav_with_expected_layout():
    import soundfile as sf

    expected = {
        "quiet_mono_16k.wav": {"samplerate": 16000, "channels": 1},
        "loud_mono_44100.wav": {"samplerate": 44100, "channels": 1},
        "pink_stereo_48k.wav": {"samplerate": 48000, "channels": 2},
        "tone_mono_22050.wav": {"samplerate": 22050, "channels": 1},
    }
    for name, meta in expected.items():
        path = os.path.join(INPUTS_DIR, name)
        info = sf.info(path)
        assert info.samplerate == meta["samplerate"], (
            f"{name}: expected samplerate {meta['samplerate']}, got {info.samplerate}."
        )
        assert info.channels == meta["channels"], (
            f"{name}: expected {meta['channels']} channel(s), got {info.channels}."
        )


def test_outputs_dir_not_yet_present_or_empty():
    out_dir = os.path.join(WORKSPACE, "outputs")
    if os.path.isdir(out_dir):
        leftovers = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
        assert leftovers == [], (
            f"Outputs directory {out_dir} should be empty before evaluation, "
            f"found: {leftovers}."
        )


def test_report_json_not_yet_present():
    report = os.path.join(WORKSPACE, "report.json")
    assert not os.path.exists(report), (
        f"{report} should not exist before the agent runs."
    )
