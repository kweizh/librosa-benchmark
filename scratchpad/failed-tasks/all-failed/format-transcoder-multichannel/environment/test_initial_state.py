import os
import importlib

import pytest


WORKSPACE = "/workspace"
INPUT_FLAC = "/workspace/input.flac"
OUTPUT_WAV = "/workspace/output.wav"
TRANSCODE_META = "/workspace/transcode_meta.json"


def test_librosa_importable():
    try:
        importlib.import_module("librosa")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"librosa is not importable in the task environment: {exc}")


def test_soundfile_importable():
    try:
        importlib.import_module("soundfile")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"soundfile is not importable in the task environment: {exc}")


def test_soxr_importable():
    try:
        importlib.import_module("soxr")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(
            f"soxr is not importable; the task requires the soxr_hq backend: {exc}"
        )


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), (
        f"Expected workspace directory {WORKSPACE} to exist before the task starts."
    )


def test_input_flac_exists():
    assert os.path.isfile(INPUT_FLAC), (
        f"Expected input audio file {INPUT_FLAC} to be present before the task starts."
    )


def test_input_flac_is_stereo_44100():
    soundfile = importlib.import_module("soundfile")
    try:
        info = soundfile.info(INPUT_FLAC)
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to read {INPUT_FLAC} with soundfile.info: {exc}")
    assert info.samplerate == 44100, (
        f"Expected input sample rate 44100 Hz, got {info.samplerate}."
    )
    assert info.channels == 2, (
        f"Expected stereo input (2 channels), got {info.channels} channels."
    )
    assert info.frames > 0, (
        f"Input file {INPUT_FLAC} contains no audio samples."
    )


def test_output_wav_not_yet_created():
    assert not os.path.exists(OUTPUT_WAV), (
        f"Expected {OUTPUT_WAV} to NOT exist before the task starts; the agent must create it."
    )


def test_transcode_meta_not_yet_created():
    assert not os.path.exists(TRANSCODE_META), (
        f"Expected {TRANSCODE_META} to NOT exist before the task starts; the agent must create it."
    )
