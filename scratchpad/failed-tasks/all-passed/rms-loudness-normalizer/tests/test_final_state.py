import json
import math
import os

import numpy as np
import pytest
import soundfile as sf


WORKSPACE = "/workspace"
INPUTS_DIR = "/workspace/inputs"
OUTPUTS_DIR = "/workspace/outputs"
REPORT_PATH = "/workspace/report.json"

TARGET_RMS_DB = -20.0
TARGET_RMS_LINEAR = 10 ** (TARGET_RMS_DB / 20.0)  # 0.1
RMS_TOLERANCE_DB = 0.5
PEAK_HEADROOM_TOLERANCE = 0.01  # within 1% of 1.0 for clip-avoid case
PEAK_HARD_LIMIT = 1.0001


EXPECTED_INPUT_FILES = [
    "quiet_mono_16k.wav",
    "loud_mono_44100.wav",
    "pink_stereo_48k.wav",
    "tone_mono_22050.wav",
]


def _load_channels_first(path: str):
    """Return waveform as ndarray (channels-first) and the sample rate."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    # soundfile returns (n_samples, n_channels); convert to channels-first.
    y = np.asarray(data, dtype=np.float64).T
    return y, sr


def _global_rms_linear(y: np.ndarray) -> float:
    """Compute the global sample-based RMS across all channels and samples."""
    flat = np.asarray(y, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(flat * flat)))


def _to_db(x: float) -> float:
    if x <= 0.0:
        return -math.inf
    return 20.0 * math.log10(x)


def _global_peak(y: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(y, dtype=np.float64)))) if y.size else 0.0


@pytest.fixture(scope="module")
def report():
    assert os.path.isfile(REPORT_PATH), (
        f"Report file {REPORT_PATH} does not exist after the agent ran."
    )
    with open(REPORT_PATH, "r") as f:
        data = json.load(f)
    assert isinstance(data, dict), (
        f"Report at {REPORT_PATH} must be a JSON object, got {type(data).__name__}."
    )
    return data


def test_outputs_dir_exists():
    assert os.path.isdir(OUTPUTS_DIR), (
        f"Outputs directory {OUTPUTS_DIR} does not exist."
    )


@pytest.mark.parametrize("name", EXPECTED_INPUT_FILES)
def test_output_file_exists_with_matching_layout(name):
    in_path = os.path.join(INPUTS_DIR, name)
    out_path = os.path.join(OUTPUTS_DIR, name)
    assert os.path.isfile(out_path), f"Missing output file {out_path}."

    in_info = sf.info(in_path)
    out_info = sf.info(out_path)
    assert out_info.samplerate == in_info.samplerate, (
        f"{name}: sample rate mismatch. Input {in_info.samplerate} Hz, "
        f"output {out_info.samplerate} Hz."
    )
    assert out_info.channels == in_info.channels, (
        f"{name}: channel count mismatch. Input has {in_info.channels} channel(s), "
        f"output has {out_info.channels}."
    )


@pytest.mark.parametrize(
    "name",
    [
        "quiet_mono_16k.wav",
        "pink_stereo_48k.wav",
        "tone_mono_22050.wav",
    ],
)
def test_non_clipping_outputs_hit_target_rms(name):
    out_path = os.path.join(OUTPUTS_DIR, name)
    assert os.path.isfile(out_path), f"Output {out_path} missing."
    y, _sr = _load_channels_first(out_path)
    rms_lin = _global_rms_linear(y)
    rms_db = _to_db(rms_lin)
    assert abs(rms_db - TARGET_RMS_DB) <= RMS_TOLERANCE_DB, (
        f"{name}: output global RMS is {rms_db:.3f} dBFS, expected within "
        f"{RMS_TOLERANCE_DB} dB of {TARGET_RMS_DB} dBFS."
    )


def test_clipping_case_uses_maximum_headroom():
    name = "loud_mono_44100.wav"
    out_path = os.path.join(OUTPUTS_DIR, name)
    assert os.path.isfile(out_path), f"Output {out_path} missing."
    y, _sr = _load_channels_first(out_path)
    peak = _global_peak(y)
    assert (1.0 - PEAK_HEADROOM_TOLERANCE) <= peak <= PEAK_HARD_LIMIT, (
        f"{name}: clip-avoided output peak is {peak:.5f}, expected between "
        f"{1.0 - PEAK_HEADROOM_TOLERANCE:.5f} and {PEAK_HARD_LIMIT:.5f}."
    )


def test_report_has_entry_for_each_input(report):
    input_basenames = sorted(
        f for f in os.listdir(INPUTS_DIR) if f.lower().endswith(".wav")
    )
    report_keys = sorted(report.keys())
    assert report_keys == input_basenames, (
        f"Report keys {report_keys} do not match input filenames {input_basenames}."
    )


@pytest.mark.parametrize("name", EXPECTED_INPUT_FILES)
def test_report_entry_schema(report, name):
    assert name in report, f"Report missing entry for {name}."
    entry = report[name]
    assert isinstance(entry, dict), (
        f"Report entry for {name} must be a JSON object, got {type(entry).__name__}."
    )
    for key in ("orig_rms_db", "gain_db", "final_rms_db", "peak_after"):
        assert key in entry, f"Report entry for {name} missing key '{key}'."
        assert isinstance(entry[key], (int, float)) and not isinstance(
            entry[key], bool
        ), (
            f"Report entry for {name} key '{key}' must be a number, "
            f"got {type(entry[key]).__name__}."
        )


@pytest.mark.parametrize("name", EXPECTED_INPUT_FILES)
def test_report_gain_db_matches_rms_delta(report, name):
    entry = report[name]
    delta = float(entry["final_rms_db"]) - float(entry["orig_rms_db"])
    gain = float(entry["gain_db"])
    assert abs(gain - delta) <= RMS_TOLERANCE_DB, (
        f"{name}: report gain_db={gain:.4f} dB does not match "
        f"final_rms_db - orig_rms_db = {delta:.4f} dB "
        f"(tolerance {RMS_TOLERANCE_DB} dB)."
    )


@pytest.mark.parametrize("name", EXPECTED_INPUT_FILES)
def test_report_peak_after_not_clipped(report, name):
    entry = report[name]
    peak_after = float(entry["peak_after"])
    assert peak_after <= PEAK_HARD_LIMIT, (
        f"{name}: report peak_after={peak_after:.5f} exceeds {PEAK_HARD_LIMIT}, "
        "indicating clipping."
    )


@pytest.mark.parametrize("name", EXPECTED_INPUT_FILES)
def test_actual_output_peak_not_clipped(name):
    out_path = os.path.join(OUTPUTS_DIR, name)
    assert os.path.isfile(out_path), f"Output {out_path} missing."
    y, _sr = _load_channels_first(out_path)
    peak = _global_peak(y)
    assert peak <= PEAK_HARD_LIMIT, (
        f"{name}: output WAV has peak {peak:.5f} > {PEAK_HARD_LIMIT}, "
        "which means the output is clipped."
    )
