import os

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import fftconvolve

import librosa


WORKSPACE = "/workspace"
DRY_PATH = os.path.join(WORKSPACE, "dry.wav")
IR_PATH = os.path.join(WORKSPACE, "ir.wav")
WET_PATH = os.path.join(WORKSPACE, "wet.wav")

PEAK_TARGET = 0.95
PEAK_LOWER = 0.90
PEAK_UPPER = 1.0
LENGTH_TOLERANCE = 1
COSINE_THRESHOLD = 0.99


def _load_mono(path):
    """Load a WAV file as float32 mono at its native sample rate."""
    data, sr = sf.read(path, always_2d=False)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


@pytest.fixture(scope="module")
def signals():
    assert os.path.isfile(DRY_PATH), f"Dry input {DRY_PATH} does not exist."
    assert os.path.isfile(IR_PATH), f"Impulse response {IR_PATH} does not exist."
    assert os.path.isfile(WET_PATH), (
        f"Wet output {WET_PATH} was not produced by the agent."
    )

    dry, sr_dry = _load_mono(DRY_PATH)
    ir, sr_ir = _load_mono(IR_PATH)
    wet, sr_wet = _load_mono(WET_PATH)

    if sr_ir != sr_dry:
        ir_aligned = librosa.resample(
            ir.astype(np.float32), orig_sr=sr_ir, target_sr=sr_dry
        ).astype(np.float64)
    else:
        ir_aligned = ir

    return {
        "dry": dry,
        "sr_dry": sr_dry,
        "ir": ir,
        "sr_ir": sr_ir,
        "ir_aligned": ir_aligned,
        "wet": wet,
        "sr_wet": sr_wet,
    }


def test_wet_output_file_exists_and_readable():
    assert os.path.isfile(WET_PATH), (
        f"Expected wet output file at {WET_PATH}, but it does not exist."
    )
    data, sr = sf.read(WET_PATH, always_2d=False)
    assert sr > 0, "Wet output sample rate must be positive."
    assert np.asarray(data).size > 0, "Wet output must contain audio samples."


def test_wet_sample_rate_matches_dry(signals):
    assert signals["sr_wet"] == signals["sr_dry"], (
        f"Wet sample rate {signals['sr_wet']} must equal dry sample rate "
        f"{signals['sr_dry']}."
    )


def test_wet_length_matches_linear_convolution(signals):
    expected_length = len(signals["dry"]) + len(signals["ir_aligned"]) - 1
    actual_length = len(signals["wet"])
    assert abs(actual_length - expected_length) <= LENGTH_TOLERANCE, (
        f"Wet length {actual_length} must equal len(dry) + len(ir_aligned) - 1 "
        f"= {expected_length} within tolerance {LENGTH_TOLERANCE}."
    )


def test_wet_peak_amplitude_is_normalized(signals):
    peak = float(np.max(np.abs(signals["wet"])))
    assert PEAK_LOWER <= peak <= PEAK_UPPER, (
        f"Wet peak amplitude {peak:.4f} must lie in "
        f"[{PEAK_LOWER}, {PEAK_UPPER}] (target {PEAK_TARGET})."
    )


def test_wet_matches_reference_fft_convolution(signals):
    dry = signals["dry"]
    ir_aligned = signals["ir_aligned"]
    wet = signals["wet"]

    reference = fftconvolve(dry, ir_aligned, mode="full")
    ref_peak = float(np.max(np.abs(reference)))
    assert ref_peak > 0, "Reference convolution must have non-zero peak amplitude."
    reference_normalized = reference * (PEAK_TARGET / ref_peak)

    # Align lengths within the allowed 1-sample tolerance for fair comparison.
    n = min(len(wet), len(reference_normalized))
    assert abs(len(wet) - len(reference_normalized)) <= LENGTH_TOLERANCE, (
        f"Wet length {len(wet)} and reference length {len(reference_normalized)} "
        f"differ by more than {LENGTH_TOLERANCE} sample(s)."
    )
    wet_aligned = wet[:n]
    ref_aligned = reference_normalized[:n]

    wet_norm = float(np.linalg.norm(wet_aligned))
    ref_norm = float(np.linalg.norm(ref_aligned))
    assert wet_norm > 0, "Wet signal is empty / all zeros."
    assert ref_norm > 0, "Reference signal is empty / all zeros."

    cosine_similarity = float(
        np.dot(wet_aligned, ref_aligned) / (wet_norm * ref_norm)
    )
    assert cosine_similarity >= COSINE_THRESHOLD, (
        f"Cosine similarity {cosine_similarity:.4f} between wet output and reference "
        f"fftconvolve result must be >= {COSINE_THRESHOLD}."
    )


def test_wet_initial_samples_track_dry_input(signals):
    """Causality check: the first 100 samples of the wet signal should be a
    scaled copy of the first 100 samples of the dry signal, because in a typical
    impulse response the response begins at (or very near) sample 0."""
    dry = signals["dry"]
    wet = signals["wet"]
    head = 100
    assert len(dry) >= head and len(wet) >= head, (
        f"Both dry and wet signals must have at least {head} samples for the "
        f"causality check."
    )
    dry_head = dry[:head]
    wet_head = wet[:head]

    dry_energy = float(np.dot(dry_head, dry_head))
    assert dry_energy > 0, "Dry signal head has zero energy; cannot run causality check."
    scale = float(np.dot(wet_head, dry_head) / dry_energy)

    residual = float(np.mean(np.abs(wet_head - scale * dry_head)))
    wet_mean_abs = float(np.mean(np.abs(wet_head))) + 1e-9
    assert residual <= 0.05 * wet_mean_abs + 1e-6, (
        f"First {head} samples of wet should match a scaled copy of the first "
        f"{head} samples of dry (residual {residual:.6f} vs allowed "
        f"{0.05 * wet_mean_abs:.6f}). This indicates either an acausal output "
        f"or an incorrect convolution."
    )
