import os

import numpy as np
import pytest


OUTPUT_PATH = "/workspace/hcqt.npy"


@pytest.fixture(scope="module")
def hcqt_array() -> np.ndarray:
    assert os.path.isfile(OUTPUT_PATH), (
        f"Expected HCQT artifact at {OUTPUT_PATH}, but the file does not exist."
    )
    try:
        arr = np.load(OUTPUT_PATH, allow_pickle=False)
    except Exception as exc:
        pytest.fail(f"Failed to load {OUTPUT_PATH} as a NumPy array: {exc!r}")
    assert isinstance(arr, np.ndarray), (
        f"Loaded object from {OUTPUT_PATH} is not a numpy.ndarray; got type={type(arr).__name__}."
    )
    return arr


def test_hcqt_output_exists():
    assert os.path.isfile(OUTPUT_PATH), (
        f"Expected the HCQT artifact at {OUTPUT_PATH}, but it was not created."
    )


def test_hcqt_dtype_is_float(hcqt_array: np.ndarray):
    assert hcqt_array.dtype in (np.float32, np.float64), (
        f"Expected dtype float32 or float64 at {OUTPUT_PATH}, got {hcqt_array.dtype}."
    )


def test_hcqt_shape(hcqt_array: np.ndarray):
    assert hcqt_array.ndim == 3, (
        f"Expected 3D array (n_harmonics, n_bins, n_frames) at {OUTPUT_PATH}, "
        f"got ndim={hcqt_array.ndim} with shape={hcqt_array.shape}."
    )
    n_harmonics, n_bins, n_frames = hcqt_array.shape
    assert n_harmonics == 6, (
        f"Expected 6 harmonics on axis 0 of {OUTPUT_PATH}, got {n_harmonics}. "
        f"Required harmonic list is [0.5, 1, 2, 3, 4, 5]."
    )
    assert n_bins == 72, (
        f"Expected 72 CQT bins on axis 1 of {OUTPUT_PATH}, got {n_bins} "
        f"(bins_per_octave=12, n_bins=72 required)."
    )
    assert n_frames >= 10, (
        f"Expected at least 10 frames on axis 2 of {OUTPUT_PATH}, got {n_frames}."
    )


def test_hcqt_has_no_nan_or_inf(hcqt_array: np.ndarray):
    assert not np.isnan(hcqt_array).any(), (
        f"HCQT array at {OUTPUT_PATH} contains NaN values; dB conversion should be finite."
    )
    assert not np.isinf(hcqt_array).any(), (
        f"HCQT array at {OUTPUT_PATH} contains Inf values; dB conversion should be finite."
    )


def test_hcqt_peak_near_zero_db(hcqt_array: np.ndarray):
    # With ref=np.max in librosa.amplitude_to_db the per-input peak is mapped to 0 dB.
    # At least one (harmonic, frame) slice must therefore peak within 0.01 dB of 0.0.
    per_slice_max = hcqt_array.max(axis=1)  # shape: (n_harmonics, n_frames)
    closest_to_zero = float(np.max(per_slice_max))
    assert closest_to_zero <= 1e-3, (
        f"Expected the array maximum to be <= 0 dB (ref=np.max), got {closest_to_zero}."
    )
    assert closest_to_zero >= -1e-2, (
        f"Expected at least one (harmonic, frame) slice to peak within 0.01 of 0.0 dB, "
        f"but the closest peak observed was {closest_to_zero} dB. "
        f"Did you forget librosa.amplitude_to_db(..., ref=np.max)?"
    )


def test_hcqt_dynamic_range(hcqt_array: np.ndarray):
    minimum = float(hcqt_array.min())
    assert minimum < -40.0, (
        f"Expected the minimum dB value in {OUTPUT_PATH} to be < -40.0 dB, got {minimum}. "
        f"This indicates the dB conversion or stacking is not producing the expected dynamic range."
    )
