import os

import numpy as np


WORKSPACE = "/workspace"
MAGNITUDE_PATH = os.path.join(WORKSPACE, "magnitude.npy")


def test_librosa_importable():
    import librosa  # noqa: F401

    assert hasattr(librosa, "griffinlim"), (
        "librosa.griffinlim is not available; check the installed librosa version."
    )


def test_soundfile_importable():
    import soundfile  # noqa: F401


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_magnitude_file_exists():
    assert os.path.isfile(MAGNITUDE_PATH), (
        f"Required input file {MAGNITUDE_PATH} does not exist."
    )


def test_magnitude_file_is_valid_2d_float32_array():
    arr = np.load(MAGNITUDE_PATH)
    assert arr.ndim == 2, (
        f"Expected magnitude.npy to be a 2D array, got shape {arr.shape}."
    )
    assert arr.dtype == np.float32, (
        f"Expected magnitude.npy dtype to be float32, got {arr.dtype}."
    )
    assert np.all(arr >= 0), "Magnitude spectrogram values must be non-negative."
    assert np.isfinite(arr).all(), "Magnitude spectrogram must contain only finite values."


def test_magnitude_shape_matches_known_n_fft():
    arr = np.load(MAGNITUDE_PATH)
    n_bins = arr.shape[0]
    # n_bins == 1 + n_fft // 2; for n_fft=1024, n_bins == 513.
    assert n_bins == 513, (
        f"Expected magnitude first dim to be 513 (n_fft=1024), got {n_bins}."
    )


def test_reconstructed_output_does_not_exist_yet():
    output_path = os.path.join(WORKSPACE, "reconstructed.wav")
    assert not os.path.exists(output_path), (
        f"{output_path} should not exist before the task runs."
    )
