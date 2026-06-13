import os

import numpy as np
import pytest
from PIL import Image

WORKSPACE = "/workspace"
CQT_DB_PATH = os.path.join(WORKSPACE, "cqt_db.npy")
CQT_PNG_PATH = os.path.join(WORKSPACE, "cqt.png")


@pytest.fixture(scope="module")
def cqt_db_array() -> np.ndarray:
    assert os.path.isfile(CQT_DB_PATH), (
        f"Expected beat-synchronous CQT numpy artifact at {CQT_DB_PATH}, but it does not exist."
    )
    arr = np.load(CQT_DB_PATH)
    return arr


@pytest.fixture(scope="module")
def cqt_png_image() -> Image.Image:
    assert os.path.isfile(CQT_PNG_PATH), (
        f"Expected CQT visualization PNG at {CQT_PNG_PATH}, but it does not exist."
    )
    try:
        img = Image.open(CQT_PNG_PATH)
        img.load()
    except Exception as exc:
        pytest.fail(f"Failed to open {CQT_PNG_PATH} as a PNG image: {exc!r}")
    assert img.format == "PNG", (
        f"Expected {CQT_PNG_PATH} to be a PNG file, got format={img.format!r}."
    )
    return img


def test_cqt_db_is_2d_float_array(cqt_db_array: np.ndarray):
    assert cqt_db_array.ndim == 2, (
        f"Expected cqt_db.npy to be a 2D array, got ndim={cqt_db_array.ndim} "
        f"and shape={cqt_db_array.shape}."
    )
    assert np.issubdtype(cqt_db_array.dtype, np.floating), (
        f"Expected cqt_db.npy to have a floating-point dtype, got dtype={cqt_db_array.dtype}."
    )


def test_cqt_db_has_no_nan_or_inf(cqt_db_array: np.ndarray):
    assert np.all(np.isfinite(cqt_db_array)), (
        "cqt_db.npy contains NaN or Inf entries; the dB CQT matrix must be fully finite."
    )


def test_cqt_db_has_enough_beat_columns(cqt_db_array: np.ndarray):
    n_beats = cqt_db_array.shape[1]
    assert n_beats >= 5, (
        f"Expected at least 5 beat columns in beat-synchronous CQT, got n_beats={n_beats} "
        f"with shape={cqt_db_array.shape}."
    )


def test_cqt_db_has_reasonable_n_bins(cqt_db_array: np.ndarray):
    n_bins = cqt_db_array.shape[0]
    # librosa.cqt defaults to 84 bins; require at least a couple of octaves' worth.
    assert n_bins >= 24, (
        f"Expected at least 24 CQT frequency bins, got n_bins={n_bins} "
        f"with shape={cqt_db_array.shape}."
    )


def test_cqt_db_values_are_in_db_range(cqt_db_array: np.ndarray):
    # amplitude_to_db with ref=np.max produces values <= 0 (within tiny floating tolerance).
    max_val = float(np.max(cqt_db_array))
    assert max_val <= 1e-3, (
        f"Expected max value of dB-scaled CQT to be approximately <= 0 "
        f"(ref=np.max convention), got max={max_val}."
    )

    neg_fraction = float(np.mean(cqt_db_array < 0.0))
    assert neg_fraction >= 0.80, (
        f"Expected at least 80% of dB CQT values to be negative, got "
        f"{neg_fraction * 100:.2f}% negative entries."
    )


def test_cqt_png_dimensions(cqt_png_image: Image.Image):
    width, height = cqt_png_image.size
    assert width >= 300, f"Expected cqt.png width >= 300 px, got width={width}."
    assert height >= 200, f"Expected cqt.png height >= 200 px, got height={height}."


def test_cqt_png_has_nontrivial_content(cqt_png_image: Image.Image):
    rgb = cqt_png_image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.float32)
    std = float(pixels.std())
    assert std > 10.0, (
        f"Expected cqt.png to contain non-trivial content "
        f"(pixel std > 10), got std={std:.4f}. The image looks blank or near-uniform."
    )
