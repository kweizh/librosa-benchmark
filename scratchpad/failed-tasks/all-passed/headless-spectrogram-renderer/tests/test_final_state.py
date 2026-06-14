import json
import os

import numpy as np
import pytest
from PIL import Image


FIGURE_PNG = "/workspace/figure.png"
FIGURE_META = "/workspace/figure_meta.json"

EXPECTED_TITLES = [
    "Waveform",
    "Linear STFT (dB)",
    "Log-Mel Spectrogram (dB)",
    "Chromagram",
]

MIN_FIGURE_BYTES = 50000
MIN_DIMENSION_PX = 800
MIN_DPI = 100.0
MIN_PIXEL_STD = 10.0


@pytest.fixture(scope="module")
def figure_bytes():
    assert os.path.isfile(FIGURE_PNG), (
        f"Expected output file {FIGURE_PNG} to exist after the task completes."
    )
    size = os.path.getsize(FIGURE_PNG)
    assert size >= MIN_FIGURE_BYTES, (
        f"{FIGURE_PNG} must be at least {MIN_FIGURE_BYTES} bytes (not blank), "
        f"got {size} bytes."
    )
    with open(FIGURE_PNG, "rb") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def figure_image(figure_bytes):
    try:
        img = Image.open(FIGURE_PNG)
        img.load()
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"Failed to open {FIGURE_PNG} as a PNG image with PIL: {exc}"
        )
    return img


@pytest.fixture(scope="module")
def meta():
    assert os.path.isfile(FIGURE_META), (
        f"Expected metadata file {FIGURE_META} to exist after the task completes."
    )
    with open(FIGURE_META, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{FIGURE_META} is not valid JSON: {exc}"
            )
    assert isinstance(data, dict), (
        f"{FIGURE_META} must be a JSON object, got {type(data).__name__}."
    )
    return data


def test_meta_has_exact_keys(meta):
    expected_keys = {"width_px", "height_px", "dpi", "panel_titles"}
    actual_keys = set(meta.keys())
    assert actual_keys == expected_keys, (
        f"{FIGURE_META} must contain exactly the keys {sorted(expected_keys)}, "
        f"got {sorted(actual_keys)}."
    )


def test_meta_dimensions_match_png(meta, figure_image):
    width_px = meta["width_px"]
    height_px = meta["height_px"]
    assert isinstance(width_px, int) and not isinstance(width_px, bool), (
        f"'width_px' must be an integer, got {width_px!r} ({type(width_px).__name__})."
    )
    assert isinstance(height_px, int) and not isinstance(height_px, bool), (
        f"'height_px' must be an integer, got {height_px!r} ({type(height_px).__name__})."
    )
    actual_width, actual_height = figure_image.size
    assert width_px == actual_width, (
        f"'width_px' ({width_px}) does not match actual PNG width ({actual_width})."
    )
    assert height_px == actual_height, (
        f"'height_px' ({height_px}) does not match actual PNG height ({actual_height})."
    )
    assert width_px >= MIN_DIMENSION_PX, (
        f"PNG width must be >= {MIN_DIMENSION_PX} px, got {width_px}."
    )
    assert height_px >= MIN_DIMENSION_PX, (
        f"PNG height must be >= {MIN_DIMENSION_PX} px, got {height_px}."
    )


def test_meta_dpi_threshold(meta):
    dpi = meta["dpi"]
    assert isinstance(dpi, (int, float)) and not isinstance(dpi, bool), (
        f"'dpi' must be numeric, got {dpi!r} ({type(dpi).__name__})."
    )
    assert float(dpi) >= MIN_DPI, (
        f"'dpi' must be >= {MIN_DPI}, got {dpi}."
    )


def test_meta_panel_titles_exact(meta):
    titles = meta["panel_titles"]
    assert isinstance(titles, list), (
        f"'panel_titles' must be a list, got {type(titles).__name__}."
    )
    assert len(titles) == 4, (
        f"'panel_titles' must contain exactly 4 entries, got {len(titles)}."
    )
    assert all(isinstance(t, str) for t in titles), (
        f"All 'panel_titles' entries must be strings, got {titles!r}."
    )
    assert titles == EXPECTED_TITLES, (
        f"'panel_titles' must be exactly {EXPECTED_TITLES!r}, got {titles!r}."
    )


def test_image_is_not_uniform(figure_image):
    gray = figure_image.convert("L")
    arr = np.asarray(gray, dtype=np.float64)
    std = float(arr.std())
    assert std > MIN_PIXEL_STD, (
        f"Rendered PNG appears to be uniform/blank: grayscale pixel std={std:.3f} "
        f"(must be > {MIN_PIXEL_STD})."
    )
