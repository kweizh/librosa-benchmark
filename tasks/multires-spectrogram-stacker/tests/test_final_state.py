import json
import os

import numpy as np
import pytest


SPEC_STACK = "/workspace/spec_stack.npz"
SPEC_META = "/workspace/spec_meta.json"
INPUT_WAV = "/workspace/input.wav"

REQUIRED_META_KEYS = (
    "n_frames",
    "hop_length",
    "sample_rate",
    "stft_freqs",
    "cqt_freqs",
    "vqt_freqs",
    "n_fft",
    "cqt_n_bins",
    "cqt_bins_per_octave",
    "vqt_n_bins",
    "vqt_bins_per_octave",
)

REQUIRED_NPZ_KEYS = ("stft_db", "cqt_db", "vqt_db")


@pytest.fixture(scope="module")
def stack():
    assert os.path.isfile(SPEC_STACK), (
        f"Expected output archive {SPEC_STACK} to exist after the task completes."
    )
    try:
        data = np.load(SPEC_STACK, allow_pickle=False)
    except Exception as exc:
        raise AssertionError(
            f"Failed to load {SPEC_STACK} as a valid .npz archive: {exc}"
        )
    for key in REQUIRED_NPZ_KEYS:
        assert key in data.files, (
            f"{SPEC_STACK} is missing required array '{key}'. Found: {sorted(data.files)}."
        )
    arrays = {key: np.asarray(data[key]) for key in REQUIRED_NPZ_KEYS}
    return arrays


@pytest.fixture(scope="module")
def meta():
    assert os.path.isfile(SPEC_META), (
        f"Expected metadata file {SPEC_META} to exist after the task completes."
    )
    with open(SPEC_META, "r") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{SPEC_META} is not valid JSON: {exc}")
    assert isinstance(payload, dict), (
        f"{SPEC_META} top-level must be a JSON object, got: {type(payload).__name__}."
    )
    for key in REQUIRED_META_KEYS:
        assert key in payload, (
            f"{SPEC_META} is missing required key '{key}'. Found keys: {sorted(payload.keys())}."
        )
    return payload


def _as_int(value, key):
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"Metadata key '{key}' must be an integer, got {type(value).__name__}: {value!r}."
    )
    assert value > 0, f"Metadata key '{key}' must be positive, got {value}."
    return int(value)


def test_metadata_scalar_types_and_positivity(meta):
    for key in (
        "n_frames",
        "hop_length",
        "sample_rate",
        "n_fft",
        "cqt_n_bins",
        "cqt_bins_per_octave",
        "vqt_n_bins",
        "vqt_bins_per_octave",
    ):
        _as_int(meta[key], key)


def test_metadata_freq_lists_are_numeric(meta):
    for key in ("stft_freqs", "cqt_freqs", "vqt_freqs"):
        vec = meta[key]
        assert isinstance(vec, list) and len(vec) > 0, (
            f"Metadata key '{key}' must be a non-empty JSON list, got {type(vec).__name__}."
        )
        for i, v in enumerate(vec):
            assert isinstance(v, (int, float)) and not isinstance(v, bool), (
                f"Metadata key '{key}' element {i} must be numeric, got {type(v).__name__}: {v!r}."
            )


def test_metadata_sample_rate_matches_input_wav(meta):
    import librosa

    _, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    assert int(meta["sample_rate"]) == int(sr), (
        f"Metadata sample_rate ({meta['sample_rate']}) does not match the input WAV "
        f"sample rate ({sr})."
    )


def test_arrays_are_two_dimensional_and_finite(stack):
    for key, arr in stack.items():
        assert arr.ndim == 2, (
            f"Array '{key}' must be 2-D, got shape {arr.shape}."
        )
        assert np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer), (
            f"Array '{key}' must be real-valued, got dtype {arr.dtype}."
        )
        assert np.isfinite(arr).all(), (
            f"Array '{key}' must contain only finite values (no NaN/Inf)."
        )


def test_arrays_share_common_time_grid(stack, meta):
    n_frames = _as_int(meta["n_frames"], "n_frames")
    for key, arr in stack.items():
        assert arr.shape[1] == n_frames, (
            f"Array '{key}' trailing dimension ({arr.shape[1]}) must equal metadata n_frames ({n_frames})."
        )


def test_stft_row_count_matches_n_fft(stack, meta):
    n_fft = _as_int(meta["n_fft"], "n_fft")
    expected_rows = 1 + n_fft // 2
    assert stack["stft_db"].shape[0] == expected_rows, (
        f"stft_db.shape[0] ({stack['stft_db'].shape[0]}) must equal 1 + n_fft // 2 "
        f"({expected_rows}) for n_fft={n_fft}."
    )


def test_cqt_row_count_matches_metadata(stack, meta):
    cqt_n_bins = _as_int(meta["cqt_n_bins"], "cqt_n_bins")
    assert stack["cqt_db"].shape[0] == cqt_n_bins, (
        f"cqt_db.shape[0] ({stack['cqt_db'].shape[0]}) must equal cqt_n_bins ({cqt_n_bins})."
    )


def test_vqt_row_count_matches_metadata(stack, meta):
    vqt_n_bins = _as_int(meta["vqt_n_bins"], "vqt_n_bins")
    assert stack["vqt_db"].shape[0] == vqt_n_bins, (
        f"vqt_db.shape[0] ({stack['vqt_db'].shape[0]}) must equal vqt_n_bins ({vqt_n_bins})."
    )


def test_frequency_vector_lengths(stack, meta):
    assert len(meta["stft_freqs"]) == stack["stft_db"].shape[0], (
        f"len(stft_freqs)={len(meta['stft_freqs'])} must equal stft_db.shape[0]={stack['stft_db'].shape[0]}."
    )
    assert len(meta["cqt_freqs"]) == stack["cqt_db"].shape[0], (
        f"len(cqt_freqs)={len(meta['cqt_freqs'])} must equal cqt_db.shape[0]={stack['cqt_db'].shape[0]}."
    )
    assert len(meta["vqt_freqs"]) == stack["vqt_db"].shape[0], (
        f"len(vqt_freqs)={len(meta['vqt_freqs'])} must equal vqt_db.shape[0]={stack['vqt_db'].shape[0]}."
    )


def test_stft_freqs_match_librosa_helper(meta):
    import librosa

    sr = _as_int(meta["sample_rate"], "sample_rate")
    n_fft = _as_int(meta["n_fft"], "n_fft")
    expected = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    actual = np.asarray(meta["stft_freqs"], dtype=float)
    assert actual.shape == expected.shape, (
        f"stft_freqs shape {actual.shape} does not match "
        f"librosa.fft_frequencies output shape {expected.shape}."
    )
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-6), (
        "stft_freqs does not match librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft) "
        "element-wise."
    )


def test_cqt_freqs_match_librosa_helper(meta):
    import librosa

    cqt_n_bins = _as_int(meta["cqt_n_bins"], "cqt_n_bins")
    cqt_bpo = _as_int(meta["cqt_bins_per_octave"], "cqt_bins_per_octave")
    cqt_freqs = np.asarray(meta["cqt_freqs"], dtype=float)
    fmin = float(cqt_freqs[0])
    assert fmin > 0, f"Recovered CQT fmin must be positive, got {fmin}."
    expected = librosa.cqt_frequencies(
        n_bins=cqt_n_bins, fmin=fmin, bins_per_octave=cqt_bpo
    )
    assert cqt_freqs.shape == expected.shape, (
        f"cqt_freqs shape {cqt_freqs.shape} does not match "
        f"librosa.cqt_frequencies output shape {expected.shape}."
    )
    assert np.allclose(cqt_freqs, expected, rtol=1e-5, atol=1e-6), (
        "cqt_freqs does not match "
        "librosa.cqt_frequencies(n_bins=cqt_n_bins, fmin=cqt_freqs[0], "
        "bins_per_octave=cqt_bins_per_octave) element-wise."
    )


def test_vqt_freqs_match_librosa_helper(meta):
    import librosa

    vqt_n_bins = _as_int(meta["vqt_n_bins"], "vqt_n_bins")
    vqt_bpo = _as_int(meta["vqt_bins_per_octave"], "vqt_bins_per_octave")
    vqt_freqs = np.asarray(meta["vqt_freqs"], dtype=float)
    fmin = float(vqt_freqs[0])
    assert fmin > 0, f"Recovered VQT fmin must be positive, got {fmin}."
    expected = librosa.interval_frequencies(
        n_bins=vqt_n_bins,
        fmin=fmin,
        intervals="equal",
        bins_per_octave=vqt_bpo,
    )
    assert vqt_freqs.shape == expected.shape, (
        f"vqt_freqs shape {vqt_freqs.shape} does not match "
        f"librosa.interval_frequencies output shape {expected.shape}."
    )
    assert np.allclose(vqt_freqs, expected, rtol=1e-5, atol=1e-6), (
        "vqt_freqs does not match "
        "librosa.interval_frequencies(n_bins=vqt_n_bins, fmin=vqt_freqs[0], "
        "intervals='equal', bins_per_octave=vqt_bins_per_octave) element-wise."
    )


def test_dynamic_range_above_threshold(stack):
    for key, arr in stack.items():
        span = float(arr.max()) - float(arr.min())
        assert span > 20.0, (
            f"Array '{key}' dynamic range must exceed 20 dB, got max-min={span:.3f}."
        )
