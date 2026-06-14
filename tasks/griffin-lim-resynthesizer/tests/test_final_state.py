import json
import math
import os

import numpy as np
import pytest


INPUT_WAV = "/workspace/input.wav"
RECONSTRUCTED_WAV = "/workspace/reconstructed.wav"
METRICS_JSON = "/workspace/metrics.json"

REQUIRED_METRIC_KEYS = {
    "spectral_convergence",
    "snr_db",
    "length_samples",
    "sample_rate",
    "n_mels",
    "n_iter",
}


def _as_int(value):
    """Accept Python ints and any integer-valued numerics; reject booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value):
    """Accept Python ints and floats; reject booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@pytest.fixture(scope="module")
def reconstructed_audio():
    import librosa

    assert os.path.isfile(RECONSTRUCTED_WAV), (
        f"Expected reconstructed audio file {RECONSTRUCTED_WAV} to exist after the task completes."
    )
    try:
        y, sr = librosa.load(RECONSTRUCTED_WAV, sr=None, mono=False)
    except Exception as exc:  # pragma: no cover - failure path
        raise AssertionError(
            f"Failed to load {RECONSTRUCTED_WAV} with librosa.load: {exc}"
        )
    arr = np.asarray(y)
    # Coerce to 1-D mono.
    if arr.ndim == 2:
        assert arr.shape[0] == 1, (
            f"Reconstructed audio must be mono, got shape {arr.shape}."
        )
        arr = arr[0]
    assert arr.ndim == 1, (
        f"Reconstructed audio must be a 1-D mono waveform, got ndim={arr.ndim}."
    )
    assert arr.size > 0, (
        f"Reconstructed waveform from {RECONSTRUCTED_WAV} is empty."
    )
    return arr.astype(np.float64), int(sr)


@pytest.fixture(scope="module")
def input_audio():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    return np.asarray(y, dtype=np.float64), int(sr)


@pytest.fixture(scope="module")
def metrics():
    assert os.path.isfile(METRICS_JSON), (
        f"Expected metrics file {METRICS_JSON} to exist after the task completes."
    )
    with open(METRICS_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{METRICS_JSON} is not valid JSON: {exc}")
    assert isinstance(data, dict), (
        f"{METRICS_JSON} must contain a JSON object, got: {type(data).__name__}."
    )
    return data


def test_metrics_has_exact_required_keys(metrics):
    keys = set(metrics.keys())
    missing = REQUIRED_METRIC_KEYS - keys
    extra = keys - REQUIRED_METRIC_KEYS
    assert not missing, f"metrics.json is missing required keys: {sorted(missing)}."
    assert not extra, (
        f"metrics.json contains unexpected keys: {sorted(extra)}. "
        f"Expected exactly: {sorted(REQUIRED_METRIC_KEYS)}."
    )


def test_metrics_types_and_ranges(metrics):
    n_mels = _as_int(metrics["n_mels"])
    n_iter = _as_int(metrics["n_iter"])
    length_samples = _as_int(metrics["length_samples"])
    sample_rate = _as_int(metrics["sample_rate"])

    assert n_mels is not None, (
        f"metrics['n_mels'] must be an integer, got: {metrics['n_mels']!r}."
    )
    assert n_iter is not None, (
        f"metrics['n_iter'] must be an integer, got: {metrics['n_iter']!r}."
    )
    assert length_samples is not None, (
        f"metrics['length_samples'] must be an integer, got: {metrics['length_samples']!r}."
    )
    assert sample_rate is not None, (
        f"metrics['sample_rate'] must be an integer, got: {metrics['sample_rate']!r}."
    )

    assert n_mels >= 128, f"metrics['n_mels'] must be >= 128, got {n_mels}."
    assert n_iter >= 32, f"metrics['n_iter'] must be >= 32, got {n_iter}."
    assert length_samples > 0, (
        f"metrics['length_samples'] must be positive, got {length_samples}."
    )
    assert sample_rate > 0, (
        f"metrics['sample_rate'] must be positive, got {sample_rate}."
    )

    sc = _as_float(metrics["spectral_convergence"])
    snr = _as_float(metrics["snr_db"])
    assert sc is not None and math.isfinite(sc), (
        f"metrics['spectral_convergence'] must be a finite float, got {metrics['spectral_convergence']!r}."
    )
    assert snr is not None and math.isfinite(snr), (
        f"metrics['snr_db'] must be a finite float, got {metrics['snr_db']!r}."
    )


def test_reconstructed_sample_rate_matches_input(reconstructed_audio, input_audio):
    _, recon_sr = reconstructed_audio
    _, in_sr = input_audio
    assert recon_sr == in_sr, (
        f"Reconstructed sample rate ({recon_sr}) must equal input sample rate ({in_sr})."
    )


def test_metrics_sample_rate_matches_reconstructed(metrics, reconstructed_audio):
    recon_y, recon_sr = reconstructed_audio
    declared_sr = _as_int(metrics["sample_rate"])
    assert declared_sr == recon_sr, (
        f"metrics['sample_rate']={declared_sr} must equal the reconstructed WAV sample rate {recon_sr}."
    )


def test_metrics_length_matches_reconstructed(metrics, reconstructed_audio):
    recon_y, _ = reconstructed_audio
    declared_len = _as_int(metrics["length_samples"])
    assert declared_len == int(recon_y.shape[-1]), (
        f"metrics['length_samples']={declared_len} must equal actual reconstructed sample count "
        f"{int(recon_y.shape[-1])}."
    )


def test_reconstructed_length_within_two_percent_of_input(reconstructed_audio, input_audio):
    recon_y, _ = reconstructed_audio
    in_y, _ = input_audio
    in_len = int(in_y.shape[-1])
    recon_len = int(recon_y.shape[-1])
    assert in_len > 0, "Input waveform length must be positive."
    rel_diff = abs(recon_len - in_len) / in_len
    assert rel_diff <= 0.02, (
        f"Reconstructed length {recon_len} differs from input length {in_len} by "
        f"{rel_diff:.4f} (> 2%)."
    )


def _aligned_pair(a: np.ndarray, b: np.ndarray):
    n = min(int(a.shape[-1]), int(b.shape[-1]))
    return a[:n].astype(np.float64), b[:n].astype(np.float64)


def _compute_spectral_convergence_and_snr(input_y: np.ndarray, recon_y: np.ndarray):
    import librosa

    ref, rec = _aligned_pair(input_y, recon_y)

    n_fft = 2048
    hop_length = 512
    S_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop_length))
    S_rec = np.abs(librosa.stft(rec, n_fft=n_fft, hop_length=hop_length))

    num = np.linalg.norm(S_ref - S_rec)
    den = np.linalg.norm(S_ref)
    sc = float(num / den) if den > 0 else float("inf")

    noise = ref - rec
    sig_power = float(np.sum(ref ** 2))
    noise_power = float(np.sum(noise ** 2))
    if noise_power <= 0 or sig_power <= 0:
        snr = float("inf") if noise_power == 0 and sig_power > 0 else float("-inf")
    else:
        snr = 10.0 * math.log10(sig_power / noise_power)
    return sc, snr


def test_recomputed_metrics_pass_thresholds(input_audio, reconstructed_audio):
    in_y, _ = input_audio
    recon_y, _ = reconstructed_audio
    sc, snr = _compute_spectral_convergence_and_snr(in_y, recon_y)
    assert math.isfinite(sc), f"Recomputed spectral_convergence is not finite: {sc}."
    assert math.isfinite(snr), f"Recomputed snr_db is not finite: {snr}."
    assert sc < 0.5, (
        f"Recomputed spectral_convergence={sc:.4f} must be < 0.5 "
        f"(reconstruction is too far from input)."
    )
    assert snr > 0.0, (
        f"Recomputed snr_db={snr:.4f} must be > 0.0 dB "
        f"(reconstruction is no better than noise)."
    )


def test_reported_metrics_match_recomputation(metrics, input_audio, reconstructed_audio):
    in_y, _ = input_audio
    recon_y, _ = reconstructed_audio
    sc_recomp, snr_recomp = _compute_spectral_convergence_and_snr(in_y, recon_y)

    reported_sc = float(metrics["spectral_convergence"])
    reported_snr = float(metrics["snr_db"])

    assert abs(reported_sc - sc_recomp) <= 0.1, (
        f"Reported spectral_convergence={reported_sc:.4f} differs from recomputed "
        f"{sc_recomp:.4f} by more than 0.1; metrics appear fabricated or inconsistent."
    )
    assert abs(reported_snr - snr_recomp) <= 3.0, (
        f"Reported snr_db={reported_snr:.4f} differs from recomputed "
        f"{snr_recomp:.4f} by more than 3 dB; metrics appear fabricated or inconsistent."
    )
