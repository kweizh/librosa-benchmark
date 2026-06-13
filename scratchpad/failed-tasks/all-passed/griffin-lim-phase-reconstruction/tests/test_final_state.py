import os

import numpy as np
import pytest


WORKSPACE = "/workspace"
MAGNITUDE_PATH = os.path.join(WORKSPACE, "magnitude.npy")
RECONSTRUCTED_PATH = os.path.join(WORKSPACE, "reconstructed.wav")

# STFT parameters baked at task generation time (must match the Dockerfile).
N_FFT = 1024
HOP_LENGTH = 256
TARGET_SR = 22050

# Length of the reference signal that produced magnitude.npy. The Dockerfile
# writes this value to /workspace/reference_length.txt so that the test stays
# in sync with the actual generated signal.
REFERENCE_LENGTH_PATH = os.path.join(WORKSPACE, "reference_length.txt")


def _read_reference_length() -> int:
    assert os.path.isfile(REFERENCE_LENGTH_PATH), (
        f"Reference length file {REFERENCE_LENGTH_PATH} is missing; "
        "the Dockerfile must record the original signal length."
    )
    with open(REFERENCE_LENGTH_PATH) as f:
        return int(f.read().strip())


def test_reconstructed_file_exists():
    assert os.path.isfile(RECONSTRUCTED_PATH), (
        f"Expected output WAV file {RECONSTRUCTED_PATH} does not exist."
    )


def test_reconstructed_file_is_readable_and_mono_22050():
    import soundfile as sf

    audio, sr = sf.read(RECONSTRUCTED_PATH, always_2d=False)
    assert sr == TARGET_SR, (
        f"Expected sample rate {TARGET_SR} Hz, got {sr} Hz."
    )
    # Allow a (n,) mono array or a (n, 1) single-channel array; reject true multichannel.
    if audio.ndim == 2:
        assert audio.shape[1] == 1, (
            f"Expected mono audio, got {audio.shape[1]} channels."
        )
    else:
        assert audio.ndim == 1, (
            f"Expected 1D mono audio array, got shape {audio.shape}."
        )


def test_reconstructed_length_matches_reference():
    import soundfile as sf

    audio, _ = sf.read(RECONSTRUCTED_PATH, always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]

    original_length = _read_reference_length()
    tolerance = 4 * HOP_LENGTH
    diff = abs(int(audio.shape[0]) - original_length)
    assert diff <= tolerance, (
        f"Reconstructed length {audio.shape[0]} differs from original "
        f"length {original_length} by {diff} samples; allowed tolerance is "
        f"{tolerance} samples (= 4 * hop_length)."
    )


def test_reconstructed_peak_amplitude_in_audible_range():
    import soundfile as sf

    audio, _ = sf.read(RECONSTRUCTED_PATH, always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.float64)

    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    assert 0.01 <= peak <= 1.0, (
        f"Peak absolute amplitude {peak:.6f} is outside the allowed range "
        f"[0.01, 1.0]."
    )


def test_reconstructed_magnitude_matches_input():
    import librosa
    import soundfile as sf

    audio, sr = sf.read(RECONSTRUCTED_PATH, always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.float32)

    assert sr == TARGET_SR, (
        f"Expected sample rate {TARGET_SR} Hz for STFT verification, got {sr} Hz."
    )

    reference_magnitude = np.load(MAGNITUDE_PATH).astype(np.float32)

    recon_stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann",
        center=True,
    )
    recon_magnitude = np.abs(recon_stft).astype(np.float32)

    # Frame counts may differ by one or two frames depending on the
    # exact output length; align by truncating to the common range.
    assert recon_magnitude.shape[0] == reference_magnitude.shape[0], (
        f"Frequency-bin mismatch: reconstructed has {recon_magnitude.shape[0]} "
        f"bins but reference has {reference_magnitude.shape[0]}."
    )

    min_frames = min(recon_magnitude.shape[1], reference_magnitude.shape[1])
    frame_delta = abs(recon_magnitude.shape[1] - reference_magnitude.shape[1])
    assert frame_delta <= 4, (
        f"Reconstructed magnitude has {recon_magnitude.shape[1]} frames but "
        f"reference has {reference_magnitude.shape[1]}; difference "
        f"{frame_delta} exceeds tolerance of 4 frames."
    )

    ref = reference_magnitude[:, :min_frames]
    rec = recon_magnitude[:, :min_frames]

    # Frobenius relative error
    ref_norm = float(np.linalg.norm(ref))
    assert ref_norm > 0, "Reference magnitude has zero norm; invalid input."
    rel_err = float(np.linalg.norm(rec - ref) / ref_norm)
    assert rel_err < 0.20, (
        f"Frobenius relative error between reconstructed and reference "
        f"magnitudes is {rel_err:.4f}, expected < 0.20."
    )

    # Cosine similarity on flattened magnitudes
    ref_flat = ref.reshape(-1).astype(np.float64)
    rec_flat = rec.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(rec_flat))
    assert denom > 0, "Cannot compute cosine similarity with zero-norm magnitude."
    cosine = float(np.dot(ref_flat, rec_flat) / denom)
    assert cosine >= 0.95, (
        f"Cosine similarity between reconstructed and reference magnitude "
        f"is {cosine:.4f}, expected >= 0.95."
    )
