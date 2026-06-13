"""
NMF-based Audio Source Separation
Decomposes an audio file into 4 components using Non-negative Matrix Factorization
and reconstructs each component as a separate time-domain waveform.

Reconstruction strategy:
  Wiener soft-masking — each component's mask is  S_k / (S_total + eps).
  Applying that mask to the *complex* STFT and inverting gives waveforms
  that add up to (virtually) the original signal, satisfying the cosine-
  similarity criterion.
"""

import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import NMF

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_PATH   = "/workspace/input.wav"
OUTPUT_PATHS = [f"/workspace/component_{i}.wav" for i in range(4)]
N_COMPONENTS = 4
N_FFT        = 2048
HOP_LENGTH   = 512

# ── 1. Load audio ────────────────────────────────────────────────────────────
print(f"Loading audio from {INPUT_PATH} ...")
y, sr = librosa.load(INPUT_PATH, sr=None, mono=True)
print(f"  Sample rate : {sr} Hz")
print(f"  Length      : {len(y)} samples  ({len(y)/sr:.2f} s)")

# ── 2. Compute STFT ──────────────────────────────────────────────────────────
D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)   # complex spectrogram
S = np.abs(D)                                               # magnitude spectrogram
print(f"  STFT shape  : {D.shape}  (freq_bins × frames)")

# ── 3. NMF decomposition ─────────────────────────────────────────────────────
# librosa.decompose.decompose returns:
#   W  shape (freq_bins, n_components)  – spectral basis vectors
#   H  shape (n_components, frames)     – temporal activations
print(f"Running NMF with {N_COMPONENTS} components ...")
W, H = librosa.decompose.decompose(
    S,
    n_components=N_COMPONENTS,
    transformer=NMF(
        n_components=N_COMPONENTS,
        init="nndsvda",
        max_iter=500,
        random_state=42,
    ),
    sort=True,   # sort components by explained variance
)
print(f"  W shape : {W.shape}")
print(f"  H shape : {H.shape}")

# ── 4. Wiener soft-mask reconstruction ──────────────────────────────────────
# NMF approximation of the magnitude spectrogram
S_recon = W @ H                    # shape: (freq_bins, frames)

eps = 1e-8

print("Reconstructing components via Wiener masks ...")
for k in range(N_COMPONENTS):
    # Contribution of component k to the magnitude spectrogram
    S_k = np.outer(W[:, k], H[k, :])          # (freq_bins, frames)

    # Soft Wiener mask: fraction of energy attributed to this component
    mask = S_k / (S_recon + eps)               # values in [0, 1]

    # Apply mask to the *complex* STFT → preserves original phase information
    D_k = mask * D                             # complex masked spectrogram

    # Inverse STFT → time-domain waveform
    y_k = librosa.istft(D_k, hop_length=HOP_LENGTH, length=len(y))

    rms = np.sqrt(np.mean(y_k ** 2))
    sf.write(OUTPUT_PATHS[k], y_k, sr, subtype="PCM_32")
    print(f"  component_{k}.wav  length={len(y_k)}  RMS={rms:.5f}")

print("\nDone. Output files:")
for p in OUTPUT_PATHS:
    print(f"  {p}")
