#!/usr/bin/env python3
"""NMF-based audio source separation."""

import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import NMF

# ── 1. Load input audio ──────────────────────────────────────────────
y, sr = librosa.load('/workspace/input.wav', sr=None, mono=True)
original_length = len(y)

# ── 2. Compute STFT ──────────────────────────────────────────────────
n_fft = 2048
S = librosa.stft(y, n_fft=n_fft)
mag = np.abs(S)
phase = np.angle(S)

# ── 3. NMF decomposition ─────────────────────────────────────────────
n_components = 4
nmf_model = NMF(
    n_components=n_components,
    init='nndsvda',
    random_state=42,
    max_iter=1000,
)
W, H = librosa.decompose.decompose(mag, transformer=nmf_model)
# W: (freqs, n_components) – spectral basis vectors
# H: (n_components, frames) – time-varying activations

# ── 4. Reconstruct each component ────────────────────────────────────
# Compute individual component magnitudes
component_mags = []
for i in range(n_components):
    component_mags.append(np.outer(W[:, i], H[i]))

# Full NMF reconstruction
recon_mag = sum(component_mags)

# Scale each component proportionally so that their sum equals the
# original magnitude spectrogram.  This guarantees perfect reconstruction
# while preserving each component's relative contribution.
epsilon = 1e-10
scale = mag / (recon_mag + epsilon)

components = []
for i in range(n_components):
    scaled_mag = component_mags[i] * scale
    # Recombine with the original phase
    component_complex = scaled_mag * np.exp(1j * phase)
    # Inverse STFT to get time-domain signal
    component_audio = librosa.istft(component_complex, length=original_length)
    components.append(component_audio)

# ── 5. Write output WAV files ─────────────────────────────────────────
for i, comp in enumerate(components):
    out_path = f'/workspace/component_{i}.wav'
    sf.write(out_path, comp, sr)
    print(f"Wrote {out_path}: shape={comp.shape}, sr={sr}, "
          f"rms={np.sqrt(np.mean(comp**2)):.6f}")

# ── 6. Verification ───────────────────────────────────────────────────
recon = np.sum(np.array(components), axis=0)
if len(recon) < original_length:
    recon = np.pad(recon, (0, original_length - len(recon)))
else:
    recon = recon[:original_length]

cos_sim = np.dot(y, recon) / (np.linalg.norm(y) * np.linalg.norm(recon) + 1e-12)
print(f"\nCosine similarity (sum vs original): {cos_sim:.4f}")

# Length check
for i, comp in enumerate(components):
    diff = abs(len(comp) - original_length)
    print(f"Component {i}: length={len(comp)}, original={original_length}, "
          f"diff={diff} (<2048: {diff < 2048})")

# Spectral centroid check
print("\nSpectral centroids of components:")
centroids = []
for i, comp in enumerate(components):
    sc = librosa.feature.spectral_centroid(y=comp, sr=sr)
    mean_sc = np.mean(sc)
    centroids.append(mean_sc)
    print(f"  Component {i}: {mean_sc:.2f} Hz")

from itertools import combinations
found = False
for a, b in combinations(centroids, 2):
    if abs(a - b) > 50:
        found = True
        break
print(f"At least two spectral centroids differ by >50 Hz: {found}")

# RMS check
for i, comp in enumerate(components):
    rms = np.sqrt(np.mean(comp**2))
    print(f"Component {i} RMS: {rms:.6f} (>1e-5: {rms > 1e-5})")

# Sample rate check
info = sf.info('/workspace/input.wav')
print(f"\nInput sample rate: {info.samplerate}")
for i in range(n_components):
    info_i = sf.info(f'/workspace/component_{i}.wav')
    print(f"Component {i} sample rate: {info_i.samplerate}")