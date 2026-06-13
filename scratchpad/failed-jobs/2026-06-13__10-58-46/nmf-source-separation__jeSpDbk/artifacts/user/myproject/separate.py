"""
NMF-based Audio Source Separation

Decomposes a mono audio file into 4 NMF components and synthesizes
a time-domain waveform for each component using soft masking for
perfect-reconstruction property.
"""

import numpy as np
import soundfile as sf
import librosa
from sklearn.decomposition import NMF


def separate(input_path, output_dir, n_components=4, fft_size=2048, hop_length=512):
    # Load audio
    y, sr = sf.read(input_path)
    y = y.astype(np.float64)

    # Ensure mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Compute complex STFT
    D = librosa.stft(y, n_fft=fft_size, hop_length=hop_length)
    S = np.abs(D)  # magnitude spectrogram (non-negative)
    phase = np.angle(D)

    # Apply NMF via librosa.decompose.decompose
    components, activations = librosa.decompose.decompose(
        S,
        n_components=n_components,
        sort=True,
        transformer=NMF(
            n_components=n_components,
            init='nndsvda',
            max_iter=2000,
            tol=1e-6,
            random_state=42,
        ),
    )

    # components: (n_freq, n_components) — spectral basis vectors
    # activations: (n_components, n_frames) — time-varying gains

    # Reconstruct the NMF-approximated magnitude for each component
    component_mags = []
    for k in range(n_components):
        comp_mag = np.outer(components[:, k], activations[k])
        component_mags.append(comp_mag)

    # Sum of all component magnitudes (the NMF reconstruction)
    total_mag = sum(component_mags)

    # Use soft (Wiener-like) masking: distribute the original magnitude
    # according to each component's proportion. This ensures the sum of
    # the component STFTs equals the original STFT (perfect reconstruction).
    eps = np.finfo(np.float64).tiny

    for k in range(n_components):
        # Soft mask: component_mag / (sum of all component_mags + eps)
        mask = component_mags[k] / (total_mag + eps)

        # Apply mask to the original complex STFT
        component_stft = mask * D

        # Inverse STFT
        component_wave = librosa.istft(
            component_stft,
            hop_length=hop_length,
            length=len(y),
        )

        # Write output
        out_path = f"{output_dir}/component_{k}.wav"
        sf.write(out_path, component_wave, sr)
        print(f"Wrote {out_path}: {len(component_wave)} samples, "
              f"RMS={np.sqrt(np.mean(component_wave**2)):.6f}")

    # Verify approximate reconstruction
    y_recon = np.zeros(len(y))
    for k in range(n_components):
        out_path = f"{output_dir}/component_{k}.wav"
        comp, _ = sf.read(out_path)
        y_recon += comp

    cos_sim = np.dot(y, y_recon) / (np.linalg.norm(y) * np.linalg.norm(y_recon))
    print(f"\nReconstruction cosine similarity: {cos_sim:.6f}")

    # Compute spectral centroids for each component
    print("\nSpectral centroids (Hz):")
    centroids = []
    for k in range(n_components):
        out_path = f"{output_dir}/component_{k}.wav"
        comp, _ = sf.read(out_path)
        cent = librosa.feature.spectral_centroid(y=comp, sr=sr).mean()
        centroids.append(cent)
        print(f"  component_{k}: {cent:.1f} Hz")

    # Check if at least two differ by > 50 Hz
    diffs = []
    for i in range(n_components):
        for j in range(i + 1, n_components):
            diff = abs(centroids[i] - centroids[j])
            diffs.append(diff)
            print(f"  |c{i} - c{j}| = {diff:.1f} Hz")
    max_diff = max(diffs) if diffs else 0
    print(f"\nMax centroid difference: {max_diff:.1f} Hz "
          f"({'PASS' if max_diff > 50 else 'FAIL'})")


if __name__ == "__main__":
    separate(
        input_path="/workspace/input.wav",
        output_dir="/workspace",
        n_components=4,
    )
