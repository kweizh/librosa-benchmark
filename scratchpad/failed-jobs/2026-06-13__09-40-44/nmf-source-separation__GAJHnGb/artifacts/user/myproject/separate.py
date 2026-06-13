import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import NMF

def separate_audio():
    input_path = '/workspace/input.wav'
    output_dir = '/workspace'
    
    # 1. Read the input audio file (mono, native sample rate)
    print(f"Reading input audio from {input_path}...")
    y, sr = librosa.load(input_path, sr=None, mono=True)
    input_length = len(y)
    print(f"Loaded audio with sample rate {sr} Hz, length {input_length} samples.")
    
    # 2. Compute complex STFT
    stft_complex = librosa.stft(y)
    S = np.abs(stft_complex)
    phase = np.angle(stft_complex)
    
    # 3. Decompose the magnitude spectrogram into exactly 4 NMF components
    print("Performing NMF decomposition...")
    # Use sort=True to sort components by ascending peak frequency
    comps, acts = librosa.decompose.decompose(S, n_components=4, sort=True, random_state=42, max_iter=1000)
    
    # Reconstructed magnitude spectrogram from NMF
    S_approx = comps.dot(acts)
    
    # 4. Reconstruct each of the 4 components using soft masking (Wiener filtering)
    print("Reconstructing component waveforms...")
    y_components = []
    for i in range(4):
        # Contribution of component i
        S_i = comps[:, [i]].dot(acts[[i], :])
        
        # Soft mask to distribute the original magnitude spectrogram
        mask_i = S_i / (S_approx + 1e-10)
        S_i_masked = S * mask_i
        
        # Recombine magnitude with original phase
        stft_i = S_i_masked * np.exp(1j * phase)
        
        # Inverse STFT to get time-domain waveform
        y_i = librosa.istft(stft_i, length=input_length)
        y_components.append(y_i)
        
        # Write to file
        output_path = os.path.join(output_dir, f"component_{i}.wav")
        sf.write(output_path, y_i, sr)
        print(f"Saved component {i} to {output_path}")

    # 5. Verification
    print("\n--- Verification Results ---")
    
    # Verify file existence and loading
    all_exist = True
    for i in range(4):
        output_path = os.path.join(output_dir, f"component_{i}.wav")
        if not os.path.exists(output_path):
            print(f"Error: {output_path} does not exist!")
            all_exist = False
            continue
        
        data, file_sr = sf.read(output_path)
        length_diff = abs(len(data) - input_length)
        rms = np.sqrt(np.mean(data**2))
        centroid = librosa.feature.spectral_centroid(y=data, sr=file_sr).mean()
        
        print(f"Component {i}:")
        print(f"  Loaded successfully: Yes")
        print(f"  Sample rate: {file_sr} Hz (Expected: {sr} Hz)")
        print(f"  Length: {len(data)} samples (Diff from input: {length_diff} samples)")
        print(f"  RMS amplitude: {rms:.6f} (Expected: > 1e-5)")
        print(f"  Spectral centroid: {centroid:.2f} Hz")
        
        assert file_sr == sr, f"Sample rate mismatch for component {i}"
        assert length_diff <= 2048, f"Length difference too large for component {i}"
        assert rms > 1e-5, f"Component {i} is trivial (RMS too low)"
        
    # Verify sum and cosine similarity
    y_sum = np.sum(y_components, axis=0)
    cos_sim = np.dot(y, y_sum) / (np.linalg.norm(y) * np.linalg.norm(y_sum))
    print(f"\nCosine similarity between sum and input: {cos_sim:.6f} (Expected: >= 0.95)")
    assert cos_sim >= 0.95, f"Cosine similarity is too low: {cos_sim:.6f}"
    
    # Verify spectral centroid difference
    centroids = [librosa.feature.spectral_centroid(y=comp, sr=sr).mean() for comp in y_components]
    max_diff = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            diff = abs(centroids[i] - centroids[j])
            if diff > max_diff:
                max_diff = diff
    print(f"Max spectral centroid difference between components: {max_diff:.2f} Hz (Expected: > 50 Hz)")
    assert max_diff > 50.0, f"Spectral centroids do not differ enough: max diff is {max_diff:.2f} Hz"
    
    print("\nAll acceptance criteria successfully verified!")

if __name__ == '__main__':
    separate_audio()
