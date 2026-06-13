import librosa
import numpy as np
import soundfile as sf
import sklearn.decomposition

def main():
    input_file = '/workspace/input.wav'
    y, sr = librosa.load(input_file, sr=None)
    
    # Compute complex STFT
    S = librosa.stft(y)
    
    # Magnitude and phase
    S_mag = np.abs(S)
    S_phase = np.angle(S)
    
    # Decompose
    n_components = 4
    transformer = sklearn.decomposition.NMF(n_components=n_components, random_state=42)
    components, activations = librosa.decompose.decompose(S_mag, transformer=transformer, sort=True)
    
    # Calculate sum of component magnitudes to create soft masks
    comp_mags = []
    for i in range(n_components):
        comp_mags.append(np.outer(components[:, i], activations[i, :]))
    
    comp_mags = np.array(comp_mags)
    sum_mags = np.sum(comp_mags, axis=0) + 1e-10
    
    for i in range(n_components):
        # Soft mask
        mask = comp_mags[i] / sum_mags
        comp_mag_masked = mask * S_mag
        
        # Combine with original phase
        comp_S = comp_mag_masked * np.exp(1j * S_phase)
        
        # Inverse STFT
        comp_y = librosa.istft(comp_S, length=len(y))
        
        # Save to file
        output_file = f'/workspace/component_{i}.wav'
        sf.write(output_file, comp_y, sr)

if __name__ == '__main__':
    main()
