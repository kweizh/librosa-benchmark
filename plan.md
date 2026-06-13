# Librosa 0.11.0 Deep Research & Benchmark Specification

This document provides a highly structured technical reference and research specification for **librosa** (specifically pinning the latest stable release, **v0.11.0**). It is designed to serve as a foundation for generating high-quality evaluation datasets, coding agent benchmarks, and robust integration tests.

---

## 1. Library Overview

### Description
**librosa** is the industry-standard Python library for audio and music analysis. It provides the fundamental building blocks necessary to build Music Information Retrieval (MIR) systems, speech processing pipelines, and general audio feature extraction workflows. Rather than focusing on real-time synthesis or low-latency playback, librosa is designed for offline, high-fidelity analysis, leveraging NumPy, SciPy, and Scikit-learn to deliver efficient, vector-optimized DSP and feature representations.

### Ecosystem Role
Within the Python scientific computing and machine learning stack, librosa acts as the bridge between raw audio data and analytical models:
* **Audio I/O**: It relies on `soundfile` (for robust WAV, FLAC, and OGG reading/writing) and `audioread` (as a fallback for MP3s and other compressed formats).
* **DSP & Linear Algebra**: It utilizes `numpy` for multi-dimensional array operations and `scipy` for core digital signal processing (e.g., filters, windowing, and FFTs).
* **Machine Learning**: It integrates seamlessly with `scikit-learn` for matrix decomposition (such as Non-negative Matrix Factorization) and clustering (such as Agglomerative Clustering for temporal segmentation).
* **Visualization**: It extends `matplotlib` via its `librosa.display` module, providing specialized, publication-quality plotting functions for waveforms, spectrograms, and chromagrams.

### Project Setup (Non-Interactive)
For execution in autonomous, non-interactive environments (such as headless Docker containers), librosa must be installed alongside its optional high-quality resampler backend (`soxr`) and without interactive graphics dependencies.

#### Installation
Run the following non-interactive pip command to install librosa v0.11.0 with high-quality resampling and audio I/O support:
```bash
pip install --no-input librosa==0.11.0 soundfile numpy scipy scikit-learn "soxr>=0.3.2" pooch
```

#### Headless Environment Configuration
In non-interactive environments, calling `matplotlib` functions can trigger display errors (e.g., `TclError: no display name and no $DISPLAY environment variable`). To prevent this, configure matplotlib to use the headless `Agg` backend before importing any visualization modules:
```python
import os
import matplotlib
matplotlib.use('Agg')  # Enforce headless rendering backend
import matplotlib.pyplot as plt
import librosa
import librosa.display
```

---

## 2. Core Primitives & APIs

Librosa represents audio data using standard multi-dimensional NumPy arrays (`np.ndarray`). 
* **Waveforms (`y`)** are represented as floating-point arrays of shape `(n_samples,)` for mono, or `(n_channels, n_samples)` for multi-channel (channels-first).
* **Spectrograms (`S` or `D`)** are represented as complex or real arrays of shape `(..., n_bins, n_frames)`. Leading dimensions represent channels.
* **Feature Matrices** (e.g., MFCCs, Chromagrams) are represented as real arrays of shape `(..., n_features, n_frames)`.

### Submodule Directory & API Links

| Submodule | Core Purpose | Key API Documentation Links |
| :--- | :--- | :--- |
| **Core IO & DSP** | Audio loading, resampling, STFT/iSTFT, and decibel scaling | [librosa.load](https://librosa.org/doc/0.11.0/generated/librosa.load.html)<br>[librosa.stft](https://librosa.org/doc/0.11.0/generated/librosa.stft.html)<br>[librosa.istft](https://librosa.org/doc/0.11.0/generated/librosa.istft.html)<br>[librosa.amplitude_to_db](https://librosa.org/doc/0.11.0/generated/librosa.amplitude_to_db.html)<br>[librosa.resample](https://librosa.org/doc/0.11.0/generated/librosa.resample.html) |
| **Feature Extraction** | Spectrograms, MFCCs, Chromagrams, and spectral descriptors | [librosa.feature.melspectrogram](https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html)<br>[librosa.feature.mfcc](https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html)<br>[librosa.feature.chroma_stft](https://librosa.org/doc/0.11.0/generated/librosa.feature.chroma_stft.html)<br>[librosa.feature.delta](https://librosa.org/doc/0.11.0/generated/librosa.feature.delta.html) |
| **Onset & Beat Tracking** | Detecting note onsets, estimating tempo, and tracking beats | [librosa.onset.onset_detect](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_detect.html)<br>[librosa.beat.beat_track](https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html) |
| **Effects & Processing** | Time-domain modifications, HPSS, pitch shifting, and time stretching | [librosa.effects.hpss](https://librosa.org/doc/0.11.0/generated/librosa.effects.hpss.html)<br>[librosa.effects.pitch_shift](https://librosa.org/doc/0.11.0/generated/librosa.effects.pitch_shift.html)<br>[librosa.effects.time_stretch](https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html) |
| **Structural Segmentation** | Recurrence matrices, cross-similarity, and temporal boundaries | [librosa.segment.recurrence_matrix](https://librosa.org/doc/0.11.0/generated/librosa.segment.recurrence_matrix.html)<br>[librosa.segment.subsegment](https://librosa.org/doc/0.11.0/generated/librosa.segment.subsegment.html) |
| **Sequential Modeling** | State-sequence decoding, transition matrices, and DTW alignment | [librosa.sequence.viterbi](https://librosa.org/doc/0.11.0/generated/librosa.sequence.viterbi.html)<br>[librosa.sequence.dtw](https://librosa.org/doc/0.11.0/generated/librosa.sequence.dtw.html) |
| **Matrix Decomposition** | Factorization (NMF, PCA) and nearest-neighbor filtering | [librosa.decompose.decompose](https://librosa.org/doc/0.11.0/generated/librosa.decompose.decompose.html) |
| **Visualization** | Plotting waveforms, spectrograms, and features | [librosa.display.waveshow](https://librosa.org/doc/0.11.0/generated/librosa.display.waveshow.html)<br>[librosa.display.specshow](https://librosa.org/doc/0.11.0/generated/librosa.display.specshow.html) |

---

### Detailed API Usage & Code Examples

*Note: In librosa 0.11.0, keyword-only arguments are strictly enforced across almost all function signatures. Positional arguments are restricted to primary data arrays (e.g., `y`).*

#### A. Core I/O, Resampling, and Spectral Analysis (STFT / iSTFT)
This example demonstrates loading audio, high-quality resampling, forward Short-Time Fourier Transform (STFT), decibel scaling, and Inverse Short-Time Fourier Transform (iSTFT) reconstruction.

```python
import numpy as np
import librosa

# 1. Load audio (Mono vs Stereo, Native vs Custom Sample Rate)
# By default, librosa.load resamples to 22050 Hz. Use sr=None to preserve native rate.
audio_path = librosa.ex('trumpet')  # Downloads/retrieves cached example WAV file

# Load as native stereo (if stereo) or native mono
y, sr = librosa.load(audio_path, sr=None, mono=False)
print(f"Loaded audio shape: {y.shape}, Native sample rate: {sr} Hz")
# Mono shape: (n_samples,)
# Stereo shape: (2, n_samples)

# 2. Resample waveform explicitly to a target rate (e.g., 16000 Hz for speech models)
# 'soxr_hq' is the default high-quality resampler if soxr is installed
y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type='soxr_hq')
print(f"Resampled audio shape: {y_resampled.shape} at 16000 Hz")

# 3. Short-Time Fourier Transform (STFT)
# Multi-channel arrays are supported natively.
n_fft = 2048
hop_length = 512
win_length = 2048

# Compute complex STFT matrix D
# D.shape is (..., 1 + n_fft // 2, n_frames)
# For mono: (1025, n_frames). For stereo: (2, 1025, n_frames)
D = librosa.stft(
    y_resampled,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window='hann',
    center=True,
    pad_mode='constant'
)
print(f"STFT Matrix shape: {D.shape}, Data type: {D.dtype}")

# 4. Decibel Scaling (Log-amplitude representation)
# Convert magnitude spectrogram to decibels. ref=np.max scales relative to peak energy.
magnitude = np.abs(D)
S_db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0)
print(f"DB Spectrogram shape: {S_db.shape}, Min DB: {S_db.min():.2f}, Max DB: {S_db.max():.2f}")

# 5. Inverse Short-Time Fourier Transform (iSTFT)
# Reconstruct the time-domain waveform from the complex STFT matrix.
# length parameter ensures the output matches the exact sample length of the input.
y_reconstructed = librosa.istft(
    D,
    hop_length=hop_length,
    win_length=win_length,
    window='hann',
    center=True,
    length=y_resampled.shape[-1]
)
print(f"Reconstructed waveform shape: {y_reconstructed.shape}")
```

#### B. Audio Feature Extraction (`librosa.feature`)
This example extracts Mel Spectrograms, Mel-Frequency Cepstral Coefficients (MFCCs), Chromagrams, and Delta features.

```python
import numpy as np
import librosa

# Load standard mono reference audio at 22050 Hz
y, sr = librosa.load(librosa.ex('nutcracker'), sr=22050, mono=True)

# 1. Mel Spectrogram
# Computes a power spectrogram, maps it to the Mel scale, and integrates it.
# Output shape: (n_mels, n_frames) -> e.g., (128, n_frames)
mel_spec = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=0.0,
    fmax=sr / 2.0,
    power=2.0  # 2.0 for power spectrogram, 1.0 for magnitude
)
# Convert to log-power scale (decibels)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
print(f"Mel Spectrogram shape: {mel_spec_db.shape}")

# 2. Mel-Frequency Cepstral Coefficients (MFCCs)
# Computes MFCCs from a log-power Mel spectrogram.
# Output shape: (n_mfcc, n_frames) -> e.g., (20, n_frames)
mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=20,
    dct_type=2,
    norm='ortho',
    lifter=0
)
print(f"MFCC shape: {mfcc.shape}")

# 3. Delta Features (First-order derivatives of features)
# Computes local estimates of the derivative of the input feature matrix.
# Output shape matches the input shape: (n_mfcc, n_frames)
mfcc_delta = librosa.feature.delta(mfcc, order=1, axis=-1)
print(f"MFCC Delta shape: {mfcc_delta.shape}")

# 4. Chromagram (Chroma Short-Time Fourier Transform)
# Projects the spectral energy of frames onto the 12 semitone pitch classes.
# Output shape: (n_chroma, n_frames) -> (12, n_frames)
chroma = librosa.feature.chroma_stft(
    y=y,
    sr=sr,
    n_chroma=12,
    n_fft=2048,
    hop_length=512,
    tuning=0.0
)
print(f"Chroma STFT shape: {chroma.shape}")
```

#### C. Onset and Beat Tracking (`librosa.onset`, `librosa.beat`)
Detects onset events (note starts), estimates global tempo (BPM), and locates beat frames.

```python
import numpy as np
import librosa

# Load mono reference audio
y, sr = librosa.load(librosa.ex('choice'), sr=22050)

# 1. Onset Strength Envelope
# Compute spectral flux onset strength envelope over time.
# Output shape: (n_frames,)
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
print(f"Onset strength envelope shape: {onset_env.shape}")

# 2. Onset Peak Detection
# Detect note onset event frame indices by picking peaks in the envelope.
onset_frames = librosa.onset.onset_detect(
    onset_envelope=onset_env,
    sr=sr,
    hop_length=512,
    backtrack=True  # Backtrack to the local minimum before peak
)
print(f"Detected {len(onset_frames)} onset events. First 5 frames: {onset_frames[:5]}")

# 3. Global Tempo & Beat Tracking
# Estimates global tempo (BPM) and locates beat frame indices.
# Note: beat_track returns (tempo, beat_frames)
tempo, beat_frames = librosa.beat.beat_track(
    y=y,
    sr=sr,
    onset_envelope=onset_env,
    hop_length=512,
    start_bpm=120.0
)
print(f"Estimated Tempo: {tempo:.2f} BPM")
print(f"Detected {len(beat_frames)} beat frames. First 5 frames: {beat_frames[:5]}")

# Convert beat frames to seconds (timestamps)
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
print(f"First 5 beat timestamps (seconds): {beat_times[:5]}")
```

#### D. Effects, HPSS, and Waveform Manipulation (`librosa.effects`)
Performs time-domain digital effects, Harmonic-Percussive Source Separation (HPSS), pitch shifting, and time stretching.

```python
import librosa

y, sr = librosa.load(librosa.ex('choice'), sr=22050)

# 1. Harmonic-Percussive Source Separation (HPSS) in Time Domain
# Automates the STFT -> HPSS -> iSTFT pipeline.
# Returns harmonic waveform (tonal elements) and percussive waveform (transient elements).
# Output shapes match the input: (n_samples,)
y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 2.0))
print(f"HPSS - Harmonic shape: {y_harmonic.shape}, Percussive shape: {y_percussive.shape}")

# 2. Pitch Shifting
# Shifts the pitch of the waveform by n_steps semitones (bins_per_octave=12).
# scale=True scales the resampled signal so that total energy remains constant.
y_shifted = librosa.effects.pitch_shift(
    y,
    sr=sr,
    n_steps=4.0,  # Shift up by 4 semitones (major third)
    bins_per_octave=12,
    scale=True
)
print(f"Pitch Shifted shape: {y_shifted.shape}")

# 3. Time Stretching
# Speeds up or slows down the waveform by a factor of 'rate' without changing pitch.
# Output shape: (round(n_samples / rate),)
y_stretched = librosa.effects.time_stretch(
    y,
    rate=1.5  # Speed up by 1.5x
)
print(f"Time Stretched shape: {y_stretched.shape}")

# 4. Trimming Silence
# Trims leading and trailing silence from an audio signal.
# Returns trimmed waveform and the start/end sample indices.
y_trimmed, index = librosa.effects.trim(y, top_db=60)
print(f"Trimmed shape: {y_trimmed.shape}, Active interval (samples): {index}")
```

#### E. Structural Segmentation (`librosa.segment`)
Computes self-similarity recurrence matrices and performs temporal clustering.

```python
import numpy as np
import librosa

y, sr = librosa.load(librosa.ex('nutcracker'), sr=22050)

# 1. Extract feature sequence (e.g., Chromagram)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

# 2. Time-Delay Embedding (stacking history to capture temporal structure)
# Shapes: (n_features * n_steps, n_frames)
chroma_stack = librosa.feature.stack_memory(chroma, n_steps=5, delay=2)

# 3. Compute Recurrence Matrix (Self-Similarity)
# R[i, j] represents similarity of frame i and frame j.
# mode='connectivity' returns binary matrix; mode='affinity' returns real-valued distances.
R = librosa.segment.recurrence_matrix(
    chroma_stack,
    k=5,               # Keep 5 nearest neighbors
    width=3,           # Suppress diagonal self-loops within +-3 frames
    metric='cosine',
    mode='connectivity'
)
print(f"Recurrence Matrix shape: {R.shape}, Data type: {R.dtype}")

# 4. Convert Recurrence Matrix to Lag Matrix
# Shifts columns to align diagonal structures horizontally (time-lag representation).
lag_matrix = librosa.segment.recurrence_to_lag(R, pad=True)
print(f"Lag Matrix shape: {lag_matrix.shape}")
```

#### F. Sequential Modeling & DTW (`librosa.sequence`)
Performs Dynamic Time Warping (DTW) to align two feature sequences.

```python
import numpy as np
import librosa

# Load two different segments of the same underlying music to align them
y_ref, sr = librosa.load(librosa.ex('pistachio'), sr=22050, duration=10)
y_comp, sr = librosa.load(librosa.ex('pistachio'), sr=22050, offset=5, duration=10)

# Compute normalized chroma features (CENS is highly robust for alignment)
chroma_ref = librosa.feature.chroma_cens(y=y_ref, sr=sr)
chroma_comp = librosa.feature.chroma_cens(y=y_comp, sr=sr)

# Compute Dynamic Time Warping (DTW) alignment path
# D is the accumulated cost matrix (N, M). wp is the warping path (L, 2).
D, wp = librosa.sequence.dtw(
    X=chroma_comp,
    Y=chroma_ref,
    metric='cosine',
    subseq=False,
    backtrack=True
)
print(f"Accumulated Cost Matrix shape: {D.shape}")
print(f"Optimal Warping Path shape: {wp.shape}, Total alignment cost: {D[-1, -1]:.4f}")
```

#### G. Headless Visualization (`librosa.display`)
Saves plots directly to disk in a non-interactive environment.

```python
import os
import matplotlib
matplotlib.use('Agg')  # Force headless rendering
import matplotlib.pyplot as plt
import librosa
import librosa.display

y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

# 1. Waveform visualization
librosa.display.waveshow(y, sr=sr, ax=ax[0], color='blue')
ax[0].set(title="Waveform (Trumpet)")

# 2. Spectrogram visualization with logarithmic frequency axis
img = librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=512,
    x_axis='time',
    y_axis='log',
    ax=ax[1]
)
ax[1].set(title="Log-Frequency Spectrogram")
fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

# Save figure without calling plt.show()
output_image_path = "trumpet_analysis.png"
plt.tight_layout()
plt.savefig(output_image_path, dpi=150)
plt.close(fig)
print(f"Saved visualization to {output_image_path}")
```

---

## 3. Real-World Use Cases & Templates

### A. Music Synchronization (Audio Alignment)
*   **Description**: Aligning two distinct recordings of the same musical piece (e.g., a live cover vs. a studio recording).
*   **Architecture**:
    1.  Load both files and resample to a common rate (e.g., 22050 Hz).
    2.  Extract Chroma Energy Normalized Statistics (CENS) using `librosa.feature.chroma_cens`.
    3.  Compute the alignment path using `librosa.sequence.dtw`.
    4.  Map timestamps of performance A to performance B using the optimal warping path.
*   **Official Example**: [Librosa Music Synchronization Gallery](https://librosa.org/doc/0.11.0/auto_examples/plot_music_sync.html)

### B. Vocal & Instrument Separation (HPSS + Nearest-Neighbor Filtering)
*   **Description**: Isolating foreground vocals (which are transient/percussive) from background instrumental accompaniments (which are harmonic).
*   **Architecture**:
    1.  Compute the complex spectrogram `D = librosa.stft(y)`.
    2.  Separate into harmonic and percussive magnitude spectrograms `H, P = librosa.decompose.hpss(np.abs(D), margin=3.0)`.
    3.  Use nearest-neighbor filtering (`librosa.decompose.nn_filter`) on the percussive spectrogram to suppress background music and isolate vocals.
    4.  Generate soft masks, apply them to the original complex STFT `D`, and run `librosa.istft` to output isolated audio.
*   **Official Example**: [Vocal Separation Gallery](https://librosa.org/doc/0.11.0/auto_examples/plot_vocal_separation.html)

### C. Structural Audio Segmentation (Song Structure Analysis)
*   **Description**: Partitioning a musical track into distinct structural parts (e.g., verse, chorus, bridge) based on timbral and harmonic changes.
*   **Architecture**:
    1.  Detect beat frames using `librosa.beat.beat_track`.
    2.  Extract beat-synchronous Constant-Q Transform (CQT) features using `librosa.util.sync`.
    3.  Construct a weighted recurrence matrix `R` with time-lag filtering.
    4.  Compute the Normalized Laplacian of the recurrence matrix and perform spectral clustering (Agglomerative) on the eigenvectors to find segment boundaries.
*   **Official Example**: [Laplacian Segmentation Gallery](https://librosa.org/doc/0.11.0/auto_examples/plot_segmentation.html)

---

## 4. Developer Friction Points & Edge Cases

### Friction Point 1: Stereo Writing Failure with `soundfile.write`
*   **Symptom/Error**:
    ```
    RuntimeError: Error opening 'output.wav': Format not recognised.
    ```
    Or, writing a corrupted WAV file that has millions of channels but only 2 samples.
*   **Underlying Cause**: 
    `librosa.load(..., mono=False)` loads multi-channel audio in **channels-first** format, returning an array of shape `(2, n_samples)`. However, `soundfile.write` expects audio in **channels-last** format, which is `(n_samples, 2)`. When passed a channels-first array, `soundfile` interprets `n_samples` (which could be millions of elements) as the number of audio channels, causing an instant crash or massive memory allocation failure.
*   **Resolution**: 
    Transpose the multi-channel NumPy array before passing it to `soundfile.write`:
    ```python
    y, sr = librosa.load("stereo_input.mp3", sr=None, mono=False)
    # Transpose shape from (2, n_samples) to (n_samples, 2)
    import soundfile as sf
    sf.write("stereo_output.wav", y.T, sr)
    ```
*   **Reference**: [GitHub Issue #1067](https://github.com/librosa/librosa/issues/1067), [GitHub Issue #131](https://github.com/librosa/librosa/issues/131)

### Friction Point 2: Unintentional Automatic Resampling to 22050 Hz
*   **Symptom/Error**:
    No explicit error is thrown, but high-frequency content above 11025 Hz is entirely lost (due to Nyquist limits), and downstream machine learning models trained on 16000 Hz or 44100 Hz fail due to sample rate mismatches.
*   **Underlying Cause**: 
    `librosa.load` has a hardcoded default parameter `sr=22050`. Unless explicitly configured, librosa automatically resamples any loaded file (even high-quality 44.1kHz WAVs) down to 22050 Hz using `soxr` or `scipy`.
*   **Resolution**: 
    Always pass `sr=None` to preserve the native sampling rate of the file, or explicitly define the target sampling rate:
    ```python
    # Preserve native sample rate
    y, sr = librosa.load("audio.wav", sr=None)
    
    # Force target rate for speech model
    y, sr = librosa.load("audio.wav", sr=16000)
    ```
*   **Reference**: [librosa.load Documentation](https://librosa.org/doc/0.11.0/generated/librosa.load.html)

### Friction Point 3: Keyword-Only Argument Constraints in v0.10.0+
*   **Symptom/Error**:
    ```
    TypeError: melspectrogram() takes 0 positional arguments but 2 were given
    ```
*   **Underlying Cause**: 
    To prevent parameter ordering confusion (e.g., passing `sr` where `n_fft` is expected), librosa 0.10.0+ converted nearly all function signatures to **keyword-only** arguments. Only primary arrays (like `y` or `S`) can be passed positionally.
*   **Resolution**: 
    Convert all positional parameters to explicit keyword arguments:
    ```python
    # CRASHES in v0.11.0:
    melspec = librosa.feature.melspectrogram(y, sr)
    
    # SUCCESS:
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    ```
*   **Reference**: [Librosa v0.10.0 Changelog](https://librosa.org/doc/0.11.0/changelog.html)

### Friction Point 4: Removal of `librosa.output` Module
*   **Symptom/Error**:
    ```
    AttributeError: module 'librosa' has no attribute 'output'
    ```
*   **Underlying Cause**: 
    The `librosa.output` module (which contained `write_wav`) was deprecated in 0.7.0 and completely removed in 0.8.0. Old tutorials and StackOverflow posts still use `librosa.output.write_wav`.
*   **Resolution**: 
    Replace all instances of `librosa.output.write_wav` with `soundfile.write` (remembering to transpose multi-channel arrays).
*   **Reference**: [GitHub Issue #1200](https://github.com/librosa/librosa/issues/1200), [GitHub Issue #917](https://github.com/librosa/librosa/issues/917)

---

## 5. Evaluation Ideas

Below is a curated set of evaluation tasks for AI coding agents, ranging from basic operations to complex systems.

### Simple Tier
1.  **Format Transcoder & Native Resampler**: Load a compressed stereo MP3 file at its native sample rate, convert it to mono, resample it to exactly 16000 Hz using the high-quality `soxr_hq` resampler, and write it back as a 16-bit PCM WAV file.
2.  **Audio Feature Aggregator**: Extract MFCCs, Mel Spectrograms, and Spectral Centroids for an input WAV file, compute their temporal mean and standard deviation, and concatenate them into a single 1D feature vector.

### Medium Tier
3.  **Beat-Synchronous Feature Aligner**: Compute the beat frames of an audio track, extract high-resolution chroma features, and aggregate the chroma features over each beat interval using the median operator to produce a synchronized chroma matrix.
4.  **Audio Waveform HPSS Splitter**: Separate an audio file into distinct harmonic and percussive waveforms in the time domain using HPSS, trim leading/trailing silence from both resulting waveforms, and export them as separate stereo WAV files.
5.  **Dynamic Pitch Transposer & Time Stretcher**: Implement an audio augmentation utility that shifts the pitch of an audio file up by 3 semitones and compresses its duration by 15% (1.15x speed) while ensuring the output duration is accurately computed.

### Complex Tier
6.  **Structural Music Section Boundary Detector**: Build a structural segmentation pipeline that takes a music track, computes a time-delay embedded recurrence matrix from beat-synchronous chroma features, and uses agglomerative clustering to output the start and end timestamps of each major song section (e.g., intro, verse, chorus).
7.  **Headless Audio Alignment Engine**: Align two different performances of a musical piece by computing robust CENS features, executing Dynamic Time Warping (DTW) to find the optimal warping path, and outputting a clean frame-to-frame timestamp mapping without using interactive displays.
8.  **Constellation-Map Audio Fingerprinter**: Build a simplified audio fingerprinting system that extracts local spectral peaks (constellation map) from a magnitude spectrogram, generates hashes from pairs of peaks with time offsets, and matches a noisy 5-second query clip against a pre-computed database.

---

## 6. Sources

1.  [Librosa Home Page](https://librosa.org/): Core entry point for the librosa library and official documentation.
2.  [Librosa latest Changelog](https://librosa.org/doc/0.11.0/changelog.html): Detailed history of changes, deprecations, and keyword-only argument rules.
3.  [Librosa 0.11.0 Tutorial](https://librosa.org/doc/0.11.0/tutorial.html): Official introductory tutorial on package structure, beat tracking, and feature aggregation.
4.  [librosa.load API Documentation](https://librosa.org/doc/0.11.0/generated/librosa.load.html): Reference for parameter definitions, default resampling rates, and file-like object handling.
5.  [librosa.stft API Documentation](https://librosa.org/doc/0.11.0/generated/librosa.stft.html): Reference for short-time Fourier transform, padding modes, and multi-channel shape behaviors.
6.  [librosa.effects.pitch_shift API Documentation](https://librosa.org/doc/0.11.0/generated/librosa.effects.pitch_shift.html): Reference for pitch shifting, scale parameters, and resampling backends.
7.  [librosa.effects.time_stretch API Documentation](https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html): Reference for time stretching, phase vocoder mechanics, and speed factors.
8.  [librosa.segment.recurrence_matrix API Documentation](https://librosa.org/doc/0.11.0/generated/librosa.segment.recurrence_matrix.html): Reference for recurrence matrices, distance metrics, and width parameters.
9.  [librosa.sequence.dtw API Documentation](https://librosa.org/doc/0.11.0/generated/librosa.sequence.dtw.html): Reference for Dynamic Time Warping, cost accumulation, and warping path backtracking.
10. [librosa.decompose.decompose API Documentation](https://librosa.org/doc/0.11.0/generated/librosa.decompose.decompose.html): Reference for matrix factorization, NMF transformer interfaces, and multi-channel flattening.
11. [GitHub Issue #1067](https://github.com/librosa/librosa/issues/1067): Discussion on soundfile write failures and channels-first format mismatches.
12. [GitHub Issue #1200](https://github.com/librosa/librosa/issues/1200): Discussion on deprecation and removal of the `librosa.output` module.
13. [GitHub Issue #827](https://github.com/librosa/librosa/issues/827): Bug report and fix for ValueError in recurrence matrix boundary conditions.
14. [Python Soundfile Documentation](https://python-soundfile.readthedocs.io/): Reference for `soundfile.write` array format expectations and data types.
