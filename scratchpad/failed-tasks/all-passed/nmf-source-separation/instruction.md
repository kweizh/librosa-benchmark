# NMF-based Audio Source Separation

## Background
You have a librosa-based Python environment available. Your job is to build a small audio analysis utility that performs unsupervised source separation on a real audio recording by factoring its magnitude spectrogram with Non-negative Matrix Factorization (NMF), and then synthesizes a separate time-domain waveform for each learned component.

## Requirements
- Read the input audio file `/workspace/input.wav` (mono, native sample rate).
- Decompose the audio into exactly **4** NMF components using `librosa.decompose.decompose` together with `sklearn.decomposition.NMF`.
- For each of the 4 components, reconstruct a time-domain audio signal that represents that component's contribution to the original mixture.
- Write the 4 resulting audio waveforms to `/workspace/component_0.wav`, `/workspace/component_1.wav`, `/workspace/component_2.wav`, and `/workspace/component_3.wav`.

## Implementation Hints
- `librosa.decompose.decompose` operates on a non-negative feature matrix such as a magnitude spectrogram. Compute a complex STFT first so that you can later use its phase for reconstruction.
- The factorization yields `components` (spectral basis vectors) and `activations` (time-varying gains). The contribution of a single component to the mixture spectrogram is the outer product of its basis with its activation row.
- To turn a component back into a waveform that sounds like part of the original recording, recombine its magnitude with the original STFT phase before running an inverse STFT.
- Make sure each output WAV preserves the input sample rate and is close in length to the original input.

## Acceptance Criteria
- Project path: /home/user/myproject
- Input audio file: /workspace/input.wav
- Output audio files: /workspace/component_0.wav, /workspace/component_1.wav, /workspace/component_2.wav, /workspace/component_3.wav
- Ensure the solver script is executed end-to-end and the four output WAV files exist.
- Each output file must be a valid WAV file that loads with `soundfile.read`.
- Each output file must have the same sample rate as the input audio.
- Each output file's length (in samples) must be within 2048 samples of the input length.
- The sample-wise sum of the 4 component waveforms must have a cosine similarity of at least 0.95 to the input waveform (NMF approximately reconstructs the original mixture).
- Each component waveform must be non-trivial, with RMS amplitude strictly greater than 1e-5.
- The 4 components must capture different spectral content: at least two of the four component waveforms must have spectral centroids that differ from each other by more than 50 Hz.

