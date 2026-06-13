# Griffin-Lim Phase Reconstruction

## Background
You are given only the magnitude of a Short-Time Fourier Transform (STFT) of a real-valued audio signal. The original phase information has been discarded. Your task is to recover an audio waveform whose magnitude STFT closely matches the provided magnitude, using librosa's Griffin-Lim algorithm.

## Requirements
- Read the magnitude spectrogram from `/workspace/magnitude.npy`. It is a 2D `float32` NumPy array of shape `(1 + n_fft // 2, n_frames)` produced by `librosa.stft` from a real-valued mono signal.
- Reconstruct a real-valued time-domain waveform using `librosa.griffinlim`.
- Write the reconstructed waveform to `/workspace/reconstructed.wav` as a mono WAV file at a sample rate of **22050 Hz**.

## Implementation Hints
- Infer the STFT frame size from the first dimension of the input array: if the input has `n_bins` frequency rows, then `n_fft = 2 * (n_bins - 1)`.
- Pick a hop length that is consistent with the STFT used to produce the magnitude (a power-of-two fraction of `n_fft` is conventional for `librosa.stft`).
- Run Griffin-Lim with enough iterations for the phase estimates to converge to a good approximation.
- Use `soundfile` (or another correct WAV writer) to save the result as a standard PCM/float WAV at 22050 Hz.

## Acceptance Criteria
- Project path: /workspace
- Output file: `/workspace/reconstructed.wav` must exist, be readable, contain a single audio channel, and have a sample rate of exactly 22050 Hz.
- The reconstructed waveform's magnitude STFT (computed with the same STFT parameters used to produce the input) must closely match the input magnitude spectrogram.
- The reconstructed waveform's length must be approximately equal to the original reference signal length (within a few hop lengths).
- The reconstructed waveform's peak absolute amplitude must lie in the audible, non-clipping range [0.01, 1.0].

