# Vocal / Accompaniment Separation Pipeline

## Background
The file `/workspace/input.wav` is a short music recording that contains a singer plus instrumental accompaniment. Your job is to split it into two waveforms: the foreground (the vocals) and the background (the instrumental music).

## Requirements
- Read `/workspace/input.wav`.
- Implement a mask-based separation pipeline that uses `librosa.decompose.nn_filter` (the REPET-SIM style nearest-neighbor filter on the magnitude spectrogram) together with a soft spectral mask, applied to the original complex STFT.
- Write the two reconstructed waveforms back to disk:
  - `/workspace/foreground.wav` — the vocals.
  - `/workspace/background.wav` — the instrumental accompaniment.
- Each output file must have the same sample rate and channel count as the input, and (approximately) the same length. When summed sample-by-sample, `foreground + background` should reconstruct the input mixture.

## Acceptance Criteria
- Project path: /workspace
- Input file (already present): /workspace/input.wav
- Output files (must be created by you): /workspace/foreground.wav and /workspace/background.wav
- Both outputs must be readable WAV files (via `soundfile`), with the same sample rate and channel count as the input.
- Each output length must be within 1024 samples of the input length.
- The sample-wise sum `foreground + background` must reconstruct the input closely: cosine similarity with the input >= 0.85 and normalized mean absolute error < 0.05.
- The foreground and background must be distinct (cosine similarity between them < 0.95).
- Neither output may be silent (RMS > 1e-4).

