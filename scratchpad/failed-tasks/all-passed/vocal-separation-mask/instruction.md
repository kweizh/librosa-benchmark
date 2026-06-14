# Vocal / Accompaniment Separation with Soft Masks

## Background
Use `librosa` to implement a foreground (vocal) vs. background (accompaniment) source separator based on nearest-neighbor median filtering of the magnitude STFT and complementary soft masks. The pipeline operates in the magnitude STFT domain, estimates a background by aggregating spectrally similar frames, derives complementary soft masks for vocal and accompaniment, applies each mask to the original complex STFT, and reconstructs two waveforms via the inverse STFT.

## Requirements
- Read the input mixture from `/workspace/input.wav` (mono).
- Produce two output WAV files at the same sample rate as the input:
  - `/workspace/vocal.wav` (foreground)
  - `/workspace/accompaniment.wav` (background)
- The two outputs must be derived from complementary soft masks applied to the original complex STFT (no naive subtraction of waveforms).

## Implementation Hints
- Allowed librosa APIs for the core separation: `librosa.load`, `librosa.stft`, `librosa.istft`, `librosa.decompose.nn_filter`, `librosa.util.softmask`.
- The background magnitude estimate must use `librosa.decompose.nn_filter` with `aggregate=np.median` and a non-local cosine recurrence width constraint to prevent neighbors that are too close in time from being aggregated.
- Complementary masks must be obtained from `librosa.util.softmask` (one mask for vocal, one for accompaniment), then multiplied element-wise with the original complex STFT before inverse transform.
- Use the same STFT parameters (`n_fft`, `hop_length`, `win_length`) for analysis and synthesis so the reconstructed signal length matches the input.
- Save outputs with `soundfile.write` (or equivalent) at the loaded sample rate; do not resample.

## Acceptance Criteria
- Project path: /workspace
- Ensure the separation pipeline is executed and both output artifacts exist.
- Output files:
  - `/workspace/vocal.wav`
  - `/workspace/accompaniment.wav`
- Both outputs are valid WAV files loadable by `soundfile`, mono, and have the same sample rate as the input.
- Each output has a length within ±1 sample of the input length.
- Each output is non-silent (per-file RMS energy strictly greater than 1e-4).
- The element-wise sum of the two output waveforms reconstructs the input mixture to a mean-absolute-error of at most 5e-2 (soft masks must be complementary).
- The cosine similarity between the magnitude STFTs of `vocal.wav` and `accompaniment.wav` (flattened) must be strictly less than the cosine similarity between the input mixture magnitude STFT and `accompaniment.wav` magnitude STFT (the masks must actually separate spectral content rather than copy the mixture).
- Neither output may be a bitwise copy of the input.

