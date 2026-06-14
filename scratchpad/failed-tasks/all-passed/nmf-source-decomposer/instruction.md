# NMF Source Decomposition with Wiener Soft Masks

## Background
Decompose a polyphonic mixture into 4 spectral sources using NMF, then resynthesize each source back to a waveform using Wiener-style soft masking applied to the original complex STFT.

## Requirements
- Read `/workspace/mixture.wav` (mono, 22050 Hz).
- Decompose its magnitude spectrogram into 4 NMF components with `librosa.decompose.decompose`.
- For each component, build a Wiener-style soft mask, apply it to the original complex STFT, and resynthesize to a waveform via the inverse STFT. The summed component waveforms should reconstruct the mixture.
- Write each resynthesized component to `/workspace/component_<i>.wav` for `i` in `0..3`.
- Write per-component spectral statistics to `/workspace/components.json`.

## Implementation Hints
- Keep STFT/iSTFT parameters consistent so component lengths match the mixture.
- The soft mask for component `k` should be its magnitude-squared spectrogram divided by the sum of magnitude-squared spectrograms across all components.
- Compute the spectral centroid per component from its resynthesized waveform.
- Verify API signatures against the librosa 0.11.0 docs.

## Acceptance Criteria
- Project path: /workspace
- Ensure the decomposition pipeline is executed and the output artifacts exist.
- Output audio files: `/workspace/component_0.wav`, `/workspace/component_1.wav`, `/workspace/component_2.wav`, `/workspace/component_3.wav`.
  - Each file must be mono, with the same sample rate as `/workspace/mixture.wav`, and the same sample count as the mixture (±1 sample).
- Output metadata file: `/workspace/components.json`.
  - Must be a JSON array of exactly 4 objects, each with the schema:

    ```json
    {
      "index": integer,
      "centroid_hz": number,
      "rms": number
    }
    ```

  - The `index` values must be exactly `0, 1, 2, 3` (each appearing once).
  - Every `centroid_hz` must satisfy `0 < centroid_hz < sr / 2`.
  - The spread `max(centroid_hz) - min(centroid_hz)` across the 4 components must exceed 200 Hz.
  - Every `rms` must exceed `1e-4`.
- Reconstruction quality: the sample-wise sum of the 4 component waveforms must approximate the mixture such that the mean absolute error is at most `5e-2` times the mixture RMS.

