# Mel Spectrogram Resynthesis via Griffin-Lim

## Background
Build a small analysis-resynthesis pipeline with `librosa`. The pipeline must analyze a single audio file into a 128-band log-power mel spectrogram and then reconstruct a time-domain waveform from that mel representation using Griffin-Lim phase recovery. The reconstruction must be reported alongside two objective fidelity metrics computed against the original waveform.

## Requirements
- Read the input WAV file from `/workspace/input.wav` as a mono signal at its native sample rate.
- Compute a 128-band power mel spectrogram and represent it in a log-power form suitable for storage / downstream use.
- Resynthesize a waveform from the mel representation using Griffin-Lim phase estimation (either through the mel-to-audio convenience wrapper, or by inverting mel to STFT magnitude first and then running Griffin-Lim explicitly).
- Write the reconstructed mono waveform to `/workspace/reconstructed.wav` at the same sample rate as the input.
- Compute two reconstruction quality metrics relative to the original waveform and write them, along with run metadata, to `/workspace/metrics.json`.

## Implementation Hints
- Verify all relevant signatures against the librosa 0.11.0 documentation; most spectral / inverse-spectral functions are keyword-only.
- The convenience wrapper for the full pipeline is `librosa.feature.inverse.mel_to_audio`; the explicit two-step path is `librosa.feature.inverse.mel_to_stft` followed by `librosa.griffinlim`. Either approach is acceptable as long as the STFT analysis parameters used for inversion match those used during analysis.
- Pick `hop_length`, `win_length`, and `n_iter` yourself; defaults from the docs are a reasonable starting point but the agent may tune them to satisfy the fidelity targets.
- Make sure the reconstructed waveform length matches the input length closely; the inverse functions expose a `length` argument that can be used to enforce this.
- For the SNR computation, treat the original waveform as the reference signal and the reconstruction as the reference plus reconstruction error; align lengths before subtracting.
- For spectral convergence, compare magnitude STFTs of the original and the reconstruction computed with the same analysis parameters.

## Acceptance Criteria
- Project path: /workspace
- Ensure the resynthesis pipeline is executed and both output artifacts exist.
- Output audio file: `/workspace/reconstructed.wav`
  - Mono, sample rate equal to the input sample rate.
  - Total number of samples must be within 2% of the input waveform length.
- Output metrics file: `/workspace/metrics.json`
  - Must be a JSON object with the following schema:

    ```json
    {
      "spectral_convergence": number,
      "snr_db": number,
      "length_samples": integer,
      "sample_rate": integer,
      "n_mels": integer,
      "n_iter": integer
    }
    ```

  - `length_samples` must equal the actual sample count of `/workspace/reconstructed.wav`.
  - `sample_rate` must equal the sample rate of `/workspace/reconstructed.wav`.
  - `n_mels` must be >= 128.
  - `n_iter` must be >= 32.
  - `spectral_convergence` is defined as `||abs(STFT_ref) - abs(STFT_recon)||_F / ||abs(STFT_ref)||_F` and must be a finite float strictly less than 0.5.
  - `snr_db` must be a finite float strictly greater than 0.0.

