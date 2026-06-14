# Reassigned Spectrogram Peak Tracking

## Background
Build a peak-tracking analysis on top of the time-frequency *reassigned* spectrogram in `librosa`. Unlike a plain STFT, the reassigned spectrogram yields refined per-bin instantaneous frequency estimates (and may produce `NaN` for low-power bins). Your job is to extract, for every analysis frame, the most prominent spectral peaks and report their reassigned frequencies plus magnitudes in dB.

## Requirements
- Read the input audio file from `/workspace/input.wav`.
- Compute the time-frequency reassigned spectrogram of the signal using `librosa`.
- For each STFT frame, select the **top-5** spectral bins ranked by reassigned magnitude (after converting magnitudes to dB). Handle `NaN` values produced by reassignment so they never appear in the output and never break sort order.
- For each selected peak, report:
  - The **reassigned instantaneous frequency** in Hz (from the frequency output of the reassigned spectrogram).
  - The magnitude in **decibels (dB)**.
- Write the result to `/workspace/peaks.json`.

## Implementation Hints
- Sanity-check the API signature and return tuple order against the librosa 0.11.0 documentation. The reassigned spectrogram returns three parallel arrays of shape `(1 + n_fft/2, n_frames)`.
- The frequency and time arrays may contain `NaN` for bins whose power falls below the reassignment threshold; these must be excluded from peak selection rather than silently propagated.
- Use a librosa helper to convert linear amplitude magnitudes to dB.
- Pick an `n_fft` / `hop_length` combination that yields a deterministic frame count for a 22050 Hz, ~5 s input and record that frame count in the output.
- The number of frames in your JSON output must match what librosa actually produces for the parameters you chose — do not pad, truncate, or invent frames.

## Acceptance Criteria
- Project path: /workspace
- Ensure the analysis pipeline is executed and the output artifact exists.
- Output file: `/workspace/peaks.json`
- The file must be valid JSON. The top-level value must be a JSON **object** with the following schema:

  ```json
  {
    "meta": {
      "n_frames": number,
      "sr": number,
      "n_fft": number,
      "hop_length": number
    },
    "frames": [
      {
        "time": number,
        "peaks": [
          {"freq_hz": number, "magnitude_db": number}
        ]
      }
    ]
  }
  ```

- `meta.n_frames` must equal `len(frames)` and must equal the number of STFT frames produced by the chosen `n_fft` / `hop_length`.
- `frames` length equals `meta.n_frames`.
- Each frame's `time` is a finite float in seconds, monotonically non-decreasing across frames, lying in `[0, audio_duration + 1e-2]`, with the last frame within `0.1` s of the audio duration.
- Each frame's `peaks` array contains **exactly 5** entries.
- Each peak: `freq_hz` is finite and in `(0.0, sr/2]`; `magnitude_db` is finite.
- Within each frame, peaks are sorted by `magnitude_db` in **descending** order.
- At least 50% of frames must contain at least one peak whose `freq_hz` is within ±10 Hz of one of the input tones at 220 Hz, 440 Hz, or 880 Hz.

