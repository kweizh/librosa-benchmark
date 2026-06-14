# Beat-Synchronous MFCC Aggregator

## Background
Build a beat-synchronous MFCC summarizer with `librosa` (0.11). The pipeline must beat-track a short rhythmic clip, compute MFCCs at the same hop length, aggregate MFCC frames into per-beat-interval medians, and persist the synchronized matrix together with the beat timing metadata.

## Requirements
- Read the input WAV from `/workspace/input.wav`.
- Estimate the tempo and the beat frame indices with `librosa.beat.beat_track`.
- Compute 20-coefficient MFCCs (`n_mfcc=20`) at the **same `hop_length`** as the beat tracker.
- Aggregate the MFCC matrix to one column per beat interval using `librosa.util.sync` with **median aggregation**.
- Write the synchronized matrix to `/workspace/beat_mfcc.npz` under the key `mfcc_sync` (shape `(20, n_beat_intervals)`).
- Write the beat metadata to `/workspace/beats.json`.

## Implementation Hints
- Pick a `hop_length` once and reuse it for both `librosa.beat.beat_track` and `librosa.feature.mfcc`.
- `librosa.beat.beat_track` returns the tempo as a 1-D ndarray even for mono input; reduce it to a scalar before serialising to JSON.
- `librosa.util.sync` accepts an `aggregate` callable (e.g. `np.median`) and pads boundaries by default; verify the resulting column count against `len(beat_times_seconds)` and record the chosen convention.
- Use `librosa.frames_to_time` (with the same `sr` and `hop_length`) to convert beat frame indices to seconds.
- Sanity-check every API signature against the librosa 0.11.0 documentation; feature, beat, and util functions are keyword-only.

## Acceptance Criteria
- Project path: /workspace
- Ensure the pipeline is executed and both output artifacts exist.
- Output files:
  - `/workspace/beat_mfcc.npz` containing the key `mfcc_sync` with shape `(20, n_beat_intervals)`.
  - `/workspace/beats.json` containing a JSON object with the following schema:

    ```json
    {
      "tempo_bpm": number,
      "beat_times_seconds": [number, ...],
      "hop_length": integer,
      "sample_rate": integer,
      "n_mfcc": 20
    }
    ```

  - `tempo_bpm` must be a float in the inclusive range `[40.0, 240.0]`.
  - `beat_times_seconds` must be a strictly increasing list of floats; every element must lie in `[0, audio_duration + 1e-3]`.
  - `hop_length` and `sample_rate` must be positive integers; `n_mfcc` must be `20`.
- The column count of `mfcc_sync` (`n_beat_intervals`) must be internally consistent with the chosen `librosa.util.sync` boundary convention; both `n_beat_intervals == len(beat_times_seconds) - 1` and `n_beat_intervals == len(beat_times_seconds)` are accepted.
- Re-computing MFCCs with the recorded `hop_length`, `sample_rate`, and `n_mfcc=20` and aggregating per beat interval with the median must reproduce each stored column of `mfcc_sync` within an absolute tolerance of `1e-3` for at least 80% of beat intervals.

