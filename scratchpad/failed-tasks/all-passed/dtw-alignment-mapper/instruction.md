# Headless Music Alignment with CENS Chroma + DTW

## Background
Build a headless audio alignment engine using `librosa`. Two short WAV files are provided: a reference recording and a tempo-altered (time-stretched) performance of the same content. The job is to align them in the chroma domain using Dynamic Time Warping, then emit a structured JSON describing the warping path, a dense reference-to-comparison timestamp mapping, and the final accumulated alignment cost.

## Requirements
- Load `/workspace/reference.wav` and `/workspace/comparison.wav` (both mono, 22050 Hz).
- Compute robust, smoothed chroma features that are suitable for cross-recording matching for each signal.
- Run Dynamic Time Warping in the chroma feature space using a cosine cost and recover the optimal warping path via backtracking.
- Write the alignment results to `/workspace/alignment.json`.

## Implementation Hints
- Pick a chroma representation that is robust to dynamics and timbre and that smooths over short transients.
- Cosine distance is a natural choice for chroma vectors; let DTW compute it for you via the appropriate `metric`.
- The warping path returned by the DTW backtracker is not in chronological order; you will likely need to reorient it before using it.
- The accumulated cost matrix's terminal corner holds the total alignment cost.
- Use `librosa.frames_to_time` (with the same `hop_length` used for the chroma) to translate frame indices to seconds when sampling the timestamp map.
- All `librosa.feature` and `librosa.sequence` parameters in 0.11.0 are keyword-only; check the docs.

## Acceptance Criteria
- Project path: /workspace
- Ensure the alignment pipeline is executed and the output artifact exists.
- Output file: `/workspace/alignment.json`
- The output file must be a JSON object with exactly these top-level keys:

  ```json
  {
    "warping_path_frames": [[int, int], ...],
    "timestamp_map": [{"ref_time": number, "comp_time": number}, ...],
    "total_cost": number,
    "hop_length": int,
    "sample_rate": int
  }
  ```

- `warping_path_frames` must be non-empty, contain only `[int, int]` pairs, be monotonically non-decreasing in BOTH the reference axis and the comparison axis (no step that decreases either index), start near `(0, 0)` (both indices `<= 2`), and end near `(N_ref - 1, N_comp - 1)` within a tolerance of 3 frames on each axis, where `N_ref` and `N_comp` are the chroma frame counts of the reference and comparison signals.
- `timestamp_map` must be sorted by `ref_time`. Consecutive `ref_time` deltas must fall in `[0.45, 0.55]` seconds, the first `ref_time` must be `<= 0.25`, and the last `ref_time` must be within `0.5` of the reference audio duration. All `comp_time` values must be monotonically non-decreasing and must lie in `[0, comparison_duration + 0.1]`.
- `total_cost` must be a finite, non-negative float.
- `hop_length` and `sample_rate` must be positive integers and must reflect the configuration that was actually used to compute the chroma features and frame timestamps.

