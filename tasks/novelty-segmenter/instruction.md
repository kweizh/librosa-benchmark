# Structural Boundary Detection with a Foote Novelty Curve

## Background
Structural segmentation of an audio recording can be performed by analyzing how a frame-level feature representation repeats over time. Foote's classic approach builds a self-similarity matrix (SSM) over short-time spectral features and convolves the diagonal of the SSM with a checkerboard kernel; large responses correspond to instants where the feature statistics change abruptly. Your job is to implement this novelty-curve pipeline with `librosa` (v0.11.0) and report the detected change-point timestamps.

## Requirements
- Read the input audio from `/workspace/input.wav`.
- Compute a frame-level spectral feature representation (MFCCs over time).
- Build a self-similarity matrix from the feature sequence using a cosine metric.
- Construct a Foote-style checkerboard kernel and convolve it along the diagonal of the SSM to obtain a one-dimensional novelty curve.
- Pick peaks from the novelty curve to identify structural change-points.
- Convert each peak from a frame index to a timestamp in seconds.
- Write the timestamps to `/workspace/boundaries.json` using the schema:
  ```json
  {
    "boundaries_sec": [<float>, <float>, ...]
  }
  ```
  Values must be in seconds, strictly increasing, and lie inside the audio's duration.

## Implementation Hints
- Inspect `librosa.feature.mfcc` for MFCC extraction and `librosa.segment.recurrence_matrix` (with `mode='affinity'`, `metric='cosine'`, and `full=True`) for the dense self-similarity matrix.
- Build the checkerboard kernel yourself: a square matrix with positive values in the top-left and bottom-right quadrants and negative values in the off-diagonal quadrants, optionally tapered with a 2-D Gaussian window. Do not rely on any pre-built Foote shortcut in librosa.
- Slide the kernel along the diagonal of the SSM to produce the novelty curve; mind boundary effects.
- Convert frame indices to time using `librosa.frames_to_time` with the same `sr` and `hop_length` used for the MFCCs.
- Use any peak-picking strategy you like (e.g., `librosa.util.peak_pick` or `scipy.signal.find_peaks`); tune it to ignore trivial peaks near the very start and very end of the signal.

## Acceptance Criteria
- Project path: /workspace
- Ensure the script is executed and the output artifact exists.
- Output file: /workspace/boundaries.json
- The file must be valid JSON containing an object with a `boundaries_sec` key.
- `boundaries_sec` must be a list of numbers (in seconds), strictly increasing, with every value in the interval `[0, audio_duration]`.
- At least two structural boundaries must be reported.
- The detected boundaries must correspond to the major structural changes present in the input audio (verified against ground-truth change-points baked into the test environment).

