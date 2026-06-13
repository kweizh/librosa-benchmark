# Beat-Synchronous CQT Visualization

## Background
Produce a beat-synchronous Constant-Q Transform (CQT) representation of an audio file, both as a numeric artifact and as a headless matplotlib rendering. The goal is to combine constant-Q spectral analysis, beat tracking, and per-beat aggregation into a single deterministic offline pipeline.

## Requirements
- Read the input audio from `/workspace/input.wav`.
- Compute a magnitude CQT of the signal and convert it to decibels referenced to the peak value.
- Track beats in the same audio and aggregate the dB CQT across each beat interval using the median.
- Write the resulting beat-synchronous CQT dB matrix as a 2D float32 NumPy array to `/workspace/cqt_db.npy` with shape `(n_cqt_bins, n_beats)`.
- Render the same beat-synchronous matrix using `librosa.display.specshow(..., y_axis='cqt_note', x_axis='time')` to `/workspace/cqt.png` at 150 dpi.
- The figure must be saved to disk (not shown interactively); the matplotlib backend must be headless.

## Implementation Hints
- Use `librosa.cqt`, `librosa.amplitude_to_db`, `librosa.beat.beat_track`, and `librosa.util.sync` (with `aggregate=np.median`) as the core primitives.
- The hop length used by the CQT must match the hop length used by `beat_track`, so that beat frame indices align with CQT columns before aggregation.
- `librosa.util.sync` aggregates along the last axis by default, which is exactly the time axis of the CQT matrix.
- Force matplotlib into the `Agg` backend before importing `pyplot` so the script runs in a non-interactive container.
- Cast the final beat-synchronous dB matrix to `float32` before saving with `numpy.save`.

## Acceptance Criteria
- Project path: /workspace
- Ensure the script is executed and the artifacts exist.
- Input file: /workspace/input.wav
- Output numeric artifact: /workspace/cqt_db.npy
  - 2D NumPy array of floating-point values.
  - Shape `(n_cqt_bins, n_beats)` with at least a few beats (>= 5 columns).
  - Values are in decibel scale (relative to the peak), so almost all values are <= 0 and the maximum is at or very near 0.
  - No NaN or Inf entries.
- Output image artifact: /workspace/cqt.png
  - Valid PNG, openable with PIL.
  - At least 300 px wide and 200 px tall.
  - Non-trivial visual content (standard deviation of pixel intensities greater than 10), i.e. not a blank/uniform image.

