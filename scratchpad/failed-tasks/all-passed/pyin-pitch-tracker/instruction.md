# Monophonic Pitch Tracker with Probabilistic YIN (pYIN)

## Background
Librosa ships an implementation of the probabilistic YIN (pYIN) algorithm for fundamental-frequency (F0) tracking of monophonic signals. You will use `librosa.pyin` to build a small offline pitch tracker for a single audio file.

## Requirements
- Read the input audio from `/workspace/input.wav` (mono, 22050 Hz, single take).
- Estimate the fundamental frequency (in Hz) for every analysis frame using `librosa.pyin`.
- Distinguish voiced from unvoiced frames using pYIN's voicing output.
- Write the per-frame results to `/workspace/pitch.csv` using the schema below.

## Implementation Hints
- Use `librosa.load` to read the WAV file. The recording uses a single channel.
- `librosa.pyin` requires `fmin`/`fmax` to be set as keyword arguments. Pick a range that comfortably contains the human-vocal / instrumental sweep frequencies in the input (the input is known to stay between about 100 Hz and 1000 Hz).
- Convert the per-frame indices to seconds with the same `sr` and `hop_length` you passed to `librosa.pyin` (e.g. via `librosa.times_like` or `librosa.frames_to_time`).
- Unvoiced frames typically come back as `NaN` from `librosa.pyin` (its default `fill_na=np.nan`). Keep that behaviour or write `0` for unvoiced frames; both are acceptable.

## Acceptance Criteria
- Project path: /workspace
- Ensure the script is executed and the artifact `/workspace/pitch.csv` exists.
- Output file: /workspace/pitch.csv
- The CSV must include a header row with exactly these three column names (in this order): `time_sec,frequency_hz,voiced`.
- One row per analysis frame. The number of data rows (excluding header) must be at least 50.
- `time_sec` values must be monotonically non-decreasing and the first row must start at (or extremely close to) `0.0`.
- `frequency_hz` must be a finite numeric value for voiced frames and may be either `NaN`/empty or `0` for unvoiced frames.
- `voiced` must be one of the boolean encodings `true`/`false`, `True`/`False`, or `1`/`0` (any one of these encodings is acceptable, but it must be used consistently for the whole file).

