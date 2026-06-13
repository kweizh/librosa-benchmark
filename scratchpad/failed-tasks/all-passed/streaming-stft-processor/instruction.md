# Block-Streaming RMS Energy Analyzer with `librosa.stream`

## Background
Large audio files often do not fit in memory, so librosa exposes `librosa.stream` to iterate over the signal as fixed-size, frame-aligned blocks. In this task you will build a streaming RMS energy analyzer: for every block yielded by `librosa.stream`, you must compute the root-mean-square energy and persist it to disk together with the time interval that the block covers.

A short mono WAV file is already available at `/workspace/input.wav` (22050 Hz). You will process it block by block and write a CSV summary that downstream tools can consume without re-reading the audio.

## Requirements
- Read `/workspace/input.wav` exclusively through `librosa.stream` (not `librosa.load`) using the exact parameters listed in the Acceptance Criteria.
- For every yielded block, compute the RMS energy of that block and the time interval (in seconds) that the block covers in the original signal.
- Write the per-block records to `/workspace/rms_stream.csv` with the exact header and column order shown below.
- The work script must be re-runnable: invoking it again should overwrite the CSV and produce the same content for the same input.

## Implementation Hints
- `librosa.stream` returns a generator of NumPy arrays; you are expected to bookkeep the block index and the corresponding sample offset yourself.
- You may compute the per-block RMS either by calling `librosa.feature.rms` on the block and aggregating, or by computing `sqrt(mean(square))` directly on the block samples. Either approach is acceptable as long as the result reflects the RMS of that specific block (not of the whole file).
- Recall that when streaming you should disable `center` padding in any frame-based analysis you call on the block.
- Use `librosa.get_samplerate` (or read the sample rate from the file) when converting sample offsets to seconds; do not assume a hard-coded rate inside your computation.

## Acceptance Criteria
- Project path: /workspace
- Input audio: /workspace/input.wav (mono, 22050 Hz, several seconds)
- Command: `python3 /workspace/solution.py` (or any script the agent creates) must produce the CSV when run from `/workspace`.
- Output file: /workspace/rms_stream.csv
- The CSV must use the exact header (in this order): `block_index,start_time_sec,end_time_sec,rms`
- Streaming parameters (these are part of the contract):
  - `block_length = 16`
  - `frame_length = 2048`
  - `hop_length = 512`
- The CSV must contain at least 2 rows of data (excluding the header).
- `block_index` must start at 0 and be strictly increasing by 1 per row.
- `start_time_sec` and `end_time_sec` must each be monotonically increasing across rows; the first `start_time_sec` must be within 0.001 s of 0.
- The last `end_time_sec` must be within 0.5 s of the true duration of `/workspace/input.wav`.
- The block-duration-weighted RMS energy reconstructed from the CSV must be within 5% of the global RMS energy of the full signal.
- Solution must call `librosa.stream`; do not load the entire file with `librosa.load` and slice it manually.

