# Multi-Resolution Spectrogram Stack with librosa

## Background
Produce a stacked time-frequency representation of a single audio file by computing three complementary spectrograms on the SAME time grid: a linear STFT, a Constant-Q transform (CQT), and a Variable-Q transform (VQT). Persist the dB-scaled magnitudes in one `.npz` archive together with a sidecar metadata JSON that fully describes the analysis parameters and the per-representation frequency vectors.

## Requirements
- Read the input WAV file from `/workspace/input.wav`.
- Compute all three transforms (STFT, CQT, VQT) using a SINGLE shared `hop_length` so the columns of every output share a common time grid.
- Convert each magnitude spectrogram to dB.
- Save the three dB arrays into `/workspace/spec_stack.npz` under the keys `stft_db`, `cqt_db`, `vqt_db`.
- Save the analysis metadata as JSON into `/workspace/spec_meta.json`.

## Implementation Hints
- Use `librosa.stft`, `librosa.cqt`, and `librosa.vqt` from librosa 0.11.0; nearly every kwarg is keyword-only.
- Take magnitudes (`np.abs`) before converting to dB with `librosa.amplitude_to_db`.
- The number of frames for STFT, CQT, and VQT must agree; choose a `hop_length` that is a power of two and divisible by `2 ** (n_octaves - 1)` so the recursive CQT/VQT implementation accepts it.
- The frequency vectors should be derived from librosa's own helpers (consult the 0.11.0 docs to pick the correct helper for each transform).
- All metadata numeric values must be plain JSON-serializable scalars; the frequency arrays must be plain JSON lists of numbers.

## Acceptance Criteria
- Project path: /workspace
- Ensure the spectrogram pipeline is executed and BOTH output artifacts exist.
- Output files:
  - `/workspace/spec_stack.npz` containing exactly the arrays `stft_db`, `cqt_db`, `vqt_db` (real-valued, finite).
  - `/workspace/spec_meta.json` containing the keys `n_frames`, `hop_length`, `sample_rate`, `stft_freqs`, `cqt_freqs`, `vqt_freqs`, `n_fft`, `cqt_n_bins`, `cqt_bins_per_octave`, `vqt_n_bins`, `vqt_bins_per_octave`.
- Shape constraints:
  - `stft_db.shape == (1 + n_fft // 2, n_frames)`.
  - `cqt_db.shape == (cqt_n_bins, n_frames)`.
  - `vqt_db.shape == (vqt_n_bins, n_frames)`.
  - The trailing dimension of all three arrays equals the metadata `n_frames`.
- Frequency-vector constraints:
  - `len(stft_freqs) == stft_db.shape[0]`, `len(cqt_freqs) == cqt_n_bins`, `len(vqt_freqs) == vqt_n_bins`.
  - `cqt_freqs` is element-wise close to the standard librosa Constant-Q frequency helper output for the recorded `cqt_n_bins`, `fmin`, `bins_per_octave`.
  - `vqt_freqs` is element-wise close to the standard librosa interval-based frequency helper output for `intervals='equal'` with the recorded `vqt_n_bins`, `fmin`, `bins_per_octave`.
- Each dB array must be finite everywhere and have dynamic range `> 20 dB` (i.e. `max - min > 20`).

