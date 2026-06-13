# HPSS Multi-Stem Mixer

## Background
The file `/workspace/input.wav` is a short music recording. Build a small creative remixer that splits it into harmonic and percussive stems, processes each stem independently, and writes the mixed result back to disk.

## Requirements
Read `/workspace/input.wav` (preserving its native sample rate) and produce `/workspace/output.wav` by performing **exactly** the following steps, in order:

1. Decompose the input into harmonic and percussive time-domain components using `librosa.effects.hpss` with `margin=(1.0, 5.0)` (strong percussive mask).
2. Pitch-shift the harmonic component up by **7 semitones** using `librosa.effects.pitch_shift`.
3. Time-stretch the percussive component by a rate of **0.85** (i.e. slow it down) using `librosa.effects.time_stretch`.
4. Re-align the lengths of the pitch-shifted harmonic and the time-stretched percussive signals so they both have the same length as the original input waveform (pad with zeros or truncate as needed), and mix them with weights: `mix = 0.7 * harmonic_shifted + 0.5 * percussive_stretched`.
5. Trim leading and trailing silence from the mix with `librosa.effects.trim(top_db=30)`.
6. Write the result to `/workspace/output.wav` at the **same sample rate as the input**.

## Implementation Hints
- `librosa.effects.hpss` returns time-domain waveforms of the same length as the input.
- `librosa.effects.pitch_shift` preserves the input length, but `librosa.effects.time_stretch` changes it; you will need to pad/truncate before mixing.
- You may use `soundfile.write` to save the WAV.

## Acceptance Criteria
- Project path: /workspace
- Input file (already present): /workspace/input.wav
- Output file (must be created by you): /workspace/output.wav
- The output must be a readable WAV file (via `soundfile`).
- The output sample rate must equal the input sample rate.
- The output length (in samples) must be less than or equal to the input length (because of trimming).
- The output peak absolute amplitude must lie in the range [0.05, 1.0].
- The spectral centroid of the output (mean over frames) must be strictly higher than that of the input, reflecting the upward harmonic pitch shift.
- The estimated tempo of the output (via `librosa.feature.tempo`) must be at most 0.95x the estimated tempo of the input, reflecting the percussive slow-down.
- The output must not be identical to the input: cosine similarity between the (length-aligned) input and output waveforms must be strictly less than 0.95.

