# Audio Augmentation Pipeline with `librosa.effects`

## Background
Build an ordered audio augmentation pipeline using `librosa` 0.11. The pipeline reads a single mono WAV, applies a fixed sequence of effects (pitch shift, time stretch, then silence trim), and persists both the resulting waveform and a metadata log capturing per-stage durations and trim indices.

## Requirements
- Read the input audio from `/workspace/input.wav`.
- Apply the following stages, in this exact order, on the loaded waveform:
  1. Pitch shift up by `+3` semitones with `bins_per_octave=12` and energy scaling enabled (`scale=True`).
  2. Time-stretch the pitch-shifted signal by `rate=0.85` (slow down).
  3. Trim leading/trailing silence with `top_db=40`.
- Write the final waveform to `/workspace/augmented.wav` as a mono file with the same sample rate as the input.
- Write a metadata JSON log to `/workspace/aug_meta.json` describing the original sample rate, the per-stage durations (in seconds, measured on the waveform produced by that stage), the trim indices returned by the trim stage, and the parameters used.

## Implementation Hints
- Use `librosa.effects.pitch_shift`, `librosa.effects.time_stretch`, and `librosa.effects.trim` for the three stages. All effect parameters in `librosa` 0.11 are keyword-only.
- Measure each stage's duration from the waveform produced by that stage (length divided by sample rate).
- The trim stage returns both the trimmed waveform and a 2-element `index` array; persist that index as `trim_indices`.
- Use `soundfile` (or `librosa` helpers built on top of it) to write the final WAV file; preserve the input sample rate.
- Cross-check API signatures against the `librosa` 0.11.0 documentation; do not assume positional argument order.

## Acceptance Criteria
- Project path: /workspace
- Ensure the augmentation pipeline is executed and the output artifacts exist.
- Output files:
  - `/workspace/augmented.wav`: mono WAV whose sample rate matches the input.
  - `/workspace/aug_meta.json`: JSON object matching the schema below.

  ```json
  {
    "sample_rate": number,
    "input_duration_seconds": number,
    "after_pitch_shift_seconds": number,
    "after_time_stretch_seconds": number,
    "after_trim_seconds": number,
    "trim_indices": [number, number],
    "n_steps": 3.0,
    "rate": 0.85,
    "top_db": 40
  }
  ```

- The augmented WAV sample count must equal `after_trim_seconds * sample_rate` within ±1 sample.
- Pitch shift must preserve length within phase-vocoder rounding: `|after_pitch_shift_seconds - input_duration_seconds| / input_duration_seconds <= 0.02`.
- Time stretch must satisfy `|after_time_stretch_seconds - after_pitch_shift_seconds / 0.85| / after_time_stretch_seconds <= 0.05`.
- `after_trim_seconds <= after_time_stretch_seconds`.
- `trim_indices` must be a `[start, end]` integer pair with `0 <= start < end` and `end <= round(after_time_stretch_seconds * sample_rate)`.
- The median voiced F0 of the augmented audio (computed via `librosa.pyin`) divided by that of the input must lie in `[2**(2.5/12), 2**(3.5/12)]` (i.e., `+3` semitones ±0.5 semitone).

