# Multi-Candidate Tempo Estimation via Fourier Tempogram

## Background
Estimate the top-3 tempo candidates of `/workspace/input.wav` using `librosa`'s Fourier tempogram. Rank candidates by salience and cross-validate against the primary BPM estimator.

## Requirements
- Load `/workspace/input.wav` mono at its native sample rate.
- Compute an onset strength envelope and a Fourier tempogram on it.
- Identify the three most salient periodicity peaks in the global tempogram spectrum, restricted to the BPM range `[40.0, 240.0]`.
- Convert peaks to BPM, assign a `harmonic_rank` (1, 2, 3 in some assignment), and emit them sorted by salience descending.
- Write `/workspace/tempo_candidates.json`.

## Implementation Hints
- Use `librosa.feature.fourier_tempogram` and `librosa.tempo_frequencies` (the Fourier variant maps to frequencies via `sr / hop_length`; remember its bin layout differs from the autocorrelation tempogram).
- Use a global statistic (e.g., mean magnitude over time) to score each tempo bin's salience.
- Cross-validate one candidate against `librosa.feature.rhythm.tempo` (a.k.a. `librosa.feature.tempo`) with `aggregate=np.mean`.
- All `librosa` rhythm/feature APIs in 0.11 are keyword-only.

## Acceptance Criteria
- Project path: /workspace
- Ensure the pipeline is executed and the output artifact exists.
- Output file: `/workspace/tempo_candidates.json`
- The output must be a JSON array of exactly 3 objects with the schema:

  ```json
  {
    "tempo_bpm": number,
    "salience": number,
    "harmonic_rank": integer
  }
  ```

  - `tempo_bpm` must be in `[40.0, 240.0]`.
  - `salience` must be a non-negative float; the list must be sorted by `salience` non-increasing.
  - `harmonic_rank` values must be positive integers using each of `1`, `2`, `3` exactly once.
- At least one candidate must lie within ±3 BPM of the reference BPM produced by `librosa.feature.rhythm.tempo(onset_envelope=..., sr=..., aggregate=np.mean)`.
- The top candidate (index 0) must lie within ±5 BPM of the tempo returned by `librosa.beat.beat_track(...)`.

