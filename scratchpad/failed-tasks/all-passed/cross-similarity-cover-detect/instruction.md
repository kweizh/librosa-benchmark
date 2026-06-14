# Cross-Similarity Cover Song Detection

## Background
Decide whether `/workspace/track_b.wav` is a cover (possibly transposed and/or tempo-altered) of `/workspace/track_a.wav` using `librosa` 0.11. Use CENS chroma, cross-similarity, and subsequence DTW to score each of the 12 possible key transpositions and pick the best one.

## Requirements
- Read `/workspace/track_a.wav` and `/workspace/track_b.wav` (mono, 22050 Hz, ~15s each).
- Compute CENS chroma features for both tracks.
- For each integer semitone shift in `[-6, 5]` (12 transpositions covering all pitch classes), circularly roll one chroma along the pitch-class axis.
- For every transposition, build a cross-similarity matrix (`librosa.segment.cross_similarity`) and run subsequence DTW (`librosa.sequence.dtw` with `subseq=True`) over the resulting distance matrix.
- Score each transposition by the minimum accumulated cost on the final DTW row, divided by the warping path length ("normalized_cost").
- Pick the transposition with the lowest normalized cost as `best_transposition_semitones`.
- Decide `is_cover` using the rule: `normalized_cost < 0.6` -> `True`, else `False`.
- Report the time offset (in seconds) of the best subsequence alignment start in track A as `best_offset_seconds`.
- Write the decision to `/workspace/cover_decision.json`.

## Implementation Hints
- Use `librosa.feature.chroma_cens` with default parameters; pass `y` and `sr` as keyword args.
- `numpy.roll` along the pitch-class axis is the standard way to simulate semitone transposition on chroma.
- `librosa.segment.cross_similarity` with `mode='distance'`, `full=True`, and `metric='cosine'` returns a dense distance matrix usable directly as DTW input cost.
- Pass the distance matrix to `librosa.sequence.dtw` as `C=...` together with `subseq=True`; do not also pass `X`/`Y`.
- Convert frame indices back to seconds with `librosa.frames_to_time(..., sr=sr, hop_length=hop_length)` using the same hop length you used when computing chroma (CENS default is 512).
- Sanity-check API signatures against the librosa 0.11.0 docs; most functions are keyword-only.

## Acceptance Criteria
- Project path: /workspace
- Ensure the detection pipeline is executed and the output artifact exists.
- Output file: `/workspace/cover_decision.json`
- The output file must be a single JSON object with this schema:

  ```json
  {
    "is_cover": bool,
    "best_transposition_semitones": int,
    "normalized_cost": float,
    "best_offset_seconds": float
  }
  ```

  - `is_cover` must be a JSON boolean.
  - `best_transposition_semitones` must be an integer in the inclusive range `[-6, 5]`.
  - `normalized_cost` must be a finite, non-negative number.
  - `best_offset_seconds` must be a finite number (`>= 0`).
- The decision rule must be: `is_cover == (normalized_cost < 0.6)`.
- `is_cover` must match the ground truth for the provided fixture audio.

