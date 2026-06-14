# Chord Recognition with Chroma + Viterbi Decoding

## Background
Build a simple major/minor chord recognition pipeline using `librosa`. The pipeline must consume a single audio file, derive chroma observations, score them against 24 chord templates (12 major + 12 minor), and use Hidden Markov Model (HMM)-style Viterbi decoding to produce a temporally smooth chord sequence. The decoded state sequence must then be summarized as time-aligned chord segments.

## Requirements
- Read the input WAV file from `/workspace/input.wav`.
- Detect chords from the set of 24 labels:
  `C:maj, C#:maj, D:maj, D#:maj, E:maj, F:maj, F#:maj, G:maj, G#:maj, A:maj, A#:maj, B:maj, C:min, C#:min, D:min, D#:min, E:min, F:min, F#:min, G:min, G#:min, A:min, A#:min, B:min`.
- Use chroma observations together with `librosa.sequence.viterbi` to decode the most likely chord state per frame.
- Merge consecutive frames that share the same decoded chord into segments and write them to `/workspace/chords.json`.

## Implementation Hints
- Use a librosa chroma feature appropriate for tonal content to obtain a `(12, n_frames)` observation matrix.
- Define 24 chord templates corresponding to major and minor triads (one per root pitch class) and turn them into per-frame, per-state non-negative likelihoods that the Viterbi routine can consume.
- Build a 24x24 transition matrix that favors staying in the current chord (self-bias) while still allowing transitions to any other chord; remember that each row must sum to 1.
- Convert the decoded frame indices to seconds with `librosa.frames_to_time` (using the same `hop_length` used to compute the chroma) and clamp the final segment end to the audio duration so the segments cover the full track without gaps or overlaps.
- Sanity-check the API signatures against the librosa 0.11.0 documentation; most feature and sequence functions are keyword-only.

## Acceptance Criteria
- Project path: /workspace
- Ensure the recognition pipeline is executed and the output artifact exists.
- Output file: `/workspace/chords.json`
- The output file must be a JSON array. Each element must be an object with the following schema:

  ```json
  {
    "start": number,
    "end": number,
    "chord": string
  }
  ```

  - `start` and `end` are floating-point timestamps in seconds with `start < end`.
  - `chord` must match the regular expression `^[A-G]#?:(maj|min)$` and be one of the 24 allowed labels listed above.
- The segments must be sorted by `start`, cover the audio from near 0s to near the audio duration, and not contain overlapping intervals.
- The decoded sequence must contain at least 2 distinct chord labels (the pipeline must produce real variation, not a constant output).

