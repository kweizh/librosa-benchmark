# Shazam-style Constellation Fingerprint Matcher

## Background
Implement a Shazam-style audio fingerprinter with `librosa` 0.11.0. A long reference recording and a shorter noisy query clip (cut from a hidden offset inside the reference) are provided. Recover the time offset at which the query starts inside the reference using a constellation map of spectral peaks and combinatorial hashing of peak pairs.

## Inputs
- `/workspace/reference.wav` (mono, 22050 Hz)
- `/workspace/query.wav` (mono, 22050 Hz) — derived from the reference with additive Gaussian noise.

## Requirements
- Read both WAV files with `librosa.load`.
- Compute a log-magnitude STFT for each signal (`librosa.stft`, `librosa.amplitude_to_db`).
- Build a constellation map of local spectral peaks and hash anchor/target pairs.
- Match query hashes against reference hashes and recover the dominant time offset.
- Write the result to `/workspace/match.json`.
- Note: all relevant `librosa` 0.11 functions used here (`load`, `stft`, `amplitude_to_db`, `frames_to_time`) use keyword-only arguments after the first positional one.

## Acceptance Criteria
- Project path: /workspace
- Ensure the matching pipeline is executed and the output artifact exists.
- Output file: `/workspace/match.json`
- The output file must be a JSON object containing exactly these four keys:

  ```json
  {
    "offset_seconds": number,
    "match_score": integer,
    "reference_hash_count": integer,
    "query_hash_count": integer
  }
  ```

  - `offset_seconds` is a float: the recovered time offset (seconds) where the query starts inside the reference.
  - `match_score` is an integer: the number of hash votes that fall in the winning offset bin.
  - `reference_hash_count` and `query_hash_count` are integers: total hashes generated from each signal.
- The recovered `offset_seconds` must match the hidden ground-truth offset within 0.30 seconds.
- `match_score` must be at least 20.
- `reference_hash_count > query_hash_count > 0`.
- `offset_seconds` must lie inside `[0, reference_duration - query_duration + 0.5]`.

