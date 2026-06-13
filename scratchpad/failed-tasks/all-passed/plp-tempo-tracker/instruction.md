# Predominant Local Pulse (PLP) Tempo & Beat Tracker

## Background
Build a tempo and beat tracking utility on top of librosa 0.11.0. Rather than relying on the dynamic-programming `librosa.beat.beat_track`, the analysis must use the Predominant Local Pulse (PLP) method, which is preferred when the tempo may vary or when only local context is available. The script must also produce a global tempo estimate from the same audio.

## Requirements
- Read the input WAV file from `/workspace/input.wav`.
- Use `librosa.beat.plp` to compute the predominant local pulse curve.
- Derive beat times in seconds by peak-picking the PLP signal (the beat track must come from the PLP curve, not from `librosa.beat.beat_track`).
- Compute a single global tempo estimate for the recording.
- Write the result as JSON to `/workspace/beats.json` with the schema:
  ```json
  {
    "global_tempo_bpm": <float>,
    "beat_times_sec": [<float>, <float>, ...]
  }
  ```

## Implementation Hints
- Compute the onset strength envelope before calling the PLP estimator so the same envelope can be reused for the global tempo call.
- Local maxima of the PLP signal are the candidate beat positions; convert their frame indices to seconds.
- The global tempo function lives in the `librosa.feature` namespace in 0.11.0.
- Make sure the values written to JSON are plain Python floats (not numpy scalars) so the file is valid JSON.

## Acceptance Criteria
- Project path: /workspace
- Ensure the analysis script is executed and `/workspace/beats.json` exists.
- Output file: /workspace/beats.json
- The JSON document must contain exactly two top-level keys: `global_tempo_bpm` (number) and `beat_times_sec` (array of numbers).
- `global_tempo_bpm` must be a finite number in the range [30, 300].
- `beat_times_sec` must be a non-empty list of strictly increasing numbers, each within `[0, audio_duration + 0.5]` seconds.
- The estimated global tempo must be within 10% of the true tempo of `/workspace/input.wav`.
- The median inter-beat interval implied by `beat_times_sec` must correspond to a tempo within 15% of `global_tempo_bpm`.

