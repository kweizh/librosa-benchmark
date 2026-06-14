# Probabilistic YIN (pYIN) Pitch Tracking

## Background
Extract a per-frame fundamental-frequency (F0) contour from a monophonic audio clip using `librosa`. The contour must include voicing decisions and per-frame voicing probability and be serialized as JSON with one record per analysis frame.

## Requirements
- Read the input WAV file from `/workspace/input.wav`.
- Run pYIN-based F0 estimation over the note range `C2` to `C7`.
- Emit one record per analysis frame to `/workspace/pitch_track.json` with timestamps in seconds, the F0 value in Hertz (or JSON `null` when unvoiced), the voicing flag, and the voicing probability.

## Implementation Hints
- Use `librosa.pyin` with `fmin` and `fmax` derived from `librosa.note_to_hz` for the required note range; rely on its default `hop_length` and `frame_length`.
- Convert per-frame indices to seconds using a librosa helper that is consistent with the pYIN hop length.
- Non-finite F0 values returned by pYIN represent unvoiced frames; they must be serialized as JSON `null`, not strings or `NaN`.
- Voicing probability is a separate output of pYIN and must be carried into the JSON as a plain float.
- Sanity-check function signatures, keyword-only arguments, and return-tuple ordering against the librosa 0.11.0 documentation.

## Acceptance Criteria
- Project path: /workspace
- Ensure the pitch-tracking pipeline is executed and the output artifact exists.
- Output file: `/workspace/pitch_track.json`
- The output file must be a JSON array of objects, one per analysis frame, with the following schema:

  ```json
  {
    "time": number,
    "f0_hz": number | null,
    "voiced": boolean,
    "voiced_prob": number
  }
  ```

- The number of records must equal the number of frames returned by `librosa.pyin` for this input.
- `time` values must be strictly monotonically increasing across the list and the final `time` must be within 0.1 seconds of the audio duration.
- For every record, `voiced_prob` must lie in the closed interval `[0.0, 1.0]`.
- For every record where `voiced` is `false`, `f0_hz` must be JSON `null`.
- For every record where `voiced` is `true`, `f0_hz` must be a finite number in the closed interval `[librosa.note_to_hz('C2'), librosa.note_to_hz('C7')]`.
- At least 30% of records must have `voiced` equal to `true`.
- At least 5% of records must have `voiced` equal to `false`.

