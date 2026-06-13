# Monophonic Onset-Based MIDI Note Transcriber

## Background
Build a small offline monophonic note transcriber on top of librosa. The transcriber reads a short audio file containing a sequence of synthesized notes, detects where each note begins, estimates its pitch, quantizes the pitch to the nearest MIDI integer, and assigns a velocity proportional to the loudness of each note. The result is written to a JSON file describing every detected note.

## Requirements
- Read the input audio from `/workspace/input.wav`.
- Detect note onsets using `librosa.onset.onset_detect` with `backtrack=True` so each onset is snapped back to the local energy minimum.
- For every per-note segment, estimate the fundamental frequency with `librosa.pyin` and use the median of the voiced f0 values as the segment's pitch.
- Convert each segment's pitch from Hz to a MIDI integer using `librosa.hz_to_midi`.
- Derive a per-note integer velocity in `[1, 127]` from the per-note RMS energy.
- Write the resulting list of notes to `/workspace/notes.json`.

## Acceptance Criteria
- Project path: /workspace
- Ensure the script is executed and the artifact `/workspace/notes.json` exists.
- Output file: /workspace/notes.json
- The file must be valid JSON containing a single JSON array. Each element of the array must be an object with exactly these four keys and types:
  - `onset_sec` (float, seconds from the start of the audio)
  - `offset_sec` (float, seconds from the start of the audio, strictly greater than `onset_sec`)
  - `pitch_midi` (integer in `[0, 127]`)
  - `velocity` (integer in `[1, 127]`)
- The notes must be sorted strictly in ascending order by `onset_sec`.
- Every `offset_sec` must be no greater than the audio duration plus a small tolerance of 0.1 s.

