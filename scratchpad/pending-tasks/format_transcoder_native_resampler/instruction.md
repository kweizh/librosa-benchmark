Audio processing pipelines often require standardizing formats and sampling rates before model inference to ensure data consistency.

You need to write a script `transcode.py` that loads a compressed MP3 file `input.mp3` at its native sample rate, converts it to mono, resamples it to exactly 16000 Hz using the `soxr_hq` backend, and exports it as a 16-bit PCM WAV file `output.wav` in a headless environment.

**Constraints:**
- Do NOT rely on librosa's default resampling; you must use `sr=None` when loading the initial audio to preserve the native sample rate.
- You must use `soundfile.write` for exporting the audio, as the `librosa.output` module has been removed.
- Use explicit keyword arguments for librosa functions (e.g., `y=y`, `sr=sr`) to comply with v0.11.0 strict signature enforcement.