# Multichannel Format Transcoder

## Background
Build a robust audio format transcoder on top of `librosa` 0.11. Consume a stereo FLAC, downmix it to mono, resample with a high-quality bandlimited backend, peak-normalize, and emit a 16-bit PCM WAV plus a metadata JSON. The pipeline must preserve the native sample rate during load and only resample once with the requested backend.

## Requirements
- Read `/workspace/input.flac` at its NATIVE sample rate without resampling at load time, keeping both channels.
- Downmix the stereo signal to mono via a weighted average of the two channels.
- Resample the mono signal to exactly 16000 Hz using the `soxr_hq` backend.
- Peak-normalize the resampled signal to -1 dBFS using `librosa.util.normalize`.
- Write `/workspace/output.wav` as a mono, 16-bit PCM WAV at 16000 Hz via `soundfile`.
- Write `/workspace/transcode_meta.json` describing the original input, the output, and the resampler backend that was used.

## Implementation Hints
- Inspect the librosa 0.11 docs for `librosa.load`, `librosa.resample`, and `librosa.util.normalize` to confirm keyword-only argument names and defaults.
- `librosa.load` returns multichannel audio as a 2D array when `mono=False`; pick the channel axis carefully when downmixing.
- -1 dBFS corresponds to a linear peak of `10**(-1/20)`; pass that as the desired post-normalization peak rather than relying on the function's default behavior.
- `soundfile.write` expects 1D arrays for mono outputs and selects bit depth via the `subtype` argument.
- Record durations as `samples / sample_rate` (seconds, floats) and dB values as `20 * log10(peak)` / `20 * log10(rms)`.

## Acceptance Criteria
- Project path: /workspace
- Ensure the transcoder is executed and the output artifacts exist.
- Output audio: `/workspace/output.wav` (mono, 16000 Hz, 16-bit PCM, loadable by `soundfile`).
- Output metadata: `/workspace/transcode_meta.json` with the following schema:

  ```json
  {
    "orig_sample_rate": number,
    "orig_channels": number,
    "orig_duration_seconds": number,
    "output_sample_rate": 16000,
    "output_channels": 1,
    "output_duration_seconds": number,
    "peak_dbfs": number,
    "rms_dbfs": number,
    "resampler_backend": "soxr_hq"
  }
  ```

  - `orig_sample_rate` must equal the input file's native sample rate.
  - `orig_channels` must equal the input file's channel count.
  - `output_sample_rate` must be `16000` and `output_channels` must be `1`.
  - `output_duration_seconds` must match the actual `/workspace/output.wav` duration within 0.005 s.
  - `|orig_duration_seconds - output_duration_seconds|` must be `< 0.05` s.
  - `peak_dbfs` must be within 0.2 dB of -1 dBFS and must match the measured peak of `/workspace/output.wav`.
  - `resampler_backend` must be the exact string `"soxr_hq"`.
- The `soxr` package must be installed and used; the produced waveform must NOT be byte-identical (or perfectly correlated) with a `scipy.signal.resample_poly`-based resample of the same downmix.

