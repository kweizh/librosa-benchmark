Isolating vocals or rhythmic elements is a common pre-processing step, but handling multi-channel audio requires careful array manipulation to avoid format-writing errors.

You need to write a Python script `hpss_split.py` that performs Harmonic-Percussive Source Separation (HPSS) on a stereo audio file `stereo_input.wav`. Trim leading and trailing silence (using a `top_db=60` threshold) from both separated waveforms, and export them as two separate stereo WAV files: `harmonic.wav` and `percussive.wav` in the local directory.

**Constraints:**
- You MUST handle librosa's channels-first `(2, n_samples)` format by explicitly transposing the arrays to a channels-last `(n_samples, 2)` format before passing them to `soundfile.write` to prevent massive memory allocation errors.
- HPSS parameters must be passed as keyword arguments.