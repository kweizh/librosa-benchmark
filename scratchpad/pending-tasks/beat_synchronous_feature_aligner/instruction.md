Music Information Retrieval (MIR) systems often compress frame-level features into beat-level features to drastically reduce dimensionality and align analytical data with the underlying musical grid.

You need to create a script `beat_sync.py` that estimates the global tempo and locates beat frames for an input audio file `track.wav`. Extract a standard chromagram (using Chroma STFT) and synchronize these chroma features to the detected beat intervals, aggregating the values to produce a beat-synchronous chroma matrix. Save the resulting matrix as a NumPy binary file `beat_chroma.npy` in the execution environment.

**Constraints:**
- Use `librosa.beat.beat_track` to detect the array of beat frames.
- You must use `librosa.util.sync` to structurally align the chromagram frames to the beat frames.
- The aggregation function within `librosa.util.sync` must be explicitly set to calculate the `median`, overriding the default mean behavior.