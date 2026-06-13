Machine learning classifiers often require fixed-length 1D feature vectors summarizing the global characteristics of an entire audio file.

You need to create a script `extract_features.py` that loads `audio.wav` at a target rate of 22050 Hz and extracts 20 MFCCs, a 128-band Mel Spectrogram, and Spectral Centroids. Compute the temporal mean and standard deviation for each feature matrix along the time axis, concatenate them into a single 1D numpy array, and save it as `features.npy` in the current working directory.

**Constraints:**
- All feature extraction parameters must use strictly keyword-only arguments to comply with librosa v0.11.0.
- The final output must be exactly a 1-Dimensional NumPy array containing the aggregated statistics.