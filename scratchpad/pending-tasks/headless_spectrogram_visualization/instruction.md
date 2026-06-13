In automated environments like CI/CD pipelines or Docker containers, generating visualizations can crash the system if an interactive display backend is accidentally triggered.

You need to develop a script `visualize.py` that loads `trumpet.wav`, computes its Short-Time Fourier Transform (STFT), converts the magnitude to a decibel-scaled spectrogram using `amplitude_to_db`, and saves a publication-quality log-frequency spectrogram plot directly to disk as `spectrogram.png` in a non-interactive environment.

**Constraints:**
- You must configure `matplotlib` to use the headless `Agg` backend *before* importing `librosa.display` to prevent `TclError`.
- Ensure the y-axis is scaled logarithmically by setting `y_axis='log'` in `specshow`.
- Do NOT call `plt.show()` anywhere in the script.