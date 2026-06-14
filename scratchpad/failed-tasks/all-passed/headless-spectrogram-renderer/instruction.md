# Headless Multi-Panel Spectrogram Figure

## Background
Produce a publication-quality 4-panel PNG figure from a single WAV file using `librosa` (0.11) and `matplotlib`, in a strictly headless environment.

## Requirements
- Read `/workspace/input.wav`.
- Render a single PNG figure with exactly 4 stacked panels that share the x-axis (time in seconds):
  - Panel A: time-domain waveform via `librosa.display.waveshow`.
  - Panel B: linear-frequency STFT magnitude spectrogram in dB via `librosa.display.specshow`.
  - Panel C: log-frequency mel spectrogram in dB.
  - Panel D: chromagram from `librosa.feature.chroma_cqt`.
- Panels B, C, and D must each carry their own colorbar.
- Each panel must carry a title, drawn from the fixed list in Acceptance Criteria, in that exact order.
- Force matplotlib into the non-interactive `Agg` backend before any `pyplot` import. Do not call `plt.show()`.
- Write the figure to `/workspace/figure.png` and accompanying metadata to `/workspace/figure_meta.json`.

## Implementation Hints
- Use `librosa.load` to read the WAV.
- Use `librosa.stft` with `librosa.amplitude_to_db` for Panel B.
- Use `librosa.feature.melspectrogram` with `librosa.power_to_db` for Panel C; render with `y_axis='mel'` for log-frequency display.
- Use `librosa.feature.chroma_cqt` for Panel D; render with `y_axis='chroma'`.
- Set `matplotlib.use('Agg')` BEFORE `import matplotlib.pyplot`.
- Save with `fig.savefig(...)` using an explicit `dpi`.
- Read pixel dimensions and dpi back from the saved file and record them in the metadata JSON.

## Acceptance Criteria
- Project path: /workspace
- Ensure the script is executed and both artifacts exist.
- Output files:
  - `/workspace/figure.png`
  - `/workspace/figure_meta.json`
- `figure.png` must be a valid PNG of at least 50000 bytes that opens with PIL.
- The image must be at least 800x800 pixels and have a DPI of at least 100.
- The image must contain real visual structure (not a uniform blank image).
- `figure_meta.json` must be a JSON object with exactly these keys:
  - `width_px` (integer): the PNG pixel width.
  - `height_px` (integer): the PNG pixel height.
  - `dpi` (float): the DPI used to save the figure.
  - `panel_titles` (array of 4 strings) equal to exactly: `["Waveform", "Linear STFT (dB)", "Log-Mel Spectrogram (dB)", "Chromagram"]`.
- The `width_px` and `height_px` fields must match the actual pixel dimensions of `figure.png`.

