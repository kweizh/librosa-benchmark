FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        ffmpeg libsndfile1 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        numpy \
        scipy \
        scikit-learn \
        "soxr>=0.3.2" \
        soundfile \
        pooch \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

# Prepare /workspace with a synthetic ~10s mono broadband signal at 22050 Hz.
# The signal mixes a log-frequency sweep across the full audible range with
# band-limited noise so that all three transforms (STFT/CQT/VQT) have
# non-trivial energy across their respective frequency grids.
RUN mkdir -p /workspace && \
    python3 - <<'PY' && \
    test -s /workspace/input.wav
import numpy as np
import soundfile as sf

sr = 22050
duration = 10.0
n = int(sr * duration)
t = np.arange(n, dtype=np.float64) / sr

# Logarithmic sweep from 55 Hz to ~sr/2 over the full clip.
f0, f1 = 55.0, sr * 0.45
k = (f1 / f0) ** (1.0 / duration)
phase = 2.0 * np.pi * f0 * (k ** t - 1.0) / np.log(k)
sweep = 0.6 * np.sin(phase)

# Add a few stationary partials so the CQT/VQT bins light up.
partials = sum(
    0.15 * np.sin(2.0 * np.pi * f * t)
    for f in (110.0, 220.0, 440.0, 880.0, 1760.0)
)

# Broadband noise for high-frequency STFT support.
rng = np.random.default_rng(20240601)
noise = 0.05 * rng.standard_normal(n)

y = (sweep + partials + noise).astype(np.float32)
peak = float(np.max(np.abs(y)))
if peak > 0:
    y = (0.9 * y / peak).astype(np.float32)

sf.write('/workspace/input.wav', y, sr, subtype='PCM_16')
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 multi-resolution spectrogram task environment with a synthetic broadband WAV at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: synthetic mono 22050 Hz ~10s broadband signal (log sweep + partials + noise)
#   - /home/user: standard user home directory
