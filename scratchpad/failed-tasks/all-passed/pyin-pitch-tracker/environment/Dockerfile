FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "numpy<2" \
        scipy \
        scikit-learn \
        soundfile \
        "soxr>=0.3.2" \
        pooch \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

# Synthesize the deterministic sine-sweep input.wav at /workspace/input.wav.
# Sweep: linear 200 Hz -> 800 Hz over 5 seconds at 22050 Hz, mono, 16-bit PCM.
# A fixed numpy seed is used so any auxiliary jitter remains reproducible.
RUN mkdir -p /workspace && \
    python3 - <<'PY'
import numpy as np
import soundfile as sf

np.random.seed(0)

sr = 22050
duration = 5.0
f_start = 200.0
f_end = 800.0

n_samples = int(round(duration * sr))
t = np.arange(n_samples, dtype=np.float64) / sr

# Phase of a linear sweep: phi(t) = 2*pi*(f0*t + 0.5*k*t^2), where k = (f1-f0)/T.
k = (f_end - f_start) / duration
phase = 2.0 * np.pi * (f_start * t + 0.5 * k * t * t)
signal = 0.8 * np.sin(phase)

# 16-bit PCM mono WAV.
sf.write("/workspace/input.wav", signal.astype(np.float32), sr, subtype="PCM_16")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment with a deterministic 200-800 Hz sine sweep at /workspace/input.wav for pYIN pitch tracking."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio backend apt packages: libsndfile1 ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa==0.11.0 plus numpy<2, scipy, scikit-learn, soundfile, soxr, pooch
#   - /workspace/input.wav: deterministic 5-second 200 Hz -> 800 Hz linear sine sweep at 22050 Hz
