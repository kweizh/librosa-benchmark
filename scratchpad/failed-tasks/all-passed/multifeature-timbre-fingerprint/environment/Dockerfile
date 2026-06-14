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

# Prepare /workspace with:
#   - input.wav: a deterministically synthesized 8-second mono 22050 Hz mixture
#     of sinusoids, a linear chirp, and shaped noise. This guarantees a tonal,
#     timbrally interesting signal with no runtime network dependency.
#   - centroids.json: three deterministic 72-dim reference centroid vectors
#     keyed by the labels "rock", "classical", and "jazz".
RUN mkdir -p /workspace && \
    python3 - <<'PY'
import json
import numpy as np
import soundfile as sf

sr = 22050
duration = 8.0
n = int(sr * duration)
t = np.arange(n, dtype=np.float64) / sr

rng = np.random.default_rng(20240613)

# Two stationary tones, a slow vibrato, and a linear chirp to spread spectral energy.
tone1 = 0.35 * np.sin(2.0 * np.pi * 220.0 * t)
tone2 = 0.25 * np.sin(2.0 * np.pi * 440.0 * t + 0.5 * np.sin(2.0 * np.pi * 5.0 * t))
tone3 = 0.18 * np.sin(2.0 * np.pi * 660.0 * t)
chirp = 0.20 * np.sin(2.0 * np.pi * (200.0 * t + 0.5 * (800.0 / duration) * t * t))
noise = 0.05 * rng.standard_normal(n)

# Apply a slow amplitude envelope to make per-frame statistics non-trivial.
env = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.25 * t - np.pi / 2.0))
y = (tone1 + tone2 + tone3 + chirp) * (0.5 + 0.5 * env) + noise

# Normalize to avoid clipping while keeping the signal energetic.
peak = float(np.max(np.abs(y)))
if peak > 0.0:
    y = 0.9 * y / peak

sf.write('/workspace/input.wav', y.astype(np.float32), sr, subtype='PCM_16')

centroid_rng = np.random.default_rng(424242)
centroids = {
    label: centroid_rng.standard_normal(72).tolist()
    for label in ('rock', 'classical', 'jazz')
}
with open('/workspace/centroids.json', 'w') as fh:
    json.dump(centroids, fh)
PY

RUN test -s /workspace/input.wav && test -s /workspace/centroids.json

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 multi-feature timbral fingerprint task environment with a synthesized 8s WAV and 3 reference centroids in /workspace."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: deterministic 8s mono 22050 Hz synthesized mixture
#   - /workspace/centroids.json: 3 deterministic 72-dim reference centroids (rock, classical, jazz)
#   - /home/user: standard user home directory
