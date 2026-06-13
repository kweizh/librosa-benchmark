FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        librosa==0.11.0 \
        soundfile \
        numpy \
        scipy \
        scikit-learn \
        "soxr>=0.3.2" \
        pooch && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /workspace && chmod 0755 /workspace

# Pre-generate deterministic 22050 Hz mono target and source clips from two
# distinct librosa example recordings. Both are loaded with the same default
# 'kaiser_best' resampler and cast to float32 so the verifier can reproduce
# them deterministically. Two distinct examples are used so that the source
# is timbrally different from the target (a baseline against which the
# mosaic must demonstrate improved similarity).
RUN python3 - <<'PY'
import os
import numpy as np
import librosa
import soundfile as sf

os.makedirs("/workspace", exist_ok=True)

# Target: solo trumpet (tonal, single-instrument).
y_target, sr_target = librosa.load(librosa.ex("trumpet"), sr=22050, mono=True)
# Source: orchestral nutcracker excerpt (rich, varied timbres providing a
# broad palette of MFCC grains for the mosaic to choose from).
y_source, sr_source = librosa.load(librosa.ex("nutcracker"), sr=22050, mono=True)

# Normalize peaks to predictable amplitudes so the agent's output stays in
# the [0.05, 1.0] range without surprises.
def _normalize(y, peak=0.9):
    m = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    if m <= 0:
        return y.astype(np.float32)
    return (y * (peak / m)).astype(np.float32)

y_target = _normalize(y_target, peak=0.9)
y_source = _normalize(y_source, peak=0.9)

sf.write("/workspace/target.wav", y_target, sr_target, subtype="PCM_16")
sf.write("/workspace/source.wav", y_source, sr_source, subtype="PCM_16")

print("target:", y_target.shape, sr_target)
print("source:", y_source.shape, sr_source)
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment with deterministic 22050 Hz mono target and source clips for the audio mosaicing engine task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio support: libsndfile1, ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa 0.11.0 with soundfile, numpy, scipy, scikit-learn, soxr, pooch
#   - /workspace/target.wav: deterministic 22050 Hz mono target derived from librosa.ex('trumpet')
#   - /workspace/source.wav: deterministic 22050 Hz mono source derived from librosa.ex('nutcracker')
