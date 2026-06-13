FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "librosa==0.11.0" \
        "soundfile>=0.12.1" \
        "numpy>=1.26" \
        "scipy>=1.11" \
        "scikit-learn>=1.4" \
        "soxr>=0.3.2" \
        pooch && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject
RUN mkdir -p /workspace && chmod 0755 /workspace

# Bake the trumpet example audio into /workspace/input.wav so the task is
# fully reproducible and does not depend on network access at runtime.
RUN python3 - <<'PY'
import shutil

import librosa
import soundfile as sf

# Download (cache) the trumpet example and load at its native sample rate.
path = librosa.ex("trumpet")
y, sr = librosa.load(path, sr=None, mono=True)
sf.write("/workspace/input.wav", y, sr, subtype="PCM_16")
print(f"Wrote /workspace/input.wav: {len(y)} samples at {sr} Hz")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment with a baked-in trumpet WAV at /workspace/input.wav for NMF-based source separation tasks."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - additional apt packages: libsndfile1 ffmpeg (audio I/O backends for soundfile/librosa)
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 with soundfile, numpy, scipy, scikit-learn, soxr, pooch
#   - /home/user: project home directory
#   - /home/user/myproject: project directory
#   - /workspace: working directory containing the deterministic input.wav
#   - /workspace/input.wav: baked-in mono trumpet sample (librosa.ex('trumpet'))
