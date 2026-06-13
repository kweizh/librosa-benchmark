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
        "soxr>=0.3.2" \
        soundfile \
        audioread \
        pooch \
        matplotlib \
        Pillow \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

RUN mkdir -p /workspace && chmod 0755 /workspace

# Bake /workspace/input.wav from the librosa 'nutcracker' example so the task
# environment is fully self-contained and the initial state test passes.
RUN python3 -c "import os, shutil, soundfile as sf, librosa; \
src = librosa.ex('nutcracker'); \
y, sr = librosa.load(src, sr=22050, mono=True, duration=30.0); \
sf.write('/workspace/input.wav', y, sr, subtype='PCM_16'); \
print('Wrote /workspace/input.wav at', sr, 'Hz, samples=', y.shape[0])"

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment for the beat-synchronous CQT visualization task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio backends: libsndfile1, ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa 0.11.0 plus numpy<2, scipy, scikit-learn, soxr, soundfile, audioread, pooch, matplotlib, Pillow
#   - /workspace: working directory containing input.wav baked from librosa.ex('nutcracker') at 22050 Hz mono, 30s clip
