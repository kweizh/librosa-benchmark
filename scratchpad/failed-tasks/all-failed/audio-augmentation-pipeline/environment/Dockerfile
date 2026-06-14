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

# Prepare /workspace with a ~5s monophonic example audio file. The librosa
# example pack provides a short solo trumpet recording (single melodic voice)
# which is ideal for pYIN-based F0 verification. The waveform is resampled to
# 22050 Hz, padded with brief leading/trailing silence, truncated to ~5 s, and
# written to /workspace/input.wav at build time so the task environment has no
# runtime network dependency.
RUN mkdir -p /workspace && \
    python3 -c "import numpy as np, soundfile as sf, librosa; \
y, sr = librosa.load(librosa.ex('trumpet'), sr=22050, mono=True); \
y = y[: int(4.5 * sr)]; \
pad = np.zeros(int(0.25 * sr), dtype=y.dtype); \
y = np.concatenate([pad, y, pad]); \
sf.write('/workspace/input.wav', y, sr, subtype='PCM_16')" && \
    test -s /workspace/input.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 audio augmentation task environment with a pre-baked ~5s monophonic WAV at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: ~5s monophonic 22050 Hz mono WAV with brief leading/trailing silence
#   - /home/user: standard user home directory
