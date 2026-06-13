FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "numpy<2.1" \
        scipy \
        scikit-learn \
        "soxr>=0.3.2" \
        pooch \
        soundfile \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /workspace && chmod 0755 /workspace

# Generate a deterministic mono 22050 Hz WAV at /workspace/input.wav
# using the canonical `librosa.ex('trumpet')` example clip.
RUN python3 - <<'PY'
import librosa
import soundfile as sf

src = librosa.ex('trumpet')
y, sr = librosa.load(src, sr=22050, mono=True)
sf.write('/workspace/input.wav', y, sr, subtype='PCM_16')
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment for the streaming STFT/RMS analyzer task; ships a deterministic mono 22050 Hz trumpet WAV at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - libsndfile1 for soundfile-based audio IO
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa 0.11.0 plus numpy/scipy/scikit-learn/soxr/pooch/soundfile for streaming audio analysis
#   - /home/user: empty home directory
#   - /workspace/input.wav: deterministic mono 22050 Hz trumpet sample for the streaming RMS task
