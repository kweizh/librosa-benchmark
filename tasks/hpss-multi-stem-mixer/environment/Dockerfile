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
        "scikit-learn>=1.3" \
        "soxr>=0.3.2" \
        "pooch>=1.7" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /workspace && chmod 0755 /workspace

# Pre-bake a deterministic librosa example as the reproducible /workspace/input.wav.
# The 'choice' clip (from the librosa example collection) is a short music
# excerpt with clear harmonic + percussive content, ideal for HPSS-based
# remixing. We load 20 seconds at 22050 Hz mono and save as 16-bit PCM WAV.
RUN python3 - <<'PY'
import librosa
import soundfile as sf

src = librosa.example('choice')
y, sr = librosa.load(src, sr=22050, mono=True, duration=20.0)
sf.write('/workspace/input.wav', y, sr, subtype='PCM_16')
print('wrote /workspace/input.wav', y.shape, 'sr=', sr)
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11 environment with a pre-baked 'choice' music WAV at /workspace/input.wav for the hpss-multi-stem-mixer task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio runtime libs: libsndfile1, ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa 0.11.0 + soundfile, numpy, scipy, scikit-learn, soxr, pooch (HPSS mixer pipeline deps)
#   - /workspace/input.wav: 20 s 22.05 kHz mono PCM_16 excerpt of librosa example 'choice' to serve as the reproducible task input
