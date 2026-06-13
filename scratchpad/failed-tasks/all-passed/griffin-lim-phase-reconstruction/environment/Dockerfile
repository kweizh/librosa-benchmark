FROM ubuntu:24.04

# System dependencies and Python tooling.
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        numpy \
        scipy \
        scikit-learn \
        "soxr>=0.3.2" \
        pooch \
        soundfile \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

# Workspace and deterministic input generation.
RUN mkdir -p /workspace && chmod 0755 /workspace

RUN python3 - <<'PY'
import os
import numpy as np
import librosa
import soundfile as sf  # noqa: F401  (ensures soundfile is importable in build)

TARGET_SR = 22050
N_FFT = 1024
HOP_LENGTH = 256

# Deterministic reference signal: librosa's trumpet example, mono, 22050 Hz.
audio_path = librosa.ex('trumpet')
y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
y = np.asarray(y, dtype=np.float32)

# Normalize to a stable peak amplitude so downstream amplitude checks are
# deterministic and well within the allowed [0.01, 1.0] range.
peak = float(np.max(np.abs(y)))
if peak > 0:
    y = (y * (0.9 / peak)).astype(np.float32)

# Compute magnitude STFT with known parameters.
stft = librosa.stft(
    y,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=N_FFT,
    window='hann',
    center=True,
    pad_mode='constant',
)
magnitude = np.abs(stft).astype(np.float32)

os.makedirs('/workspace', exist_ok=True)
np.save('/workspace/magnitude.npy', magnitude)

with open('/workspace/reference_length.txt', 'w') as f:
    f.write(str(int(y.shape[0])))

print(f"magnitude shape={magnitude.shape}, reference length={y.shape[0]} samples")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="Librosa 0.11.0 environment with a precomputed magnitude STFT of a deterministic reference signal for Griffin-Lim phase reconstruction."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio I/O system libs: libsndfile1, ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa==0.11.0 with soundfile, numpy, scipy, scikit-learn, soxr, pooch
#   - /workspace/magnitude.npy: float32 magnitude STFT (n_fft=1024, hop_length=256) of librosa.ex('trumpet') resampled to 22050 Hz mono
#   - /workspace/reference_length.txt: integer length of the original reference signal in samples
