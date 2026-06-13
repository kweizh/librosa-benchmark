FROM ubuntu:24.04

ENV LIBROSA_DATA_DIR=/opt/librosa-data/audio

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
RUN mkdir -p /workspace && chmod 0755 /workspace

# Pre-download librosa example data so the build is reproducible without
# relying on network access at run time.
RUN git clone --depth 1 https://github.com/librosa/data.git /opt/librosa-data

# Build the database.wav (full nutcracker example, ~17s, 22050 Hz mono PCM_16)
# and the noisy query.wav (5-second clip starting at offset=5.0s of database.wav
# mixed with white Gaussian noise at SNR ~10 dB using a fixed seed).
RUN python3 - <<'PY'
import os
os.environ.setdefault("LIBROSA_DATA_DIR", "/opt/librosa-data/audio")

import numpy as np
import soundfile as sf
import librosa

SR = 22050
OFFSET_SECONDS = 5.0
QUERY_DURATION_SECONDS = 5.0
TARGET_SNR_DB = 10.0
SEED = 42

src_path = librosa.example("nutcracker")
y, sr = librosa.load(src_path, sr=SR, mono=True)
assert sr == SR

# database.wav: the full clean nutcracker example.
os.makedirs("/workspace", exist_ok=True)
sf.write("/workspace/database.wav", y, sr, subtype="PCM_16")

# Extract the clean query slice from a known offset inside the database.
start = int(round(OFFSET_SECONDS * sr))
end = start + int(round(QUERY_DURATION_SECONDS * sr))
assert end <= y.shape[0], (
    f"database too short for query extraction: end={end} > len(y)={y.shape[0]}"
)
query_clean = y[start:end].astype(np.float32)

# Add deterministic white Gaussian noise at SNR ~10 dB.
rng = np.random.default_rng(SEED)
noise = rng.standard_normal(query_clean.shape).astype(np.float32)
signal_power = float(np.mean(query_clean.astype(np.float64) ** 2))
noise_power = float(np.mean(noise.astype(np.float64) ** 2))
if signal_power <= 0 or noise_power <= 0:
    raise RuntimeError("Unexpected zero-power signal or noise.")

desired_noise_power = signal_power / (10.0 ** (TARGET_SNR_DB / 10.0))
noise *= np.sqrt(desired_noise_power / noise_power)
query = query_clean + noise

# Light peak normalization to avoid 16-bit clipping while preserving SNR.
peak = float(np.max(np.abs(query)))
if peak > 0.99:
    query = query * (0.99 / peak)

sf.write("/workspace/query.wav", query.astype(np.float32), sr, subtype="PCM_16")

print(
    f"Built database.wav (duration={y.shape[0] / sr:.3f}s) and "
    f"query.wav (duration={query.shape[0] / sr:.3f}s, "
    f"ground-truth offset={OFFSET_SECONDS:.3f}s, SNR={TARGET_SNR_DB:.1f} dB)"
)
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment for the constellation-map audio fingerprinter task: includes /workspace/database.wav (nutcracker, ~17s) and /workspace/query.wav (noisy 5s clip extracted at 5.0s)."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep libsndfile1 ffmpeg
#   - pip3: pytest-json-ctrf numpy scipy scikit-learn soxr pooch soundfile librosa==0.11.0
#   - /opt/librosa-data: cloned librosa example-data repository (offline cache via LIBROSA_DATA_DIR)
#   - /workspace/database.wav: full nutcracker clip (22050 Hz mono, ~17s)
#   - /workspace/query.wav: noisy 5s clip extracted from database.wav at offset=5.0s (SNR ~10 dB, seed=42)
