FROM ubuntu:24.04

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
        soundfile \
        "soxr>=0.3.2" \
        pooch \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /workspace && chmod 0755 /workspace

# Bake a deterministic 120 BPM click track at /workspace/input.wav.
# Parameters: 22050 Hz, 10 s duration, unit impulses every 0.5 s (120 BPM),
# plus a tiny seeded Gaussian noise floor so onset strength is well-defined.
RUN python3 - <<'PY'
import numpy as np
import soundfile as sf

sr = 22050
duration_sec = 10.0
bpm = 120.0
interval_sec = 60.0 / bpm  # 0.5 s

n_samples = int(round(sr * duration_sec))
rng = np.random.default_rng(42)
audio = rng.normal(loc=0.0, scale=1e-4, size=n_samples).astype(np.float32)

click_times = np.arange(0.0, duration_sec, interval_sec)
click_samples = np.round(click_times * sr).astype(int)
click_samples = click_samples[click_samples < n_samples]

# Short windowed click (a few samples of energy) for robust onset detection.
click_kernel = np.array([0.3, 0.7, 1.0, 0.7, 0.3], dtype=np.float32)
half = len(click_kernel) // 2
for idx in click_samples:
    start = max(0, idx - half)
    end = min(n_samples, idx + half + 1)
    k_start = start - (idx - half)
    k_end = k_start + (end - start)
    audio[start:end] += click_kernel[k_start:k_end]

# Normalize to a safe peak well below clipping.
peak = float(np.max(np.abs(audio)))
if peak > 0:
    audio = (audio / peak) * 0.8

sf.write("/workspace/input.wav", audio, sr, subtype="PCM_16")
print("Wrote /workspace/input.wav", audio.shape, sr)
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="Librosa 0.11.0 PLP tempo & beat tracking task environment with a baked deterministic 120 BPM click track at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - extra apt packages: libsndfile1 ffmpeg (audio I/O backends for soundfile/librosa)
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa==0.11.0 plus numpy, scipy, scikit-learn, soundfile, soxr, pooch
#   - /workspace: working directory used by the task
#   - /workspace/input.wav: baked deterministic 120 BPM click track (10 s, 22050 Hz, 16-bit PCM)
