FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        numpy \
        "scipy>=1.10" \
        scikit-learn \
        "soxr>=0.3.2" \
        pooch \
        soundfile \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /workspace && chmod 0755 /workspace

# Synthesize the dry signal (mono trumpet at 22050 Hz) and a synthetic
# exponentially-decaying random impulse response at 44100 Hz so that the
# agent must align sample rates before convolving.
RUN cat >/tmp/synthesize_inputs.py <<'PYEOF'
import os

import numpy as np
import soundfile as sf

import librosa

WORKSPACE = "/workspace"
os.makedirs(WORKSPACE, exist_ok=True)

# --- Dry signal: trumpet example, mono, 22050 Hz ---
dry_sr = 22050
trumpet_path = librosa.ex("trumpet")
dry, sr_loaded = librosa.load(trumpet_path, sr=dry_sr, mono=True)
assert sr_loaded == dry_sr, f"Expected dry sr {dry_sr}, got {sr_loaded}"
# Keep amplitude modest so convolution does not clip pre-normalization.
peak = float(np.max(np.abs(dry))) or 1.0
dry = (dry / peak) * 0.7
sf.write(os.path.join(WORKSPACE, "dry.wav"), dry.astype(np.float32), dry_sr, subtype="FLOAT")

# --- Impulse response: exponential-decay random noise, 0.5s, 44100 Hz ---
ir_sr = 44100
ir_duration = 0.5
n_ir = int(round(ir_sr * ir_duration))

rng = np.random.default_rng(seed=20240601)
noise = rng.standard_normal(n_ir).astype(np.float64)

t = np.arange(n_ir, dtype=np.float64) / ir_sr
# Exponential decay with a ~120 ms time constant; strong early reflection at t=0.
decay = np.exp(-t / 0.12)
ir = noise * decay

# Boost the very first sample slightly so the IR has an impulse-like onset
# (causality check in the verifier expects a strong t=0 component).
ir[0] = max(abs(ir[0]), 0.8) * np.sign(ir[0] if ir[0] != 0 else 1.0)

peak = float(np.max(np.abs(ir))) or 1.0
ir = (ir / peak) * 0.9

sf.write(os.path.join(WORKSPACE, "ir.wav"), ir.astype(np.float32), ir_sr, subtype="FLOAT")

print(
    f"Wrote dry.wav: {len(dry)} samples @ {dry_sr} Hz; "
    f"ir.wav: {len(ir)} samples @ {ir_sr} Hz"
)
PYEOF

RUN python3 /tmp/synthesize_inputs.py && rm -f /tmp/synthesize_inputs.py

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment with pre-synthesized dry vocal-like signal (22050 Hz) and exponentially-decaying random impulse response (44100 Hz) for an FFT convolution reverb task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa 0.11.0 + numpy/scipy/scikit-learn/soxr/pooch/soundfile for audio processing
#   - libsndfile1 / ffmpeg system libraries for soundfile and librosa I/O
#   - /workspace/dry.wav: mono trumpet example signal at 22050 Hz (peak ~= 0.7)
#   - /workspace/ir.wav: exponentially-decaying random impulse response, 0.5 s at 44100 Hz (peak ~= 0.9)
