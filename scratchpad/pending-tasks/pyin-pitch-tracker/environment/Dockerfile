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

# Bake a deterministic monophonic synthetic melody into /workspace/input.wav.
# - mono, 22050 Hz, exactly 8.0 seconds
# - 8 one-second segments: A3, C4, E4, (silence), G4, A4, C5, E5
# - voiced notes use a fundamental + 2nd + 3rd harmonic with short attack/release envelopes
# - one full second of digital silence (~12.5% of the clip) guarantees an unvoiced section
RUN mkdir -p /workspace && \
    python3 - <<'PY' && \
    test -s /workspace/input.wav
import numpy as np
import soundfile as sf

sr = 22050
seg_dur = 1.0
n_samples = int(sr * seg_dur)
t = np.arange(n_samples, dtype=np.float64) / sr

def voiced_note(freq):
    sig = np.sin(2 * np.pi * freq * t)
    sig += 0.5 * np.sin(2 * np.pi * 2.0 * freq * t)
    sig += 0.25 * np.sin(2 * np.pi * 3.0 * freq * t)
    env = np.ones(n_samples, dtype=np.float64)
    ramp = int(0.02 * sr)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    sig *= env
    peak = float(np.max(np.abs(sig)))
    if peak > 0:
        sig *= 0.7 / peak
    return sig

# A3, C4, E4, silence, G4, A4, C5, E5
program = [220.00, 261.63, 329.63, None, 392.00, 440.00, 523.25, 659.25]
parts = []
for freq in program:
    if freq is None:
        parts.append(np.zeros(n_samples, dtype=np.float64))
    else:
        parts.append(voiced_note(freq))

y = np.concatenate(parts).astype(np.float32)
assert y.shape[0] == sr * 8, "synthetic audio must be exactly 8 seconds"
sf.write("/workspace/input.wav", y, sr, subtype="PCM_16")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 pYIN pitch-tracking task environment with a baked synthetic monophonic melody at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: 8s mono 22050 Hz synthetic melody with one 1s silent section
#   - /home/user: standard user home directory
