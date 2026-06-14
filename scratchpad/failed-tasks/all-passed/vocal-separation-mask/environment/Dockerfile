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

# Prepare /workspace with a deterministic, synthetic mono mixture at 22050 Hz
# that contains a clearly distinguishable vocal-like melody and a repetitive
# accompaniment-like chord, so REPET-SIM-style nearest-neighbor median filtering
# can separate them. The mixture is generated once at build time so the task has
# no runtime network or random-seed dependency.
RUN mkdir -p /workspace && python3 - <<'PYEOF'
import numpy as np
import soundfile as sf

sr = 22050
duration = 15.0
n = int(sr * duration)
t = np.arange(n, dtype=np.float64) / sr

# Accompaniment: sustained C major triad (C4, E4, G4) with gentle 2 Hz amplitude
# modulation. Stable harmonic content => spectrally recurrent frames => background.
chord_freqs = [261.63, 329.63, 392.00]
accomp = np.zeros(n, dtype=np.float64)
for f in chord_freqs:
    accomp += np.sin(2 * np.pi * f * t)
accomp *= 0.25 * (0.8 + 0.2 * np.sin(2 * np.pi * 2.0 * t))

# Vocal: a melodic line that changes pitch every ~1.5s with 6 Hz vibrato.
# Non-repetitive across time => foreground for REPET-SIM.
melody_hz = [523.25, 587.33, 659.25, 698.46, 783.99,
             698.46, 659.25, 587.33, 523.25, 659.25]
seg_len = n // len(melody_hz)
vocal = np.zeros(n, dtype=np.float64)
phase = 0.0
for i, f0 in enumerate(melody_hz):
    s = i * seg_len
    e = n if i == len(melody_hz) - 1 else (i + 1) * seg_len
    tt = np.arange(e - s, dtype=np.float64) / sr
    inst = f0 + 4.0 * np.sin(2 * np.pi * 6.0 * tt)
    seg_phase = 2 * np.pi * np.cumsum(inst) / sr + phase
    seg = 0.5 * np.sin(seg_phase)
    env = np.ones_like(seg)
    ramp = min(int(0.02 * sr), len(seg) // 4)
    if ramp > 0:
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    vocal[s:e] = seg * env
    phase = float(seg_phase[-1])

mix = accomp + vocal
peak = float(np.max(np.abs(mix)))
if peak > 0:
    mix = mix / peak * 0.9
sf.write("/workspace/input.wav", mix.astype(np.float32), sr, subtype="PCM_16")
PYEOF

RUN test -s /workspace/input.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 vocal/accompaniment separation task environment with a pre-baked synthetic mixture at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: deterministic synthetic vocal+accompaniment mixture (mono, 22050 Hz, ~15 s)
#   - /home/user: standard user home directory
