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

# Prepare /workspace with a synthetic mixture: 10 seconds of mono audio at 22050 Hz
# containing four sinusoidal sources at non-overlapping frequency bands (220, 660,
# 1320, 2640 Hz) gated with overlapping but distinct time envelopes. This produces
# a mixture amenable to 4-component NMF decomposition while still requiring the
# agent to recover individual sources via soft-masking and iSTFT resynthesis.
RUN mkdir -p /workspace && python3 - <<'PY'
import numpy as np
import soundfile as sf

sr = 22050
duration = 10.0
N = int(sr * duration)
t = np.arange(N, dtype=np.float64) / sr


def envelope(start_s: float, end_s: float, ramp_s: float = 0.05) -> np.ndarray:
    e = np.zeros(N, dtype=np.float64)
    s = max(0, int(round(start_s * sr)))
    f = min(N, int(round(end_s * sr)))
    if f <= s:
        return e
    e[s:f] = 1.0
    r = int(round(ramp_s * sr))
    if r > 0 and (f - s) > 2 * r:
        e[s:s + r] *= np.linspace(0.0, 1.0, r, endpoint=False)
        e[f - r:f] *= np.linspace(1.0, 0.0, r, endpoint=False)
    return e


sources = [
    (220.0, 0.0, 6.0),
    (660.0, 2.0, 8.0),
    (1320.0, 4.0, 10.0),
    (2640.0, 1.0, 9.0),
]

mixture = np.zeros(N, dtype=np.float64)
for freq, start_s, end_s in sources:
    mixture += 0.25 * np.sin(2.0 * np.pi * freq * t) * envelope(start_s, end_s)

peak = float(np.max(np.abs(mixture)))
if peak > 0.95:
    mixture *= 0.95 / peak

sf.write("/workspace/mixture.wav", mixture.astype(np.float32), sr, subtype="PCM_16")
PY

RUN test -s /workspace/mixture.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 NMF source decomposition task environment with a pre-baked synthetic 4-source mixture WAV at /workspace/mixture.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/mixture.wav: synthetic 4-source mixture (10s, 22050 Hz mono, PCM_16)
#   - /home/user: standard user home directory
