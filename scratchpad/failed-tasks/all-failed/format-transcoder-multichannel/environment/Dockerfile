FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg flac && \
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

# Pre-bake a deterministic stereo FLAC at 44100 Hz, ~6 s, as /workspace/input.flac.
# Channels carry distinct tonal content (different fundamentals + a chirp) so that
# downmixing and resampling cannot trivially collapse to silence.
RUN python3 - <<'PY'
import numpy as np
import soundfile as sf

sr = 44100
duration = 6.0
n = int(sr * duration)
t = np.arange(n, dtype=np.float64) / sr

# Left channel: 220 Hz tone with a slow tremolo.
left = 0.6 * np.sin(2 * np.pi * 220.0 * t) * (0.85 + 0.15 * np.sin(2 * np.pi * 4.0 * t))

# Right channel: 277.18 Hz tone plus a linear chirp from 500 Hz -> 1500 Hz.
chirp_inst_freq = 500.0 + (1500.0 - 500.0) * (t / duration)
chirp_phase = 2 * np.pi * np.cumsum(chirp_inst_freq) / sr
right = 0.55 * np.sin(2 * np.pi * 277.18 * t) + 0.25 * np.sin(chirp_phase)

# soundfile expects (frames, channels) for 2D arrays.
stereo = np.stack([left, right], axis=1).astype(np.float32)

# Keep peaks comfortably below clipping so the downstream peak normalizer has headroom.
peak = float(np.max(np.abs(stereo)))
if peak > 0:
    stereo = (stereo / peak * 0.7).astype(np.float32)

sf.write('/workspace/input.flac', stereo, sr, format='FLAC', subtype='PCM_16')

info = sf.info('/workspace/input.flac')
assert info.samplerate == 44100, info.samplerate
assert info.channels == 2, info.channels
assert info.frames == n, (info.frames, n)
print('wrote /workspace/input.flac', stereo.shape, 'sr=', sr)
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11 environment with a pre-baked stereo 44.1 kHz FLAC at /workspace/input.flac for the format-transcoder-multichannel task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio runtime libs: libsndfile1, ffmpeg, flac
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa 0.11.0 + soundfile, numpy, scipy, scikit-learn, soxr, pooch (transcoder pipeline deps)
#   - /workspace/input.flac: 6 s 44.1 kHz stereo FLAC (PCM_16) with distinct per-channel tonal content
