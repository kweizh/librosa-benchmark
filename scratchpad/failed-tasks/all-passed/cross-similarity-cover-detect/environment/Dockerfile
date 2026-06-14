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

# Prepare /workspace with two fixture audio files:
#   - /workspace/track_a.wav: ~15s of tonal music (Tchaikovsky's "Dance of the
#     Sugar Plum Fairy" from the Nutcracker Suite, distributed by the librosa
#     example pack via pooch).
#   - /workspace/track_b.wav: a transposed (+3 semitones) and tempo-altered
#     (x1.10) copy of track_a, i.e. a true cover.
#   - /workspace/ground_truth.json: {"is_cover": true}, the truth used by the
#     verifier. The agent must NOT read this file; it is only documented in the
#     verification plan.
RUN mkdir -p /workspace && \
    python3 - <<'PY'
import json
import os

import numpy as np
import soundfile as sf
import librosa

SR = 22050
DUR = 15.0
PITCH_SHIFT_SEMITONES = 3
TIME_STRETCH_RATE = 1.10

src = librosa.ex("nutcracker")
y, _ = librosa.load(src, sr=SR, mono=True, duration=DUR)
# Pad to exactly DUR seconds if shorter
target_len = int(SR * DUR)
if y.shape[0] < target_len:
    y = np.pad(y, (0, target_len - y.shape[0]))
else:
    y = y[:target_len]

# track_a: the original 15s clip at 22050 Hz mono
sf.write("/workspace/track_a.wav", y.astype(np.float32), SR, subtype="PCM_16")

# track_b: pitch-shifted +3 semitones, then time-stretched by 1.10
y_shift = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=PITCH_SHIFT_SEMITONES)
y_stretch = librosa.effects.time_stretch(y=y_shift, rate=TIME_STRETCH_RATE)
# Normalize peak to avoid clipping after effects
peak = float(np.max(np.abs(y_stretch))) or 1.0
y_stretch = (y_stretch / peak * 0.95).astype(np.float32)
sf.write("/workspace/track_b.wav", y_stretch, SR, subtype="PCM_16")

with open("/workspace/ground_truth.json", "w") as fh:
    json.dump({"is_cover": True}, fh)

# Sanity-check the fixtures actually loaded and are the expected lengths.
ya, sra = librosa.load("/workspace/track_a.wav", sr=None, mono=True)
yb, srb = librosa.load("/workspace/track_b.wav", sr=None, mono=True)
assert sra == SR and srb == SR, (sra, srb)
assert ya.size > 0 and yb.size > 0
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 cover-song detection task environment with baked track_a.wav / track_b.wav fixtures and a verifier ground_truth.json."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/track_a.wav: tonal 15s mono @22050Hz clip baked from librosa.ex('nutcracker')
#   - /workspace/track_b.wav: pitch-shifted (+3 semitones) and time-stretched (x1.10) copy of track_a (a true cover)
#   - /workspace/ground_truth.json: {"is_cover": true} used by the verifier
#   - /home/user: standard user home directory
