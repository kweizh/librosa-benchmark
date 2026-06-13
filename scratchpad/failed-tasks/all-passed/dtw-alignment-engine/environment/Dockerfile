FROM ubuntu:24.04

# System dependencies, Python, and the librosa stack pinned to v0.11.0.
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

# Standard user directory required by Harbor.
RUN mkdir -p /home/user && chmod 0755 /home/user

# Workspace for the task inputs/outputs.
RUN mkdir -p /workspace && chmod 0755 /workspace

# Prebuild reference.wav and performance.wav from the librosa "pistachio" example.
# reference.wav  = the full pistachio clip resampled to 22050 Hz mono.
# performance.wav = 5 seconds of leading silence + time_stretch(reference, rate=1.05).
# Network access is only required at build time so the runtime container is fully offline.
RUN python3 - <<'PYEOF'
import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 22050
OFFSET_SECONDS = 5.0
STRETCH_RATE = 1.05

src_path = librosa.ex('pistachio')
y_ref, sr = librosa.load(src_path, sr=TARGET_SR, mono=True)
assert sr == TARGET_SR, f"Unexpected sample rate {sr}"
assert y_ref.size > 0, "Empty pistachio example"

# Reference: full clip.
sf.write('/workspace/reference.wav', y_ref, sr, subtype='PCM_16')

# Performance: time-stretch then prepend silence.
y_stretched = librosa.effects.time_stretch(y=y_ref, rate=STRETCH_RATE)
n_silence = int(round(OFFSET_SECONDS * sr))
silence = np.zeros(n_silence, dtype=y_stretched.dtype)
y_perf = np.concatenate([silence, y_stretched])
sf.write('/workspace/performance.wav', y_perf, sr, subtype='PCM_16')

ref_dur = y_ref.shape[-1] / sr
perf_dur = y_perf.shape[-1] / sr
print(f"reference duration = {ref_dur:.3f}s, performance duration = {perf_dur:.3f}s")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment for the headless DTW audio alignment engine task. /workspace contains reference.wav and performance.wav generated from the pistachio example."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa==0.11.0 plus its scientific stack (numpy, scipy, scikit-learn, soxr, pooch, soundfile)
#   - libsndfile1, ffmpeg system libs for robust audio I/O
#   - /workspace/reference.wav: full pistachio example clip at 22050 Hz mono
#   - /workspace/performance.wav: same clip preceded by 5.0 s of silence and time-stretched by 1.05x
