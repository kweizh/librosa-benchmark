FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "numpy>=1.24,<2.3" \
        "scipy>=1.10" \
        "scikit-learn>=1.3" \
        "soxr>=0.3.2" \
        soundfile \
        pooch \
        "librosa==0.11.0" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /workspace && chmod 0755 /workspace

# Pre-bake the deterministic example audio (`librosa.ex('nutcracker')`) into
# /workspace/input.wav. We resample to mono 22050 Hz and write 16-bit PCM so the
# file is portable and the agent never needs network access at run time.
RUN python3 - <<'PY'
import os
import librosa
import soundfile as sf

os.makedirs('/workspace', exist_ok=True)
src = librosa.ex('nutcracker')
y, sr = librosa.load(src, sr=22050, mono=True)
sf.write('/workspace/input.wav', y, sr, subtype='PCM_16')
print(f"Wrote /workspace/input.wav: {len(y)} samples @ {sr} Hz")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 Laplacian-segmentation task environment with a pre-baked /workspace/input.wav example."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio I/O system libraries: libsndfile1, ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - target library + scientific stack: librosa==0.11.0, numpy, scipy, scikit-learn, soxr, soundfile, pooch
#   - /home/user: default user directory
#   - /workspace: agent working directory (contains the pre-baked input.wav)
#   - /workspace/input.wav: deterministic librosa.ex('nutcracker') example audio, 16-bit PCM mono @ 22050 Hz
