FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
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
RUN mkdir -p /workspace/train/speech /workspace/train/music /workspace/test && \
    chmod -R 0755 /workspace

# Pre-bake the labeled training set and the unlabeled test set into the image.
# We use librosa's built-in example clips:
#   speech: libri1, libri2, libri3 (LibriSpeech narrations)
#   music : trumpet, nutcracker, vibeace (instrumental recordings)
# Each source clip is split into two non-overlapping snippets via clip offsets so
# each class ends up with 4 distinct training files and 2 distinct test files.
# Snippets are written as 16-bit PCM mono WAV files at 22050 Hz.
RUN python3 - <<'PY'
import os
import numpy as np
import librosa
import soundfile as sf

SR = 22050
SUBTYPE = "PCM_16"

def two_snippets(src_path):
    """Load a librosa example and return two non-overlapping snippets."""
    y, _ = librosa.load(src_path, sr=SR, mono=True)
    n = len(y)
    if n < SR:
        raise RuntimeError(f"Source {src_path} too short ({n} samples) for splitting.")
    half = n // 2
    # Leave a tiny gap at the boundary and clamp snippets to <= 5 seconds.
    max_snip = 5 * SR
    snippet_len = min(half - SR // 10, max_snip)
    if snippet_len < SR // 2:
        # If half is very small, fall back to half exactly.
        snippet_len = max(half, SR // 4)
    s1 = y[:snippet_len].astype(np.float32, copy=False)
    s2 = y[half: half + snippet_len].astype(np.float32, copy=False)
    return s1, s2

# Class -> ordered list of source librosa example keys.
SOURCES = {
    "speech": ["libri1", "libri2", "libri3"],
    "music":  ["trumpet", "nutcracker", "vibeace"],
}

# For each class we collect 6 snippets (2 from each of 3 sources), then assign
# the first 4 to the training split and the last 2 to the test split.
TRAIN_DIR = "/workspace/train"
TEST_DIR = "/workspace/test"

for cls, keys in SOURCES.items():
    snippets = []
    for key in keys:
        src = librosa.ex(key)
        a, b = two_snippets(src)
        snippets.append((key, 0, a))
        snippets.append((key, 1, b))

    # 4 training files: first two sources contribute 4 snippets total.
    train_snips = snippets[:4]
    # 2 test files: the third source contributes the remaining 2 snippets.
    test_snips = snippets[4:6]

    cls_train_dir = os.path.join(TRAIN_DIR, cls)
    for i, (_key, _idx, audio) in enumerate(train_snips):
        out_path = os.path.join(cls_train_dir, f"{cls}_train_{i}.wav")
        sf.write(out_path, audio, SR, subtype=SUBTYPE)
        print("wrote", out_path, audio.shape)

    for i, (_key, _idx, audio) in enumerate(test_snips):
        out_path = os.path.join(TEST_DIR, f"{cls}_test_{i}.wav")
        sf.write(out_path, audio, SR, subtype=SUBTYPE)
        print("wrote", out_path, audio.shape)

print("dataset generation complete")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11 environment pre-baked with a small speech/music WAV dataset under /workspace/train and /workspace/test for the spectral-contrast-classifier task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio runtime libs: libsndfile1, ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa 0.11.0 + soundfile, numpy, scipy, scikit-learn, soxr, pooch (feature extraction + classifier deps)
#   - /workspace/train/speech/*.wav : 4 deterministic speech training clips drawn from libri1 / libri2
#   - /workspace/train/music/*.wav  : 4 deterministic music training clips drawn from trumpet / nutcracker
#   - /workspace/test/*.wav         : 4 unlabeled test clips (2 speech from libri3, 2 music from vibeace)
