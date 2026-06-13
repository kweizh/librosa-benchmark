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
        "soxr>=0.3.2" \
        pooch \
        soundfile \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /workspace && chmod 0755 /workspace

# Bake /workspace/input.wav deterministically from librosa.ex('trumpet').
# This calls pooch to fetch the example file during image build so that the
# resulting container does not depend on network access at runtime.
RUN python3 -c "import shutil, librosa; src = librosa.ex('trumpet'); shutil.copyfile(src, '/workspace/input.wav')" && \
    test -s /workspace/input.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 HCQT pitchgram task environment with a baked trumpet example WAV at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system packages: libsndfile1 ffmpeg (for soundfile/audioread fallback)
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 plus numpy, scipy, scikit-learn, soxr, pooch, soundfile
#   - /workspace: working directory
#   - /workspace/input.wav: baked from librosa.ex('trumpet') for deterministic evaluation
