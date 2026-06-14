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

# Prepare /workspace with a rhythmic ~15s clip at 22050 Hz mono. The clip is
# derived from a librosa example (Brahms - Hungarian Dance #5, distributed via
# the librosa example pack and downloaded by pooch). The example is fetched
# once at build time so the task environment has no runtime network dependency.
RUN mkdir -p /workspace && \
    python3 -c "import librosa, soundfile as sf; y, sr = librosa.load(librosa.ex('choice'), sr=22050, mono=True, duration=15.0); sf.write('/workspace/input.wav', y, sr, subtype='PCM_16')" && \
    test -s /workspace/input.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 beat-synchronous MFCC task environment with a pre-baked rhythmic ~15s WAV at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: rhythmic ~15s mono 22050 Hz clip baked from librosa.ex('choice')
#   - /home/user: standard user home directory
