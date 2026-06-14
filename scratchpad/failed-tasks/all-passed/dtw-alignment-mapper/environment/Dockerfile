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

# Prepare /workspace with two related WAV files for the alignment task:
#   - reference.wav: a 15s mono 22050 Hz clip drawn from librosa.ex('nutcracker').
#   - comparison.wav: the same clip time-stretched (faster) via librosa.effects.time_stretch.
# Both files are baked at build time so the runtime has no network dependency.
RUN mkdir -p /workspace && \
    python3 -c "import librosa, soundfile as sf; \
SR = 22050; DURATION = 15.0; RATE = 1.1; \
y, sr = librosa.load(librosa.ex('nutcracker'), sr=SR, mono=True, duration=DURATION); \
sf.write('/workspace/reference.wav', y, sr, subtype='PCM_16'); \
y2 = librosa.effects.time_stretch(y=y, rate=RATE); \
sf.write('/workspace/comparison.wav', y2, sr, subtype='PCM_16')" && \
    test -s /workspace/reference.wav && \
    test -s /workspace/comparison.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 DTW alignment task environment with reference/comparison WAV pair pre-baked into /workspace."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/reference.wav: 15s mono 22050Hz clip from librosa.ex('nutcracker')
#   - /workspace/comparison.wav: same clip time-stretched 1.1x via librosa.effects.time_stretch
#   - /home/user: standard user home directory
