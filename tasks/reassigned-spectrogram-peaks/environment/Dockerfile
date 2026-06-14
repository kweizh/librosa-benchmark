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

# Prepare /workspace with a deterministic 3-tone harmonic input at /workspace/input.wav.
# The signal is the sum of sine waves at 220 Hz, 440 Hz, and 880 Hz, with low-level
# Gaussian noise (deterministic via numpy.random.default_rng(seed=0)), at 22050 Hz mono,
# 5.0 seconds in duration. This guarantees the reassigned spectrogram will recover
# the planted tones, while the noise floor keeps the task realistic.
RUN mkdir -p /workspace && \
    python3 -c "import numpy as np, soundfile as sf; sr=22050; dur=5.0; t=np.arange(int(sr*dur))/sr; rng=np.random.default_rng(0); y=(0.4*np.sin(2*np.pi*220.0*t)+0.3*np.sin(2*np.pi*440.0*t)+0.2*np.sin(2*np.pi*880.0*t)+0.01*rng.standard_normal(t.shape)).astype('float32'); y=y/np.max(np.abs(y))*0.9; sf.write('/workspace/input.wav', y, sr, subtype='PCM_16')" && \
    test -s /workspace/input.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 reassigned-spectrogram peak-tracking task environment with a deterministic 3-tone harmonic WAV at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: deterministic 22050 Hz mono 5s 3-tone (220/440/880 Hz) + light Gaussian noise PCM_16 WAV
#   - /home/user: standard user home directory
