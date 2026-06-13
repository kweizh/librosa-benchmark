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
        soundfile \
        pooch \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /workspace /workspace/inputs && chmod 0755 /workspace /workspace/inputs

# Generate the deterministic input WAV fixtures used by the task.
# Uses numpy with fixed seeds so the dataset is reproducible across builds.
RUN python3 - <<'PYGEN'
import os

import numpy as np
import soundfile as sf

os.makedirs("/workspace/inputs", exist_ok=True)


def write_wav(path, y, sr):
    """Write float64 (channels-first or 1-D) audio as 16-bit PCM WAV."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 2:
        # channels-first -> channels-last
        arr = arr.T
    # Safety clamp before quantization.
    arr = np.clip(arr, -1.0, 1.0)
    sf.write(path, arr.astype(np.float32), sr, subtype="PCM_16")


# 1) Quiet mono 16 kHz sine tone (very low RMS, no clip risk after boost).
sr = 16000
duration = 2.0
t = np.arange(int(sr * duration)) / sr
tone = 0.01 * np.sin(2.0 * np.pi * 440.0 * t)
write_wav("/workspace/inputs/quiet_mono_16k.wav", tone, sr)

# 2) "Loud" mono 44.1 kHz file with sparse near-full-scale bursts.
# Bursts have high peaks, total RMS is low, so reaching -20 dBFS RMS would
# require positive gain that pushes the peak above 1.0 -> clip-avoid case.
sr = 44100
duration = 2.0
n = int(sr * duration)
burst_signal = np.zeros(n, dtype=np.float64)
burst_freq = 1000.0
burst_len = int(0.005 * sr)  # 5 ms burst
t_burst = np.arange(burst_len) / sr
window = np.hanning(burst_len)
burst = 0.98 * window * np.sin(2.0 * np.pi * burst_freq * t_burst)
# Place a couple of bursts at fixed offsets for determinism.
for offset_sec in (0.10, 0.90):
    start = int(offset_sec * sr)
    end = start + burst_len
    if end <= n:
        burst_signal[start:end] = burst
write_wav("/workspace/inputs/loud_mono_44100.wav", burst_signal, sr)

# 3) Stereo 48 kHz pink-noise clip (fixed seed, moderate RMS, safe headroom).
sr = 48000
duration = 2.0
n = int(sr * duration)
rng = np.random.default_rng(seed=20240613)


def make_pink(n_samples, generator):
    """FFT-based pink-noise generator with 1/sqrt(f) spectral shaping."""
    white = generator.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:] * n_samples)
    shaped = spectrum * scale
    pink = np.fft.irfft(shaped, n=n_samples)
    # Normalize so the RMS sits comfortably below the -20 dBFS target.
    rms = np.sqrt(np.mean(pink * pink))
    if rms > 0:
        pink = pink * (0.056 / rms)  # ~ -25 dBFS RMS
    # Hard clamp far from full scale.
    peak = np.max(np.abs(pink))
    if peak > 0.9:
        pink = pink * (0.9 / peak)
    return pink


left = make_pink(n, rng)
right = make_pink(n, rng)
stereo = np.stack([left, right], axis=0)  # channels-first
write_wav("/workspace/inputs/pink_stereo_48k.wav", stereo, sr)

# 4) Normal-level mono 22050 Hz sine tone (will be attenuated to reach -20 dBFS).
sr = 22050
duration = 2.0
t = np.arange(int(sr * duration)) / sr
tone = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
write_wav("/workspace/inputs/tone_mono_22050.wav", tone, sr)
PYGEN

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment with deterministic WAV fixtures for the RMS loudness normalizer task."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio runtime libs: libsndfile1, ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa 0.11.0 stack: numpy, scipy, scikit-learn, soxr, soundfile, pooch, librosa
#   - /workspace: project root the agent operates in
#   - /workspace/inputs: four deterministic WAV fixtures (quiet 16k mono, sparse-burst loud 44.1k mono, fixed-seed pink stereo 48k, normal-level 22050 mono)
