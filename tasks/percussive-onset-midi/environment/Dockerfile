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

# Bake a deterministic ~12s percussive drum loop at 120 BPM into /workspace/input.wav
# so the task environment has no runtime network dependency. The loop is rendered
# with kick, snare, and hi-hat impulses on a 16th-note grid (steady tempo) using
# numpy + soundfile only. The synth script is materialized via printf so the
# Dockerfile stays compatible with the standard Docker parser (no heredocs).
RUN mkdir -p /workspace && \
    printf '%s\n' \
        'import numpy as np' \
        'import soundfile as sf' \
        'rng = np.random.default_rng(20240601)' \
        'sr = 22050' \
        'tempo_bpm = 120.0' \
        'duration_s = 12.0' \
        'step_s = 60.0 / tempo_bpm / 4.0' \
        'n_samples = int(round(sr * duration_s))' \
        'mix = np.zeros(n_samples, dtype=np.float64)' \
        'def add_sample(start_s, sample):' \
        '    start = int(round(start_s * sr))' \
        '    end = min(start + len(sample), len(mix))' \
        '    if start < len(mix):' \
        '        mix[start:end] += sample[: end - start]' \
        'def kick(amp=1.0):' \
        '    t = np.arange(int(sr * 0.18)) / sr' \
        '    env = np.exp(-t * 18.0)' \
        '    pitch = 110.0 * np.exp(-t * 35.0) + 45.0' \
        '    phase = 2 * np.pi * np.cumsum(pitch) / sr' \
        '    return amp * env * np.sin(phase)' \
        'def snare(amp=0.9):' \
        '    n2 = int(sr * 0.18)' \
        '    t = np.arange(n2) / sr' \
        '    env = np.exp(-t * 22.0)' \
        '    noise = rng.standard_normal(n2)' \
        '    tone = np.sin(2 * np.pi * 200.0 * t) * np.exp(-t * 30.0)' \
        '    return amp * env * (0.7 * noise + 0.4 * tone)' \
        'def hat(amp=0.45):' \
        '    n2 = int(sr * 0.06)' \
        '    t = np.arange(n2) / sr' \
        '    env = np.exp(-t * 60.0)' \
        '    noise = rng.standard_normal(n2)' \
        '    return amp * env * noise' \
        'kick_steps = {0, 6, 8, 14}' \
        'snare_steps = {4, 12}' \
        'hat_steps = {0, 2, 4, 6, 8, 10, 12, 14}' \
        'total_steps = int(np.floor(duration_s / step_s))' \
        'for step in range(total_steps):' \
        '    pos = step * step_s' \
        '    bar_idx = step % 16' \
        '    if bar_idx in kick_steps:' \
        '        add_sample(pos, kick(1.0))' \
        '    if bar_idx in snare_steps:' \
        '        add_sample(pos, snare(0.95))' \
        '    if bar_idx in hat_steps:' \
        '        amp = 0.5 if bar_idx % 4 == 0 else 0.35' \
        '        add_sample(pos, hat(amp))' \
        'peak = float(np.max(np.abs(mix)))' \
        'if peak > 0:' \
        '    mix = mix / peak * 0.85' \
        'sf.write("/workspace/input.wav", mix.astype(np.float32), sr, subtype="PCM_16")' \
        > /tmp/gen_drum_loop.py && \
    python3 /tmp/gen_drum_loop.py && \
    rm /tmp/gen_drum_loop.py && \
    test -s /workspace/input.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 percussive onset / grid quantizer task environment with a pre-baked 12s ~120 BPM drum loop at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: synthesized 12s drum loop at 120 BPM, 22050 Hz mono
#   - /home/user: standard user home directory
