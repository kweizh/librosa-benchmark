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

# Prepare /workspace with a deterministic synthetic mono 22050 Hz audio file
# that contains an explicit A/B/A multi-section structure. Each section is a
# distinct triad rendered as a sum of sinusoids with a soft amplitude envelope,
# so the resulting waveform exposes both repetition (A reused) and contrast
# (B differs). The synthesis is fully deterministic across builds.
RUN mkdir -p /workspace /tmp/gen && \
    printf '%s\n' \
        'import numpy as np' \
        'import soundfile as sf' \
        'sr = 22050' \
        'section_duration = 11.0' \
        'sections = [' \
        '    [261.63, 329.63, 392.00],  # A: C major triad' \
        '    [349.23, 440.00, 523.25],  # B: F major triad' \
        '    [261.63, 329.63, 392.00],  # A: C major triad reprise' \
        ']' \
        'n = int(round(section_duration * sr))' \
        'fade = int(round(0.05 * sr))' \
        'fade_in = np.linspace(0.0, 1.0, fade, endpoint=True).astype(np.float32)' \
        'fade_out = np.linspace(1.0, 0.0, fade, endpoint=True).astype(np.float32)' \
        't = np.arange(n, dtype=np.float64) / sr' \
        'pieces = []' \
        'for freqs in sections:' \
        '    wave = np.zeros_like(t)' \
        '    for f in freqs:' \
        '        wave += np.sin(2.0 * np.pi * f * t)' \
        '    wave /= max(1.0, float(len(freqs)))' \
        '    envelope = 0.6 + 0.4 * np.sin(2.0 * np.pi * 1.5 * t)' \
        '    wave = (wave * envelope).astype(np.float32)' \
        '    wave[:fade] *= fade_in' \
        '    wave[-fade:] *= fade_out' \
        '    pieces.append(wave)' \
        'audio = np.concatenate(pieces, axis=0)' \
        'peak = float(np.max(np.abs(audio)))' \
        'if peak > 0.0:' \
        '    audio = (audio / peak * 0.85).astype(np.float32)' \
        'sf.write("/workspace/input.wav", audio, sr, subtype="PCM_16")' \
        > /tmp/gen/build_input.py && \
    python3 /tmp/gen/build_input.py && \
    test -s /workspace/input.wav && \
    rm -rf /tmp/gen

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 Laplacian structural segmentation task environment with a deterministic A/B/A synthetic WAV at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: deterministic ~33s mono 22050 Hz A/B/A synthetic music
#   - /home/user: standard user home directory
