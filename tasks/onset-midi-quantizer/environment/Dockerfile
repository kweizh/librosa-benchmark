FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "numpy<2" \
        scipy \
        scikit-learn \
        soundfile \
        "soxr>=0.3.2" \
        pooch \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

# Synthesize a deterministic 5-note monophonic melody at 22050 Hz.
# Notes (MIDI): 60, 64, 67, 72, 67  (C4, E4, G4, C5, G4)
# Each note: 1.0 second sine wave with a 0.05 s linear fade-in and 0.05 s linear fade-out.
# Notes are separated by 0.2 s of silence. Total duration ~ 5.8 s.
# A fixed numpy seed is used so the generated waveform is fully reproducible.
RUN mkdir -p /workspace && \
    python3 - <<'PY'
import numpy as np
import soundfile as sf

np.random.seed(0)

sr = 22050
note_midi = [60, 64, 67, 72, 67]
note_dur = 1.0       # seconds per note
gap_dur = 0.2        # seconds of silence between notes
fade_dur = 0.05      # seconds of linear fade-in / fade-out
amplitude = 0.8

def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def make_note(freq, duration_sec, sr, amp, fade_sec):
    n = int(round(duration_sec * sr))
    t = np.arange(n, dtype=np.float64) / sr
    wave = amp * np.sin(2.0 * np.pi * freq * t)
    fade_n = max(1, int(round(fade_sec * sr)))
    envelope = np.ones(n, dtype=np.float64)
    if fade_n * 2 < n:
        ramp = np.linspace(0.0, 1.0, fade_n, endpoint=False)
        envelope[:fade_n] = ramp
        envelope[-fade_n:] = ramp[::-1]
    return wave * envelope

silence = np.zeros(int(round(gap_dur * sr)), dtype=np.float64)
segments = []
for i, m in enumerate(note_midi):
    if i > 0:
        segments.append(silence)
    segments.append(make_note(midi_to_hz(m), note_dur, sr, amplitude, fade_dur))

signal = np.concatenate(segments).astype(np.float32)

sf.write("/workspace/input.wav", signal, sr, subtype="PCM_16")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment with a deterministic 5-note synthesized melody at /workspace/input.wav for onset-based MIDI note transcription."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio backend apt packages: libsndfile1 ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa==0.11.0 plus numpy<2, scipy, scikit-learn, soundfile, soxr, pooch
#   - /workspace/input.wav: deterministic 5-note melody (MIDI 60,64,67,72,67) at 22050 Hz,
#     each note 1.0 s with 0.05 s fade-in/out separated by 0.2 s silence.
