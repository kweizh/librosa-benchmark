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

# Prepare /workspace with a synthetic monophonic melody at /workspace/input.wav.
# The melody is 7 sine-tone notes (mono, 22050 Hz, ~5.95 s total) spanning a
# range from MIDI 60 (C4) to MIDI 76 (E5), separated by short silences. Each
# note is shaped with a short attack/release envelope to avoid clicks. The
# wider-than-one-octave range and brief inter-note silences make the audio
# well-suited to a pYIN-based melody extraction task.
RUN mkdir -p /workspace && \
    python3 -c "import numpy as np, soundfile as sf; \
sr=22050; \
midi_notes=[60, 64, 67, 72, 76, 67, 60]; \
note_dur=0.70; gap_dur=0.15; \
freqs=[440.0*(2.0**((m-69)/12.0)) for m in midi_notes]; \
n_note=int(round(note_dur*sr)); n_gap=int(round(gap_dur*sr)); \
t=np.arange(n_note)/sr; \
env=np.ones(n_note); fade=int(0.02*sr); \
env[:fade]=np.linspace(0.0,1.0,fade); env[-fade:]=np.linspace(1.0,0.0,fade); \
gap=np.zeros(n_gap); \
chunks=[]; \
[chunks.extend([0.6*np.sin(2*np.pi*f*t)*env, gap]) for f in freqs]; \
y=np.concatenate(chunks).astype(np.float32); \
sf.write('/workspace/input.wav', y, sr, subtype='PCM_16')" && \
    test -s /workspace/input.wav

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 F0/melody extraction task environment with a pre-baked synthetic monophonic WAV at /workspace/input.wav."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/input.wav: synthetic monophonic 7-note melody (mono, 22050 Hz, ~5.95 s)
#   - /home/user: standard user home directory
