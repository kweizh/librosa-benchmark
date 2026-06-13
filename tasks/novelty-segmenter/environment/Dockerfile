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
RUN mkdir -p /workspace && chmod 0755 /workspace

# Pre-build the deterministic 3-section input audio at /workspace/input.wav.
# Layout: 5 s trumpet + 5 s nutcracker + 5 s choice (all 22050 Hz mono),
# concatenated with smooth equal-power crossfades shorter than 100 ms.
# Ground-truth structural change-points fall at 5.0 s and 10.0 s.
RUN python3 - <<'PY'
import numpy as np
import librosa
import soundfile as sf

SR = 22050
SECTION_SEC = 5.0
SECTION_LEN = int(round(SECTION_SEC * SR))
CROSSFADE_SEC = 0.05  # 50 ms — well under the 100 ms ceiling
CROSSFADE_LEN = int(round(CROSSFADE_SEC * SR))


def load_section(example_key: str) -> np.ndarray:
    path = librosa.ex(example_key)
    y, _ = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        raise RuntimeError(f"Example {example_key!r} loaded empty audio.")
    # Loop the example if it is shorter than the desired section length,
    # then trim to an exact deterministic length.
    if y.shape[0] < SECTION_LEN:
        n_repeats = int(np.ceil(SECTION_LEN / y.shape[0]))
        y = np.tile(y, n_repeats)
    y = y[:SECTION_LEN].astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = (y / peak) * 0.9
    return y


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Concatenate `a` and `b` with an equal-power crossfade of `n` samples.

    The total output length is exactly `len(a) + len(b) - n`, so the
    structural boundary between `a` and `b` lands at sample `len(a) - n/2`.
    """
    assert n > 0 and n < len(a) and n < len(b)
    t = np.arange(n, dtype=np.float32) / float(n)
    fade_out = np.cos(0.5 * np.pi * t).astype(np.float32)
    fade_in = np.sin(0.5 * np.pi * t).astype(np.float32)
    head = a[: len(a) - n]
    mix = a[len(a) - n:] * fade_out + b[:n] * fade_in
    tail = b[n:]
    return np.concatenate([head, mix, tail]).astype(np.float32)


trumpet = load_section("trumpet")
nutcracker = load_section("nutcracker")
choice = load_section("choice")

# Build the waveform such that the change-points sit exactly on 5.0 s and 10.0 s
# boundaries: pre-pad each subsequent section with `CROSSFADE_LEN / 2` zero samples
# inside the crossfade so the perceptual boundary lands on the section boundary.
half_cf = CROSSFADE_LEN // 2

# Pad the second & third sections at the front and the prior section at the back
# so that after the equal_power_crossfade the perceptual midpoint of the
# crossfade coincides with the 5.0 s and 10.0 s ticks.
pad = np.zeros(half_cf, dtype=np.float32)

a = np.concatenate([trumpet, pad])
b = np.concatenate([pad, nutcracker, pad])
c = np.concatenate([pad, choice])

stage1 = equal_power_crossfade(a, b, CROSSFADE_LEN)
full = equal_power_crossfade(stage1, c, CROSSFADE_LEN)

expected_total = 3 * SECTION_LEN
if abs(full.shape[0] - expected_total) > SR:  # within ~1 s tolerance
    raise RuntimeError(
        f"Unexpected output length: got {full.shape[0]} samples, expected ~{expected_total}."
    )

peak = float(np.max(np.abs(full)))
if peak > 1.0:
    full = (full / peak) * 0.99

sf.write("/workspace/input.wav", full, SR, subtype="PCM_16")

duration = full.shape[0] / SR
print(f"Wrote /workspace/input.wav: {full.shape[0]} samples, {duration:.4f} s, sr={SR}")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 environment with a deterministic 3-section /workspace/input.wav for Foote-style novelty segmentation."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - additional apt packages: libsndfile1, ffmpeg (audio I/O backends)
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa==0.11.0 plus numpy, scipy, scikit-learn, soxr, soundfile, pooch
#   - /workspace: working directory for the task
#   - /workspace/input.wav: deterministic 3-section input audio (trumpet/nutcracker/choice, 5 s each)
#     with equal-power crossfades < 100 ms; ground-truth change-points at 5.0 s and 10.0 s
