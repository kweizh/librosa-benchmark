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

# Prepare /workspace with a deterministic reference/query pair derived from the
# tonal librosa example "nutcracker". The reference is the first 25 seconds of
# the example (mono, 22050 Hz). The query is a 5.0-second slice starting at
# 9.7 seconds with additive Gaussian noise at roughly 10 dB SNR (seed=42).
# The hidden ground truth (true offset = 9.7 s) is kept in the verifier-side
# tests directory and is NOT baked into /workspace.
RUN mkdir -p /workspace && \
    python3 - <<'PY'
import numpy as np
import soundfile as sf
import librosa

SR = 22050
REF_DURATION = 25.0
QRY_DURATION = 5.0
TRUE_OFFSET = 9.7
SNR_DB = 10.0
SEED = 42

src_path = librosa.ex('nutcracker')
y_full, sr_native = librosa.load(src_path, sr=SR, mono=True)
assert sr_native == SR, f"Unexpected sample rate after resample: {sr_native}"

n_ref = int(round(REF_DURATION * SR))
assert y_full.size >= n_ref, (
    f"Source example is shorter than required reference duration: "
    f"{y_full.size} < {n_ref}"
)
reference = y_full[:n_ref].astype(np.float32, copy=True)

start_sample = int(round(TRUE_OFFSET * SR))
n_qry = int(round(QRY_DURATION * SR))
clean_query = reference[start_sample:start_sample + n_qry].astype(np.float32, copy=True)
assert clean_query.size == n_qry, (
    f"Clean query slice has wrong size: {clean_query.size} vs {n_qry}"
)

rng = np.random.default_rng(SEED)
signal_power = float(np.mean(clean_query.astype(np.float64) ** 2))
noise_power = signal_power / (10.0 ** (SNR_DB / 10.0))
noise = rng.normal(0.0, np.sqrt(noise_power), size=clean_query.shape).astype(np.float32)
noisy_query = (clean_query + noise).astype(np.float32)

peak = float(np.max(np.abs(noisy_query)))
if peak > 0.99:
    noisy_query = (noisy_query * (0.99 / peak)).astype(np.float32)

sf.write('/workspace/reference.wav', reference, SR, subtype='PCM_16')
sf.write('/workspace/query.wav', noisy_query, SR, subtype='PCM_16')
print('reference duration (s):', reference.size / SR)
print('query duration (s):', noisy_query.size / SR)
print('true offset (s):', TRUE_OFFSET)
PY
RUN test -s /workspace/reference.wav && test -s /workspace/query.wav && \
    test ! -e /workspace/ground_truth.json && \
    test ! -e /workspace/match.json

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 constellation fingerprint matching task environment with pre-baked reference.wav and noisy query.wav under /workspace."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libraries: ffmpeg, libsndfile1
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: librosa==0.11.0 (+ numpy, scipy, scikit-learn, soxr, soundfile, pooch)
#   - /workspace/reference.wav: 25s mono 22050 Hz slice of librosa.ex('nutcracker')
#   - /workspace/query.wav: 5s noisy slice of reference starting at 9.7s (SNR ~10 dB, seed=42)
#   - /home/user: standard user home directory
