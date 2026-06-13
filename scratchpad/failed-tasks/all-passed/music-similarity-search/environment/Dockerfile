FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep \
        libsndfile1 ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        numpy==1.26.4 \
        scipy==1.13.1 \
        scikit-learn==1.5.2 \
        "soxr>=0.3.2" \
        soundfile==0.12.1 \
        pooch==1.8.2 \
        librosa==0.11.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user

# Build the deterministic music library and the noisy query clip.
# We pull six distinct librosa example tracks at fixed offsets/durations and
# bake the ground-truth match name into /workspace/.ground_truth_match.
RUN mkdir -p /workspace/library && \
    python3 - <<'PYEOF'
import os
import numpy as np
import librosa
import soundfile as sf

WORKSPACE = "/workspace"
LIBRARY_DIR = os.path.join(WORKSPACE, "library")
os.makedirs(LIBRARY_DIR, exist_ok=True)

TARGET_SR = 22050

# (example_name, offset_seconds, max_duration_seconds, output_basename)
TRACKS = [
    ("nutcracker", 0.0, 15.0, "nutcracker.wav"),
    ("pistachio",  2.0, 15.0, "pistachio.wav"),
    ("brahms",     1.0, 12.0, "brahms.wav"),
    ("sweetwaltz", 0.5, 12.0, "sweetwaltz.wav"),
    ("fishin",     0.0,  6.0, "fishin.wav"),
    ("choice",     0.0,  6.0, "choice.wav"),
]

# Map of basename -> waveform for ground-truth construction.
loaded = {}
for example_name, offset, duration, out_name in TRACKS:
    src = librosa.ex(example_name)
    y, sr = librosa.load(src, sr=TARGET_SR, mono=True, offset=offset, duration=duration)
    if y.size == 0:
        # Fallback: load whole file if the offset exceeds file length.
        y, sr = librosa.load(src, sr=TARGET_SR, mono=True)
    out_path = os.path.join(LIBRARY_DIR, out_name)
    sf.write(out_path, y.astype(np.float32), TARGET_SR, subtype="PCM_16")
    loaded[out_name] = y.astype(np.float32)
    print(f"wrote {out_path} ({y.shape[0]} samples)")

# Choose the ground-truth match track.
GT_NAME = "nutcracker.wav"
gt_audio = loaded[GT_NAME]

# Take a 5-second snippet from the centre of the chosen track.
snippet_seconds = 5.0
snippet_len = int(snippet_seconds * TARGET_SR)
if gt_audio.shape[0] < snippet_len:
    snippet = np.pad(gt_audio, (0, snippet_len - gt_audio.shape[0]))
else:
    start = (gt_audio.shape[0] - snippet_len) // 2
    snippet = gt_audio[start:start + snippet_len]

snippet = snippet.astype(np.float32)

# Mix in white Gaussian noise at exactly 15 dB SNR with a fixed seed.
rng = np.random.RandomState(20260613)
signal_power = float(np.mean(snippet.astype(np.float64) ** 2))
if signal_power <= 0.0:
    raise RuntimeError("Ground-truth snippet has zero signal power.")
snr_db = 15.0
noise_power = signal_power / (10.0 ** (snr_db / 10.0))
noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power),
                   size=snippet.shape).astype(np.float32)
query = snippet + noise

# Avoid hard clipping while preserving SNR characteristics.
peak = float(np.max(np.abs(query)))
if peak > 0.999:
    query = (query / peak) * 0.99
query = query.astype(np.float32)

sf.write(os.path.join(WORKSPACE, "query.wav"),
         query, TARGET_SR, subtype="PCM_16")

with open(os.path.join(WORKSPACE, ".ground_truth_match"), "w") as f:
    f.write(GT_NAME + "\n")

print("ground truth:", GT_NAME)
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="librosa 0.11.0 music similarity search task: pre-built /workspace/library WAVs, a noisy /workspace/query.wav, and a baked ground-truth match label."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - audio system libs: libsndfile1, ffmpeg
#   - always required pip3 packages: pytest-json-ctrf
#   - librosa==0.11.0 plus numpy/scipy/scikit-learn/soundfile/soxr/pooch for feature extraction
#   - /workspace/library/: 6 distinct librosa example tracks saved as PCM_16 WAV
#   - /workspace/query.wav: 5 s snippet of one library track mixed with white noise at 15 dB SNR
#   - /workspace/.ground_truth_match: basename of the library track from which the query was derived
