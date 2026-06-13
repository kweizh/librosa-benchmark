import os

import numpy as np
import pytest
import soundfile as sf


WORKSPACE = "/workspace"
INPUT_PATH = os.path.join(WORKSPACE, "input.wav")
OUTPUT_PATH = os.path.join(WORKSPACE, "output.wav")

PEAK_MIN = 0.05
PEAK_MAX = 1.0
TEMPO_MAX_RATIO = 0.95
SIMILARITY_MAX = 0.95


def _read_wav_mono(path):
    """Read a WAV file as a 1-D mono float64 waveform and return (y, sr)."""
    data, sr = sf.read(path, always_2d=True)
    data = np.asarray(data, dtype=np.float64)
    # soundfile returns shape (n_samples, channels); average across channels.
    mono = data.mean(axis=1)
    return mono, sr


def _crop_or_pad(arr, target_len):
    """Crop or zero-pad a 1-D array along its last axis to exactly target_len."""
    cur = arr.shape[-1]
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[..., :target_len]
    pad = np.zeros(target_len - cur, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.fixture(scope="module")
def input_audio():
    assert os.path.isfile(INPUT_PATH), f"Input file {INPUT_PATH} is missing."
    y, sr = _read_wav_mono(INPUT_PATH)
    return y, sr


@pytest.fixture(scope="module")
def output_audio():
    assert os.path.isfile(OUTPUT_PATH), (
        f"Output file {OUTPUT_PATH} was not created."
    )
    y, sr = _read_wav_mono(OUTPUT_PATH)
    return y, sr


def test_output_file_exists_and_readable():
    assert os.path.isfile(OUTPUT_PATH), (
        f"Expected output file at {OUTPUT_PATH}."
    )
    try:
        sf.read(OUTPUT_PATH)
    except Exception as exc:
        pytest.fail(f"Output WAV {OUTPUT_PATH} is not readable: {exc}")


def test_output_sample_rate_matches_input(input_audio, output_audio):
    _, sr_in = input_audio
    _, sr_out = output_audio
    assert sr_out == sr_in, (
        f"Output sample rate {sr_out} does not match input sample rate {sr_in}."
    )


def test_output_length_not_exceeding_input(input_audio, output_audio):
    in_y, _ = input_audio
    out_y, _ = output_audio
    n_in = in_y.shape[-1]
    n_out = out_y.shape[-1]
    assert n_out <= n_in, (
        f"Output length {n_out} samples exceeds input length {n_in} samples; "
        f"the trim step should make the output no longer than the input."
    )
    assert n_out > 0, "Output waveform is empty."


def test_output_peak_amplitude_in_range(output_audio):
    out_y, _ = output_audio
    peak = float(np.max(np.abs(out_y)))
    assert PEAK_MIN <= peak <= PEAK_MAX, (
        f"Output peak amplitude {peak:.4f} is outside the allowed "
        f"range [{PEAK_MIN}, {PEAK_MAX}]."
    )


def test_output_spectral_centroid_is_higher_than_input(input_audio, output_audio):
    import librosa

    in_y, sr_in = input_audio
    out_y, sr_out = output_audio

    centroid_in = librosa.feature.spectral_centroid(
        y=in_y.astype(np.float32), sr=sr_in
    )
    centroid_out = librosa.feature.spectral_centroid(
        y=out_y.astype(np.float32), sr=sr_out
    )
    mean_in = float(np.mean(centroid_in))
    mean_out = float(np.mean(centroid_out))
    assert mean_out > mean_in, (
        f"Output mean spectral centroid {mean_out:.2f} Hz is not strictly "
        f"greater than the input mean spectral centroid {mean_in:.2f} Hz; "
        f"the harmonic component should have been pitch-shifted up."
    )


def test_output_tempo_is_slower_than_input(input_audio, output_audio):
    import librosa

    in_y, sr_in = input_audio
    out_y, sr_out = output_audio

    tempo_in_arr = librosa.feature.tempo(y=in_y.astype(np.float32), sr=sr_in)
    tempo_out_arr = librosa.feature.tempo(y=out_y.astype(np.float32), sr=sr_out)
    tempo_in = float(np.asarray(tempo_in_arr).ravel()[0])
    tempo_out = float(np.asarray(tempo_out_arr).ravel()[0])
    assert tempo_in > 0, f"Input tempo estimate {tempo_in} BPM is non-positive."
    ratio = tempo_out / tempo_in
    assert ratio <= TEMPO_MAX_RATIO, (
        f"Output tempo {tempo_out:.2f} BPM is {ratio:.3f}x the input tempo "
        f"{tempo_in:.2f} BPM; expected the output tempo to be at most "
        f"{TEMPO_MAX_RATIO}x the input tempo because the percussive component "
        f"was time-stretched to 0.85x."
    )


def test_output_is_not_identical_to_input(input_audio, output_audio):
    in_y, _ = input_audio
    out_y, _ = output_audio
    aligned = _crop_or_pad(out_y, in_y.shape[-1])
    cos_sim = _cosine_similarity(aligned, in_y)
    assert cos_sim < SIMILARITY_MAX, (
        f"Output is too similar to the input (cosine similarity = "
        f"{cos_sim:.4f}, expected < {SIMILARITY_MAX}); the executor likely "
        f"did not actually transform the audio."
    )
