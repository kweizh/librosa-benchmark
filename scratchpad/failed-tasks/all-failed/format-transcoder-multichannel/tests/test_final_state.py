import json
import math
import os

import numpy as np
import pytest


WORKSPACE = "/workspace"
INPUT_FLAC = os.path.join(WORKSPACE, "input.flac")
OUTPUT_WAV = os.path.join(WORKSPACE, "output.wav")
META_JSON = os.path.join(WORKSPACE, "transcode_meta.json")

REQUIRED_META_KEYS = {
    "orig_sample_rate",
    "orig_channels",
    "orig_duration_seconds",
    "output_sample_rate",
    "output_channels",
    "output_duration_seconds",
    "peak_dbfs",
    "rms_dbfs",
    "resampler_backend",
}


def test_soxr_is_installed_in_verifier_env():
    try:
        import soxr  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(
            f"soxr must be installed so the verifier can confirm the soxr_hq backend is usable: {exc}"
        )


@pytest.fixture(scope="module")
def meta():
    assert os.path.isfile(META_JSON), (
        f"Expected metadata file {META_JSON} to exist after the task completes."
    )
    with open(META_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{META_JSON} is not valid JSON: {exc}")
    assert isinstance(data, dict), (
        f"{META_JSON} must be a JSON object, got: {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def output_info():
    import soundfile as sf

    assert os.path.isfile(OUTPUT_WAV), (
        f"Expected output audio file {OUTPUT_WAV} to exist after the task completes."
    )
    info = sf.info(OUTPUT_WAV)
    return info


@pytest.fixture(scope="module")
def output_samples():
    import soundfile as sf

    samples, sr = sf.read(OUTPUT_WAV, dtype="float32", always_2d=False)
    return samples, sr


@pytest.fixture(scope="module")
def input_info():
    import soundfile as sf

    info = sf.info(INPUT_FLAC)
    return info


def test_output_wav_samplerate_is_16000(output_info):
    assert output_info.samplerate == 16000, (
        f"Expected output sample rate 16000, got {output_info.samplerate}."
    )


def test_output_wav_is_mono(output_info):
    assert output_info.channels == 1, (
        f"Expected mono output (1 channel), got {output_info.channels} channels."
    )


def test_output_wav_subtype_is_pcm_16(output_info):
    assert "PCM_16" in (output_info.subtype or ""), (
        f"Expected output subtype to contain 'PCM_16', got {output_info.subtype!r}."
    )


def test_meta_has_required_keys(meta):
    missing = REQUIRED_META_KEYS - set(meta.keys())
    assert not missing, (
        f"transcode_meta.json is missing required keys: {sorted(missing)}; "
        f"present keys: {sorted(meta.keys())}."
    )


def test_meta_orig_sample_rate(meta):
    assert int(meta["orig_sample_rate"]) == 44100, (
        f"Expected orig_sample_rate == 44100, got {meta['orig_sample_rate']!r}."
    )


def test_meta_orig_channels(meta):
    assert int(meta["orig_channels"]) == 2, (
        f"Expected orig_channels == 2, got {meta['orig_channels']!r}."
    )


def test_meta_output_sample_rate(meta):
    assert int(meta["output_sample_rate"]) == 16000, (
        f"Expected output_sample_rate == 16000, got {meta['output_sample_rate']!r}."
    )


def test_meta_output_channels(meta):
    assert int(meta["output_channels"]) == 1, (
        f"Expected output_channels == 1, got {meta['output_channels']!r}."
    )


def test_meta_resampler_backend(meta):
    assert meta["resampler_backend"] == "soxr_hq", (
        f"Expected resampler_backend == 'soxr_hq', got {meta['resampler_backend']!r}."
    )


def test_meta_output_duration_matches_file(meta, output_info):
    actual_duration = float(output_info.frames) / float(output_info.samplerate)
    reported = float(meta["output_duration_seconds"])
    assert abs(reported - actual_duration) <= 0.005, (
        f"meta['output_duration_seconds']={reported} does not match actual file "
        f"duration {actual_duration} within 0.005 s."
    )


def test_meta_orig_duration_matches_input(meta, input_info):
    actual_duration = float(input_info.frames) / float(input_info.samplerate)
    reported = float(meta["orig_duration_seconds"])
    assert abs(reported - actual_duration) <= 0.005, (
        f"meta['orig_duration_seconds']={reported} does not match actual input "
        f"duration {actual_duration} within 0.005 s."
    )


def test_meta_durations_close(meta):
    diff = abs(float(meta["orig_duration_seconds"]) - float(meta["output_duration_seconds"]))
    assert diff < 0.05, (
        f"|orig_duration_seconds - output_duration_seconds| must be < 0.05 s, got {diff}."
    )


def test_output_peak_is_near_minus_1_dbfs(output_samples):
    samples, _ = output_samples
    samples = np.asarray(samples)
    assert samples.size > 0, "Output WAV contains no samples."
    peak = float(np.max(np.abs(samples)))
    assert peak > 0.0, "Output WAV peak is zero; expected a peak-normalized signal."
    actual_peak_dbfs = 20.0 * math.log10(peak)
    assert abs(actual_peak_dbfs - (-1.0)) <= 0.2, (
        f"Measured peak of output.wav is {actual_peak_dbfs:.4f} dBFS, expected within "
        f"0.2 dB of -1.0 dBFS."
    )


def test_meta_peak_dbfs_near_minus_1(meta):
    reported = float(meta["peak_dbfs"])
    assert abs(reported - (-1.0)) <= 0.2, (
        f"meta['peak_dbfs']={reported} is not within 0.2 dB of -1.0 dBFS."
    )


def test_meta_peak_dbfs_matches_measured(meta, output_samples):
    samples, _ = output_samples
    samples = np.asarray(samples)
    peak = float(np.max(np.abs(samples)))
    measured = 20.0 * math.log10(peak)
    reported = float(meta["peak_dbfs"])
    assert abs(reported - measured) <= 0.2, (
        f"meta['peak_dbfs']={reported} does not match measured peak {measured:.4f} dBFS "
        f"within 0.2 dB."
    )


def test_output_not_identical_to_scipy_resample_poly(output_samples):
    """The output must use the soxr_hq backend, not scipy.signal.resample_poly."""
    import soundfile as sf
    from scipy.signal import resample_poly

    y_stereo, sr_in = sf.read(INPUT_FLAC, dtype="float32", always_2d=True)
    assert sr_in == 44100, f"Input sample rate is unexpectedly {sr_in}."
    assert y_stereo.shape[1] == 2, (
        f"Input is expected to be stereo, got {y_stereo.shape[1]} channels."
    )
    # Weighted (equal-weight) downmix to mono — the same downmix the agent is asked to perform.
    mono = (y_stereo[:, 0] + y_stereo[:, 1]) / 2.0
    # Naive baseline resample with scipy.signal.resample_poly (sr_in=44100 -> 16000 => up=160, down=441).
    y_poly = resample_poly(mono, 160, 441).astype(np.float32)

    y_out, sr_out = output_samples
    y_out = np.asarray(y_out, dtype=np.float32)
    assert sr_out == 16000, f"Output sample rate is unexpectedly {sr_out}."

    # Peak-normalize the baseline to -1 dBFS so amplitude differences alone don't bias the check.
    target_peak = 10.0 ** (-1.0 / 20.0)
    poly_peak = float(np.max(np.abs(y_poly)))
    if poly_peak > 0:
        y_poly = y_poly * (target_peak / poly_peak)

    # Align lengths for a sample-wise comparison.
    n = min(len(y_out), len(y_poly))
    assert n > 0, "Cannot compare empty signals."
    diff = np.abs(y_out[:n] - y_poly[:n])
    mean_abs_diff = float(np.mean(diff))

    assert not np.array_equal(y_out[:n], y_poly[:n]), (
        "Output is byte-identical to a scipy.signal.resample_poly result; the soxr_hq backend was not used."
    )
    # Two different bandlimited resamplers should differ by far more than numerical noise.
    assert mean_abs_diff > 1e-4, (
        f"Output is suspiciously close to a scipy.signal.resample_poly baseline "
        f"(mean |diff| = {mean_abs_diff:.2e}); expected the soxr_hq backend to produce a clearly distinct waveform."
    )
