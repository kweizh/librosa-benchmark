import json
import math
import os

import pytest


PEAKS_JSON = "/workspace/peaks.json"
INPUT_WAV = "/workspace/input.wav"

TARGET_TONES_HZ = (220.0, 440.0, 880.0)
TONE_TOLERANCE_HZ = 10.0


@pytest.fixture(scope="module")
def payload():
    assert os.path.isfile(PEAKS_JSON), (
        f"Expected output file {PEAKS_JSON} to exist after the task completes."
    )
    with open(PEAKS_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{PEAKS_JSON} is not valid JSON: {exc}")
    assert isinstance(data, dict), (
        f"{PEAKS_JSON} top-level value must be a JSON object, got {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def audio_info():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    duration = float(len(y)) / float(sr)
    assert duration > 0, f"Reference audio duration must be positive, got {duration}."
    return {"sr": int(sr), "duration": duration, "n_samples": int(len(y))}


def test_meta_block_is_valid(payload):
    assert "meta" in payload, "Top-level object must contain a 'meta' field."
    meta = payload["meta"]
    assert isinstance(meta, dict), (
        f"'meta' must be a JSON object, got {type(meta).__name__}."
    )
    for key in ("n_frames", "sr", "n_fft", "hop_length"):
        assert key in meta, f"'meta' is missing required key '{key}': {meta!r}."
        assert isinstance(meta[key], (int, float)) and not isinstance(meta[key], bool), (
            f"'meta.{key}' must be numeric, got {meta[key]!r}."
        )
    assert int(meta["n_fft"]) > 0, f"meta.n_fft must be > 0, got {meta['n_fft']}."
    assert int(meta["hop_length"]) > 0, (
        f"meta.hop_length must be > 0, got {meta['hop_length']}."
    )
    assert int(meta["sr"]) > 0, f"meta.sr must be > 0, got {meta['sr']}."
    assert int(meta["n_frames"]) > 0, (
        f"meta.n_frames must be > 0, got {meta['n_frames']}."
    )


def test_frames_list_matches_meta_n_frames(payload):
    assert "frames" in payload, "Top-level object must contain a 'frames' field."
    frames = payload["frames"]
    assert isinstance(frames, list), (
        f"'frames' must be a JSON list, got {type(frames).__name__}."
    )
    assert len(frames) == int(payload["meta"]["n_frames"]), (
        f"len(frames)={len(frames)} must equal meta.n_frames={payload['meta']['n_frames']}."
    )


def test_meta_sr_matches_input_audio(payload, audio_info):
    sr_meta = float(payload["meta"]["sr"])
    sr_ref = float(audio_info["sr"])
    assert abs(sr_meta - sr_ref) <= 1e-6, (
        f"meta.sr ({sr_meta}) must equal audio sample rate ({sr_ref})."
    )


def test_each_frame_schema(payload):
    frames = payload["frames"]
    for idx, frame in enumerate(frames):
        assert isinstance(frame, dict), (
            f"Frame {idx} must be a JSON object, got {type(frame).__name__}."
        )
        assert "time" in frame and "peaks" in frame, (
            f"Frame {idx} must have 'time' and 'peaks' keys, got keys={list(frame.keys())}."
        )
        assert isinstance(frame["time"], (int, float)) and not isinstance(
            frame["time"], bool
        ), f"Frame {idx} 'time' must be numeric, got {frame['time']!r}."
        assert math.isfinite(float(frame["time"])), (
            f"Frame {idx} 'time' must be finite, got {frame['time']!r}."
        )
        assert isinstance(frame["peaks"], list), (
            f"Frame {idx} 'peaks' must be a list, got {type(frame['peaks']).__name__}."
        )
        assert len(frame["peaks"]) == 5, (
            f"Frame {idx} must contain exactly 5 peaks, got {len(frame['peaks'])}."
        )


def test_time_values_monotonic_and_in_bounds(payload, audio_info):
    frames = payload["frames"]
    duration = audio_info["duration"]
    times = [float(f["time"]) for f in frames]

    for i, t in enumerate(times):
        assert t >= 0.0, f"Frame {i} time must be >= 0, got {t}."
        assert t <= duration + 1e-2, (
            f"Frame {i} time {t} exceeds audio_duration + 1e-2 ({duration + 1e-2})."
        )

    for i in range(len(times) - 1):
        assert times[i + 1] >= times[i] - 1e-9, (
            f"Frame times must be non-decreasing; times[{i}]={times[i]}, "
            f"times[{i+1}]={times[i+1]}."
        )

    last_time = times[-1]
    assert abs(last_time - duration) <= 0.1 + 1e-6, (
        f"Last frame time {last_time} must be within 0.1s of audio duration "
        f"{duration}."
    )


def test_peaks_are_valid_and_sorted(payload, audio_info):
    sr = float(audio_info["sr"])
    nyquist = sr / 2.0
    frames = payload["frames"]

    for f_idx, frame in enumerate(frames):
        peaks = frame["peaks"]
        prev_db = None
        for p_idx, peak in enumerate(peaks):
            assert isinstance(peak, dict), (
                f"Frame {f_idx} peak {p_idx} must be a JSON object."
            )
            assert "freq_hz" in peak and "magnitude_db" in peak, (
                f"Frame {f_idx} peak {p_idx} must have 'freq_hz' and 'magnitude_db'."
            )
            freq = peak["freq_hz"]
            mag = peak["magnitude_db"]
            assert isinstance(freq, (int, float)) and not isinstance(freq, bool), (
                f"Frame {f_idx} peak {p_idx} freq_hz must be numeric, got {freq!r}."
            )
            assert isinstance(mag, (int, float)) and not isinstance(mag, bool), (
                f"Frame {f_idx} peak {p_idx} magnitude_db must be numeric, got {mag!r}."
            )
            freq_f = float(freq)
            mag_f = float(mag)
            assert math.isfinite(freq_f), (
                f"Frame {f_idx} peak {p_idx} freq_hz must be finite, got {freq_f}."
            )
            assert math.isfinite(mag_f), (
                f"Frame {f_idx} peak {p_idx} magnitude_db must be finite, got {mag_f}."
            )
            assert freq_f > 0.0, (
                f"Frame {f_idx} peak {p_idx} freq_hz must be > 0, got {freq_f}."
            )
            assert freq_f <= nyquist + 1e-6, (
                f"Frame {f_idx} peak {p_idx} freq_hz {freq_f} must be <= sr/2 "
                f"({nyquist})."
            )
            if prev_db is not None:
                assert mag_f <= prev_db + 1e-9, (
                    f"Frame {f_idx} peaks must be sorted by magnitude_db descending; "
                    f"peak {p_idx-1} dB={prev_db}, peak {p_idx} dB={mag_f}."
                )
            prev_db = mag_f


def test_majority_of_frames_recover_input_tones(payload):
    frames = payload["frames"]
    hits = 0
    for frame in frames:
        for peak in frame["peaks"]:
            f = float(peak["freq_hz"])
            if any(abs(f - tone) <= TONE_TOLERANCE_HZ for tone in TARGET_TONES_HZ):
                hits += 1
                break
    ratio = hits / float(len(frames))
    assert ratio >= 0.5, (
        f"At least 50% of frames must contain a peak within +/-{TONE_TOLERANCE_HZ} Hz "
        f"of one of {TARGET_TONES_HZ}; got {hits}/{len(frames)} ({ratio:.2%})."
    )
