import bisect
import json
import os

import numpy as np
import pytest
import soundfile as sf

WORKSPACE = "/workspace"
REFERENCE_WAV = os.path.join(WORKSPACE, "reference.wav")
PERFORMANCE_WAV = os.path.join(WORKSPACE, "performance.wav")
ALIGNMENT_JSON = os.path.join(WORKSPACE, "alignment.json")

# Ground truth used by the Dockerfile when synthesizing performance.wav from reference.wav.
GROUND_TRUTH_OFFSET_SECONDS = 5.0
GROUND_TRUTH_STRETCH_RATE = 1.05

# Tolerances from task design.
MONOTONIC_TOL = 1e-6
BOUNDARY_TOL_SECONDS = 0.5
ACCURACY_TOL_SECONDS = 1.5
MIN_POINTS = 50


def _load_alignment():
    assert os.path.isfile(ALIGNMENT_JSON), (
        f"Expected alignment output at {ALIGNMENT_JSON}, but it does not exist."
    )
    with open(ALIGNMENT_JSON, "r") as f:
        data = json.load(f)
    return data


def _wav_duration(path: str) -> float:
    info = sf.info(path)
    assert info.samplerate > 0, f"Invalid sample rate in {path}: {info.samplerate}"
    return float(info.frames) / float(info.samplerate)


@pytest.fixture(scope="module")
def alignment():
    return _load_alignment()


@pytest.fixture(scope="module")
def ref_duration():
    return _wav_duration(REFERENCE_WAV)


@pytest.fixture(scope="module")
def perf_duration():
    return _wav_duration(PERFORMANCE_WAV)


def test_alignment_file_is_valid_json_list(alignment):
    assert isinstance(alignment, list), (
        f"Expected /workspace/alignment.json to contain a JSON list, got {type(alignment).__name__}."
    )


def test_alignment_has_minimum_points(alignment):
    assert len(alignment) >= MIN_POINTS, (
        f"Expected at least {MIN_POINTS} mapping points in alignment.json, "
        f"got {len(alignment)}."
    )


def test_alignment_entries_have_required_float_keys(alignment):
    for i, entry in enumerate(alignment):
        assert isinstance(entry, dict), (
            f"Entry #{i} is not a JSON object: {entry!r}."
        )
        assert set(entry.keys()) == {"ref_time", "perf_time"}, (
            f"Entry #{i} must have exactly the keys 'ref_time' and 'perf_time'; "
            f"got keys {sorted(entry.keys())!r}."
        )
        for key in ("ref_time", "perf_time"):
            value = entry[key]
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"Entry #{i} key {key!r} must be a numeric (float) value, got {value!r} "
                f"of type {type(value).__name__}."
            )
            assert np.isfinite(float(value)), (
                f"Entry #{i} key {key!r} must be a finite number, got {value!r}."
            )


def test_alignment_is_monotonic_non_decreasing(alignment):
    ref_times = [float(e["ref_time"]) for e in alignment]
    perf_times = [float(e["perf_time"]) for e in alignment]
    for i in range(1, len(ref_times)):
        assert ref_times[i] >= ref_times[i - 1] - MONOTONIC_TOL, (
            f"ref_time sequence is not monotonic non-decreasing at index {i}: "
            f"{ref_times[i - 1]} -> {ref_times[i]}."
        )
        assert perf_times[i] >= perf_times[i - 1] - MONOTONIC_TOL, (
            f"perf_time sequence is not monotonic non-decreasing at index {i}: "
            f"{perf_times[i - 1]} -> {perf_times[i]}."
        )


def test_alignment_starts_near_origin(alignment):
    first = alignment[0]
    assert abs(float(first["ref_time"])) <= BOUNDARY_TOL_SECONDS, (
        f"First mapping's ref_time must be within {BOUNDARY_TOL_SECONDS}s of 0, "
        f"got {first['ref_time']}."
    )
    assert abs(float(first["perf_time"])) <= BOUNDARY_TOL_SECONDS, (
        f"First mapping's perf_time must be within {BOUNDARY_TOL_SECONDS}s of 0, "
        f"got {first['perf_time']}."
    )


def test_alignment_ends_near_durations(alignment, ref_duration, perf_duration):
    last = alignment[-1]
    assert abs(float(last["ref_time"]) - ref_duration) <= BOUNDARY_TOL_SECONDS, (
        f"Last mapping's ref_time must be within {BOUNDARY_TOL_SECONDS}s of "
        f"reference duration {ref_duration:.3f}, got {last['ref_time']}."
    )
    assert abs(float(last["perf_time"]) - perf_duration) <= BOUNDARY_TOL_SECONDS, (
        f"Last mapping's perf_time must be within {BOUNDARY_TOL_SECONDS}s of "
        f"performance duration {perf_duration:.3f}, got {last['perf_time']}."
    )


def _predict_perf_time(alignment, t_ref: float) -> float:
    """Linear interpolation of perf_time for a given ref_time using the alignment mapping."""
    ref_times = [float(e["ref_time"]) for e in alignment]
    perf_times = [float(e["perf_time"]) for e in alignment]

    if t_ref <= ref_times[0]:
        return perf_times[0]
    if t_ref >= ref_times[-1]:
        return perf_times[-1]

    # bisect_left on monotone non-decreasing sequence; ensure we have strictly increasing pair.
    idx = bisect.bisect_left(ref_times, t_ref)
    # Find the bracketing pair (lo, hi) with ref_times[lo] <= t_ref <= ref_times[hi]
    hi = idx
    while hi < len(ref_times) and ref_times[hi] < t_ref:
        hi += 1
    if hi >= len(ref_times):
        return perf_times[-1]
    lo = hi - 1
    while lo > 0 and ref_times[lo] > t_ref:
        lo -= 1

    r_lo, r_hi = ref_times[lo], ref_times[hi]
    p_lo, p_hi = perf_times[lo], perf_times[hi]
    if r_hi <= r_lo:
        return p_lo
    frac = (t_ref - r_lo) / (r_hi - r_lo)
    return p_lo + frac * (p_hi - p_lo)


def test_alignment_accuracy_against_ground_truth(alignment, ref_duration):
    """For known reference timestamps, predicted performance time should be close to
    OFFSET + t_ref / STRETCH_RATE, since performance.wav = silence(OFFSET) + time_stretch(reference, STRETCH_RATE).
    """
    candidate_refs = [2.0, 5.0, 10.0, 20.0, 40.0]
    sampled = [t for t in candidate_refs if t < ref_duration - 0.5]
    assert len(sampled) >= 3, (
        f"Reference audio is too short to validate alignment accuracy at expected "
        f"reference timestamps {candidate_refs}; ref_duration={ref_duration:.3f}."
    )

    errors = []
    for t_ref in sampled:
        expected_perf = GROUND_TRUTH_OFFSET_SECONDS + (t_ref / GROUND_TRUTH_STRETCH_RATE)
        predicted_perf = _predict_perf_time(alignment, t_ref)
        err = abs(predicted_perf - expected_perf)
        errors.append((t_ref, expected_perf, predicted_perf, err))
        assert err <= ACCURACY_TOL_SECONDS, (
            f"Alignment is inaccurate at ref_time={t_ref:.3f}s: "
            f"expected perf_time≈{expected_perf:.3f}s, "
            f"predicted {predicted_perf:.3f}s, error={err:.3f}s (tolerance {ACCURACY_TOL_SECONDS}s)."
        )
