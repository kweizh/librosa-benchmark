import json
import math
import os

import pytest


ALIGNMENT_JSON = "/workspace/alignment.json"
REFERENCE_WAV = "/workspace/reference.wav"
COMPARISON_WAV = "/workspace/comparison.wav"

REQUIRED_KEYS = {
    "warping_path_frames",
    "timestamp_map",
    "total_cost",
    "hop_length",
    "sample_rate",
}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@pytest.fixture(scope="module")
def payload():
    assert os.path.isfile(ALIGNMENT_JSON), (
        f"Expected output file {ALIGNMENT_JSON} to exist after the task completes."
    )
    with open(ALIGNMENT_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{ALIGNMENT_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, dict), (
        f"{ALIGNMENT_JSON} must be a JSON object, got: {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def audio_info():
    import librosa

    y_ref, sr_ref = librosa.load(REFERENCE_WAV, sr=None, mono=True)
    y_comp, sr_comp = librosa.load(COMPARISON_WAV, sr=None, mono=True)
    return {
        "y_ref": y_ref,
        "sr_ref": int(sr_ref),
        "y_comp": y_comp,
        "sr_comp": int(sr_comp),
        "duration_ref": float(len(y_ref)) / float(sr_ref),
        "duration_comp": float(len(y_comp)) / float(sr_comp),
    }


@pytest.fixture(scope="module")
def chroma_frame_counts(payload, audio_info):
    import librosa

    hop_length = payload.get("hop_length")
    sample_rate = payload.get("sample_rate")
    assert _is_int(hop_length) and hop_length > 0, (
        f"hop_length must be a positive int, got: {hop_length!r}."
    )
    assert _is_int(sample_rate) and sample_rate > 0, (
        f"sample_rate must be a positive int, got: {sample_rate!r}."
    )

    chroma_ref = librosa.feature.chroma_cens(
        y=audio_info["y_ref"], sr=sample_rate, hop_length=hop_length
    )
    chroma_comp = librosa.feature.chroma_cens(
        y=audio_info["y_comp"], sr=sample_rate, hop_length=hop_length
    )
    return int(chroma_ref.shape[-1]), int(chroma_comp.shape[-1])


def test_payload_has_required_top_level_keys(payload):
    missing = REQUIRED_KEYS - set(payload.keys())
    assert not missing, (
        f"alignment.json is missing required top-level keys: {sorted(missing)}."
    )


def test_hop_length_and_sample_rate_are_positive_ints(payload):
    hop_length = payload.get("hop_length")
    sample_rate = payload.get("sample_rate")
    assert _is_int(hop_length) and hop_length > 0, (
        f"hop_length must be a positive int, got: {hop_length!r}."
    )
    assert _is_int(sample_rate) and sample_rate > 0, (
        f"sample_rate must be a positive int, got: {sample_rate!r}."
    )


def test_total_cost_is_finite_non_negative(payload):
    total_cost = payload.get("total_cost")
    assert _is_number(total_cost), (
        f"total_cost must be a number, got: {total_cost!r}."
    )
    assert math.isfinite(float(total_cost)), (
        f"total_cost must be finite, got: {total_cost!r}."
    )
    assert float(total_cost) >= 0.0, (
        f"total_cost must be non-negative, got: {total_cost!r}."
    )


def test_warping_path_pairs_well_formed(payload):
    wp = payload.get("warping_path_frames")
    assert isinstance(wp, list) and len(wp) > 0, (
        f"warping_path_frames must be a non-empty list, got: {type(wp).__name__} "
        f"with length {len(wp) if hasattr(wp, '__len__') else 'N/A'}."
    )
    for idx, pair in enumerate(wp):
        assert isinstance(pair, list) and len(pair) == 2, (
            f"warping_path_frames[{idx}] must be a length-2 list, got: {pair!r}."
        )
        assert _is_int(pair[0]) and _is_int(pair[1]), (
            f"warping_path_frames[{idx}] must contain two ints, got: {pair!r}."
        )
        assert pair[0] >= 0 and pair[1] >= 0, (
            f"warping_path_frames[{idx}] must contain non-negative ints, got: {pair!r}."
        )


def test_warping_path_monotonic_non_decreasing(payload):
    wp = payload["warping_path_frames"]
    for i in range(len(wp) - 1):
        a, b = wp[i], wp[i + 1]
        assert b[0] - a[0] >= 0, (
            f"warping_path_frames not monotonic on the reference axis between "
            f"index {i} ({a}) and {i+1} ({b})."
        )
        assert b[1] - a[1] >= 0, (
            f"warping_path_frames not monotonic on the comparison axis between "
            f"index {i} ({a}) and {i+1} ({b})."
        )


def test_warping_path_endpoints(payload, chroma_frame_counts):
    n_ref, n_comp = chroma_frame_counts
    wp = payload["warping_path_frames"]
    first = wp[0]
    last = wp[-1]
    assert first[0] <= 2 and first[1] <= 2, (
        f"warping_path_frames[0]={first} must start near (0, 0) with both indices <= 2."
    )
    assert abs(last[0] - (n_ref - 1)) <= 3, (
        f"warping_path_frames[-1][0]={last[0]} must be within 3 frames of "
        f"N_ref-1={n_ref - 1}."
    )
    assert abs(last[1] - (n_comp - 1)) <= 3, (
        f"warping_path_frames[-1][1]={last[1]} must be within 3 frames of "
        f"N_comp-1={n_comp - 1}."
    )


def test_timestamp_map_schema_and_sorting(payload):
    tm = payload.get("timestamp_map")
    assert isinstance(tm, list) and len(tm) >= 2, (
        f"timestamp_map must be a list with at least 2 entries, got length "
        f"{len(tm) if hasattr(tm, '__len__') else 'N/A'}."
    )
    for idx, entry in enumerate(tm):
        assert isinstance(entry, dict), (
            f"timestamp_map[{idx}] must be an object, got: {type(entry).__name__}."
        )
        for key in ("ref_time", "comp_time"):
            assert key in entry, (
                f"timestamp_map[{idx}] is missing required key '{key}': {entry!r}."
            )
            assert _is_number(entry[key]), (
                f"timestamp_map[{idx}][{key!r}] must be numeric, got: {entry[key]!r}."
            )
    ref_times = [float(e["ref_time"]) for e in tm]
    for i in range(len(ref_times) - 1):
        assert ref_times[i + 1] >= ref_times[i], (
            f"timestamp_map must be sorted by ref_time; index {i+1} ({ref_times[i+1]}) "
            f"is smaller than index {i} ({ref_times[i]})."
        )


def test_timestamp_map_ref_time_spacing_and_span(payload, audio_info):
    tm = payload["timestamp_map"]
    ref_times = [float(e["ref_time"]) for e in tm]

    assert 0.0 <= ref_times[0] <= 0.25 + 1e-6, (
        f"First ref_time must be in [0, 0.25], got {ref_times[0]}."
    )

    duration_ref = audio_info["duration_ref"]
    assert duration_ref - ref_times[-1] <= 0.5 + 1e-6, (
        f"Last ref_time {ref_times[-1]} must be within 0.5s of reference duration "
        f"{duration_ref}."
    )

    for i in range(len(ref_times) - 1):
        delta = ref_times[i + 1] - ref_times[i]
        assert 0.45 - 1e-6 <= delta <= 0.55 + 1e-6, (
            f"Consecutive ref_time delta at index {i} must be in [0.45, 0.55], "
            f"got {delta}."
        )


def test_timestamp_map_comp_time_monotonic_and_in_range(payload, audio_info):
    tm = payload["timestamp_map"]
    comp_times = [float(e["comp_time"]) for e in tm]

    duration_comp = audio_info["duration_comp"]
    upper = duration_comp + 0.1
    for idx, ct in enumerate(comp_times):
        assert -1e-6 <= ct <= upper + 1e-6, (
            f"comp_time at index {idx} must be in [0, {upper}], got {ct}."
        )

    for i in range(len(comp_times) - 1):
        assert comp_times[i + 1] - comp_times[i] >= -1e-6, (
            f"comp_time must be monotonically non-decreasing; index {i+1} "
            f"({comp_times[i+1]}) is smaller than index {i} ({comp_times[i]})."
        )
