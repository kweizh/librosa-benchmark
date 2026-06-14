import json
import os

import pytest


MATCH_JSON = "/workspace/match.json"
REFERENCE_WAV = "/workspace/reference.wav"
QUERY_WAV = "/workspace/query.wav"
GROUND_TRUTH_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json")

REQUIRED_KEYS = {
    "offset_seconds",
    "match_score",
    "reference_hash_count",
    "query_hash_count",
}


@pytest.fixture(scope="module")
def match_result():
    assert os.path.isfile(MATCH_JSON), (
        f"Expected output file {MATCH_JSON} to exist after the task completes."
    )
    with open(MATCH_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{MATCH_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, dict), (
        f"{MATCH_JSON} must be a JSON object, got: {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def audio_durations():
    import librosa

    y_ref, sr_ref = librosa.load(REFERENCE_WAV, sr=None, mono=True)
    y_qry, sr_qry = librosa.load(QUERY_WAV, sr=None, mono=True)
    ref_dur = float(len(y_ref)) / float(sr_ref)
    qry_dur = float(len(y_qry)) / float(sr_qry)
    assert ref_dur > 0 and qry_dur > 0, (
        f"Audio durations must be positive, got ref={ref_dur}, qry={qry_dur}."
    )
    return ref_dur, qry_dur


@pytest.fixture(scope="module")
def true_offset():
    assert os.path.isfile(GROUND_TRUTH_JSON), (
        f"Verifier ground truth file {GROUND_TRUTH_JSON} is missing; cannot verify."
    )
    with open(GROUND_TRUTH_JSON, "r") as fh:
        gt = json.load(fh)
    assert "true_offset_seconds" in gt, (
        f"Ground truth file {GROUND_TRUTH_JSON} must contain key 'true_offset_seconds'."
    )
    value = gt["true_offset_seconds"]
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"Ground truth 'true_offset_seconds' must be numeric, got: {value!r}."
    )
    return float(value)


def test_match_json_has_exactly_required_keys(match_result):
    actual_keys = set(match_result.keys())
    assert actual_keys == REQUIRED_KEYS, (
        f"{MATCH_JSON} must contain exactly the keys {sorted(REQUIRED_KEYS)}, "
        f"got {sorted(actual_keys)}."
    )


def test_match_json_value_types(match_result):
    offset = match_result["offset_seconds"]
    assert isinstance(offset, (int, float)) and not isinstance(offset, bool), (
        f"'offset_seconds' must be numeric (int or float), got: {offset!r} "
        f"({type(offset).__name__})."
    )
    for key in ("match_score", "reference_hash_count", "query_hash_count"):
        value = match_result[key]
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"'{key}' must be an integer, got: {value!r} ({type(value).__name__})."
        )


def test_offset_seconds_matches_ground_truth(match_result, true_offset):
    offset = float(match_result["offset_seconds"])
    diff = abs(offset - true_offset)
    assert diff <= 0.30, (
        f"Recovered offset_seconds={offset:.4f} differs from true offset "
        f"{true_offset:.4f} by {diff:.4f}s, which exceeds the 0.30s tolerance."
    )


def test_match_score_meets_minimum(match_result):
    score = match_result["match_score"]
    assert score >= 20, (
        f"'match_score' must be at least 20 to indicate real hash matching, got {score}."
    )


def test_reference_hash_count_greater_than_query_hash_count(match_result):
    ref_n = match_result["reference_hash_count"]
    qry_n = match_result["query_hash_count"]
    assert qry_n > 0, (
        f"'query_hash_count' must be positive (non-empty hash set), got {qry_n}."
    )
    assert ref_n > qry_n, (
        f"'reference_hash_count' ({ref_n}) must be strictly greater than "
        f"'query_hash_count' ({qry_n}); the longer reference must yield more hashes."
    )


def test_offset_seconds_in_feasible_range(match_result, audio_durations):
    ref_dur, qry_dur = audio_durations
    offset = float(match_result["offset_seconds"])
    upper = ref_dur - qry_dur + 0.5
    assert 0.0 <= offset <= upper, (
        f"'offset_seconds'={offset:.4f} must lie in [0, {upper:.4f}] "
        f"(reference_duration={ref_dur:.4f}, query_duration={qry_dur:.4f})."
    )
