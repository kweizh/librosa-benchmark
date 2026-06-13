import json
import os

import pytest


WORKSPACE = "/workspace"
MATCH_JSON = os.path.join(WORKSPACE, "match.json")

# Ground truth: query.wav was extracted from database.wav starting at 5.0s.
GROUND_TRUTH_OFFSET = 5.0
OFFSET_TOLERANCE = 0.75


@pytest.fixture(scope="module")
def match_payload():
    assert os.path.isfile(MATCH_JSON), (
        f"Expected the agent to create {MATCH_JSON}, but it does not exist."
    )
    with open(MATCH_JSON, "r") as f:
        raw = f.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"{MATCH_JSON} is not valid JSON: {exc}. Raw contents: {raw!r}"
        )
    assert isinstance(payload, dict), (
        f"{MATCH_JSON} must be a JSON object, got type {type(payload).__name__}."
    )
    return payload


def test_match_json_has_required_keys(match_payload):
    for key in ("start_time", "confidence"):
        assert key in match_payload, (
            f"Expected key '{key}' in {MATCH_JSON}, got keys: "
            f"{sorted(match_payload.keys())}."
        )


def test_match_json_start_time_is_float(match_payload):
    start_time = match_payload["start_time"]
    assert isinstance(start_time, (int, float)) and not isinstance(start_time, bool), (
        f"'start_time' must be a number, got {type(start_time).__name__}: "
        f"{start_time!r}."
    )


def test_match_json_confidence_is_float_in_range(match_payload):
    confidence = match_payload["confidence"]
    assert isinstance(confidence, (int, float)) and not isinstance(confidence, bool), (
        f"'confidence' must be a number, got {type(confidence).__name__}: "
        f"{confidence!r}."
    )
    assert 0.0 < float(confidence) <= 1.0, (
        f"'confidence' must be in (0.0, 1.0], got {confidence!r}."
    )


def test_match_json_start_time_close_to_ground_truth(match_payload):
    start_time = float(match_payload["start_time"])
    delta = abs(start_time - GROUND_TRUTH_OFFSET)
    assert delta <= OFFSET_TOLERANCE, (
        f"Reported start_time={start_time:.3f}s is {delta:.3f}s away from the "
        f"ground-truth offset {GROUND_TRUTH_OFFSET:.3f}s, which exceeds the "
        f"allowed tolerance of {OFFSET_TOLERANCE:.3f}s."
    )
