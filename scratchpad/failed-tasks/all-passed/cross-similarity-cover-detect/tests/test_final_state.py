import json
import math
import os

import pytest


OUTPUT_JSON = "/workspace/cover_decision.json"
GROUND_TRUTH = "/workspace/ground_truth.json"

REQUIRED_KEYS = {
    "is_cover",
    "best_transposition_semitones",
    "normalized_cost",
    "best_offset_seconds",
}


@pytest.fixture(scope="module")
def decision():
    assert os.path.isfile(OUTPUT_JSON), (
        f"Expected output file {OUTPUT_JSON} to exist after the task completes."
    )
    with open(OUTPUT_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{OUTPUT_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, dict), (
        f"{OUTPUT_JSON} must be a JSON object, got: {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def ground_truth():
    assert os.path.isfile(GROUND_TRUTH), (
        f"Verifier expected ground-truth file {GROUND_TRUTH} to exist."
    )
    with open(GROUND_TRUTH, "r") as fh:
        data = json.load(fh)
    assert isinstance(data, dict) and "is_cover" in data, (
        f"{GROUND_TRUTH} must be an object containing key 'is_cover'."
    )
    assert isinstance(data["is_cover"], bool), (
        f"{GROUND_TRUTH} 'is_cover' must be a boolean, got: {data['is_cover']!r}."
    )
    return data


def test_output_has_required_keys(decision):
    missing = REQUIRED_KEYS - set(decision.keys())
    assert not missing, (
        f"{OUTPUT_JSON} is missing required keys: {sorted(missing)}; "
        f"got keys: {sorted(decision.keys())}."
    )


def test_is_cover_is_boolean(decision):
    assert isinstance(decision["is_cover"], bool), (
        f"'is_cover' must be a JSON boolean, got: {decision['is_cover']!r} "
        f"({type(decision['is_cover']).__name__})."
    )


def test_best_transposition_is_int_in_range(decision):
    value = decision["best_transposition_semitones"]
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"'best_transposition_semitones' must be an integer, got: {value!r} "
        f"({type(value).__name__})."
    )
    assert -6 <= value <= 5, (
        f"'best_transposition_semitones' must be in inclusive range [-6, 5], got: {value}."
    )


def test_normalized_cost_is_finite_nonnegative(decision):
    value = decision["normalized_cost"]
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"'normalized_cost' must be numeric, got: {value!r} ({type(value).__name__})."
    )
    fvalue = float(value)
    assert math.isfinite(fvalue), (
        f"'normalized_cost' must be a finite number, got: {fvalue}."
    )
    assert fvalue >= 0.0, (
        f"'normalized_cost' must be non-negative, got: {fvalue}."
    )


def test_best_offset_seconds_is_finite_nonnegative(decision):
    value = decision["best_offset_seconds"]
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"'best_offset_seconds' must be numeric, got: {value!r} ({type(value).__name__})."
    )
    fvalue = float(value)
    assert math.isfinite(fvalue), (
        f"'best_offset_seconds' must be a finite number, got: {fvalue}."
    )
    assert fvalue >= 0.0, (
        f"'best_offset_seconds' must be non-negative, got: {fvalue}."
    )


def test_decision_rule_consistent(decision):
    cost = float(decision["normalized_cost"])
    expected = cost < 0.6
    assert bool(decision["is_cover"]) == expected, (
        f"Decision rule violated: is_cover={decision['is_cover']}, normalized_cost={cost}, "
        f"expected is_cover==(normalized_cost<0.6)={expected}."
    )


def test_is_cover_matches_ground_truth(decision, ground_truth):
    assert bool(decision["is_cover"]) == bool(ground_truth["is_cover"]), (
        f"Reported is_cover={decision['is_cover']} does not match ground truth "
        f"is_cover={ground_truth['is_cover']}."
    )


def test_normalized_cost_consistent_with_truth(decision, ground_truth):
    cost = float(decision["normalized_cost"])
    if bool(ground_truth["is_cover"]):
        assert cost < 0.6, (
            f"Ground truth is is_cover=True but reported normalized_cost={cost} is not < 0.6."
        )
    else:
        assert cost >= 0.6, (
            f"Ground truth is is_cover=False but reported normalized_cost={cost} is not >= 0.6."
        )
