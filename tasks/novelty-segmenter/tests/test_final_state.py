import json
import os

import pytest


OUTPUT_PATH = "/workspace/boundaries.json"
INPUT_AUDIO_PATH = "/workspace/input.wav"

# Ground-truth structural change-points baked into the Dockerfile-built input:
# 5 s of trumpet, 5 s of nutcracker, 5 s of choice (all 22050 Hz mono),
# concatenated with crossfades < 100 ms.
GROUND_TRUTH_BOUNDARIES_SEC = [5.0, 10.0]
GROUND_TRUTH_TOLERANCE_SEC = 1.0


def _load_boundaries():
    assert os.path.isfile(OUTPUT_PATH), (
        f"Expected the executor to produce {OUTPUT_PATH}, but it does not exist."
    )
    with open(OUTPUT_PATH, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{OUTPUT_PATH} is not valid JSON: {exc}"
            ) from exc
    return data


def _audio_duration_sec():
    import librosa

    return float(librosa.get_duration(path=INPUT_AUDIO_PATH))


def test_output_file_exists_and_has_expected_schema():
    data = _load_boundaries()
    assert isinstance(data, dict), (
        f"Top-level JSON object in {OUTPUT_PATH} must be a dict, got {type(data).__name__}."
    )
    assert "boundaries_sec" in data, (
        f"Expected key 'boundaries_sec' to be present in {OUTPUT_PATH}, got keys: {list(data.keys())}."
    )
    boundaries = data["boundaries_sec"]
    assert isinstance(boundaries, list), (
        f"'boundaries_sec' must be a JSON list, got {type(boundaries).__name__}."
    )


def test_boundaries_are_floats_and_strictly_increasing():
    data = _load_boundaries()
    boundaries = data["boundaries_sec"]
    assert len(boundaries) >= 1, "'boundaries_sec' must not be empty."
    for i, value in enumerate(boundaries):
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"boundaries_sec[{i}] must be a number, got {type(value).__name__}: {value!r}."
        )
    floats = [float(v) for v in boundaries]
    for i in range(1, len(floats)):
        assert floats[i] > floats[i - 1], (
            f"boundaries_sec must be strictly increasing, but "
            f"boundaries_sec[{i - 1}]={floats[i - 1]} >= boundaries_sec[{i}]={floats[i]}."
        )


def test_boundaries_lie_inside_audio_duration():
    data = _load_boundaries()
    boundaries = [float(v) for v in data["boundaries_sec"]]
    duration = _audio_duration_sec()
    eps = 1e-6
    for i, value in enumerate(boundaries):
        assert -eps <= value <= duration + eps, (
            f"boundaries_sec[{i}]={value} is outside the valid range [0, {duration:.4f}]."
        )


def test_at_least_two_boundaries_detected():
    data = _load_boundaries()
    boundaries = data["boundaries_sec"]
    assert len(boundaries) >= 2, (
        f"Expected at least 2 detected structural boundaries, got {len(boundaries)}: {boundaries}."
    )


@pytest.mark.parametrize("ground_truth_sec", GROUND_TRUTH_BOUNDARIES_SEC)
def test_each_ground_truth_change_point_is_covered(ground_truth_sec):
    data = _load_boundaries()
    boundaries = [float(v) for v in data["boundaries_sec"]]
    nearest = min((abs(b - ground_truth_sec) for b in boundaries), default=float("inf"))
    assert nearest <= GROUND_TRUTH_TOLERANCE_SEC, (
        f"No detected boundary within {GROUND_TRUTH_TOLERANCE_SEC:.2f}s of the "
        f"ground-truth change-point at {ground_truth_sec:.2f}s. "
        f"Closest detected boundary distance was {nearest:.3f}s. "
        f"Detected: {boundaries}."
    )
