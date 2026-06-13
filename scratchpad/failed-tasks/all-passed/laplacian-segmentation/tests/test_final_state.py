import json
import os

import pytest

WORKSPACE = "/workspace"
INPUT_WAV = os.path.join(WORKSPACE, "input.wav")
SEGMENTS_JSON = os.path.join(WORKSPACE, "segments.json")

# Tolerances for boundary alignment (in seconds).
START_TOLERANCE = 0.1
END_TOLERANCE = 1.0
MAX_GAP = 0.5
MAX_OVERLAP = 0.1
MIN_DURATION = 1.0
MIN_SEGMENTS = 3
MAX_SEGMENTS = 15


@pytest.fixture(scope="module")
def audio_duration() -> float:
    import librosa

    assert os.path.isfile(INPUT_WAV), (
        f"Reference input audio {INPUT_WAV} is missing; "
        "cannot determine the expected total duration."
    )
    duration = float(librosa.get_duration(path=INPUT_WAV))
    assert duration > 1.0, (
        f"Reference input audio at {INPUT_WAV} has an unexpectedly short "
        f"duration ({duration:.3f}s)."
    )
    return duration


@pytest.fixture(scope="module")
def segments() -> list:
    assert os.path.isfile(SEGMENTS_JSON), (
        f"Expected output artifact {SEGMENTS_JSON} was not produced by the agent."
    )
    with open(SEGMENTS_JSON, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"{SEGMENTS_JSON} is not valid JSON: {exc.msg} (line {exc.lineno}, "
            f"col {exc.colno})."
        )
    assert isinstance(parsed, list), (
        f"{SEGMENTS_JSON} must contain a JSON list at the top level, "
        f"got {type(parsed).__name__}."
    )
    return parsed


def test_segments_file_exists():
    assert os.path.isfile(SEGMENTS_JSON), (
        f"Expected segmentation output {SEGMENTS_JSON} does not exist."
    )


def test_segments_json_is_a_non_empty_list(segments):
    assert len(segments) > 0, f"{SEGMENTS_JSON} contains an empty list of segments."


def test_segment_count_within_bounds(segments):
    n = len(segments)
    assert MIN_SEGMENTS <= n <= MAX_SEGMENTS, (
        f"Expected between {MIN_SEGMENTS} and {MAX_SEGMENTS} segments "
        f"(inclusive), got {n}."
    )


def test_each_segment_has_required_float_fields(segments):
    for idx, seg in enumerate(segments):
        assert isinstance(seg, dict), (
            f"Segment #{idx} is not a JSON object: got {type(seg).__name__}."
        )
        for key in ("start", "end"):
            assert key in seg, (
                f"Segment #{idx} is missing required key '{key}': {seg!r}."
            )
            value = seg[key]
            # Accept native ints/floats; reject bools and non-numeric strings.
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"Segment #{idx} field '{key}' must be a numeric (float) value, "
                f"got {type(value).__name__}: {value!r}."
            )


def test_segments_are_sorted_by_start(segments):
    starts = [float(s["start"]) for s in segments]
    for i in range(len(starts) - 1):
        assert starts[i] <= starts[i + 1], (
            f"Segments are not sorted by start time: "
            f"segment[{i}].start={starts[i]} > segment[{i + 1}].start={starts[i + 1]}."
        )


def test_each_segment_has_positive_minimum_duration(segments):
    for idx, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        duration = end - start
        assert duration > MIN_DURATION, (
            f"Segment #{idx} has non-positive or too-short duration "
            f"({duration:.4f}s; start={start}, end={end}); must be > "
            f"{MIN_DURATION}s."
        )


def test_first_segment_starts_near_zero(segments):
    first_start = float(segments[0]["start"])
    assert abs(first_start - 0.0) <= START_TOLERANCE, (
        f"First segment must start within {START_TOLERANCE}s of 0.0, "
        f"got start={first_start:.4f}s."
    )


def test_last_segment_ends_near_audio_duration(segments, audio_duration):
    last_end = float(segments[-1]["end"])
    diff = abs(last_end - audio_duration)
    assert diff <= END_TOLERANCE, (
        f"Last segment must end within {END_TOLERANCE}s of the audio "
        f"duration ({audio_duration:.4f}s); got end={last_end:.4f}s "
        f"(|diff|={diff:.4f}s)."
    )


def test_adjacent_segments_are_contiguous(segments):
    for i in range(len(segments) - 1):
        end_i = float(segments[i]["end"])
        start_next = float(segments[i + 1]["start"])
        gap = start_next - end_i
        if gap > 0:
            assert gap <= MAX_GAP, (
                f"Gap between segment #{i} (end={end_i:.4f}s) and "
                f"segment #{i + 1} (start={start_next:.4f}s) is "
                f"{gap:.4f}s, which exceeds the allowed maximum of "
                f"{MAX_GAP}s."
            )
        else:
            overlap = -gap
            assert overlap <= MAX_OVERLAP, (
                f"Overlap between segment #{i} (end={end_i:.4f}s) and "
                f"segment #{i + 1} (start={start_next:.4f}s) is "
                f"{overlap:.4f}s, which exceeds the allowed maximum of "
                f"{MAX_OVERLAP}s."
            )
