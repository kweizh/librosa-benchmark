import json
import os
from collections import defaultdict

import pytest


HIERARCHY_JSON = "/workspace/hierarchy.json"
INPUT_WAV = "/workspace/input.wav"


@pytest.fixture(scope="module")
def hierarchy():
    assert os.path.isfile(HIERARCHY_JSON), (
        f"Expected output file {HIERARCHY_JSON} to exist after the task completes."
    )
    with open(HIERARCHY_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{HIERARCHY_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, dict), (
        f"{HIERARCHY_JSON} must be a JSON object, got: {type(data).__name__}."
    )
    assert set(data.keys()) == {"coarse", "fine"}, (
        f"{HIERARCHY_JSON} must have exactly top-level keys 'coarse' and 'fine', "
        f"got: {sorted(data.keys())}."
    )
    assert isinstance(data["coarse"], list) and len(data["coarse"]) > 0, (
        f"'coarse' must be a non-empty list, got: "
        f"{type(data['coarse']).__name__} of length "
        f"{len(data['coarse']) if hasattr(data['coarse'], '__len__') else 'N/A'}."
    )
    assert isinstance(data["fine"], list) and len(data["fine"]) > 0, (
        f"'fine' must be a non-empty list, got: "
        f"{type(data['fine']).__name__} of length "
        f"{len(data['fine']) if hasattr(data['fine'], '__len__') else 'N/A'}."
    )
    return data


@pytest.fixture(scope="module")
def audio_duration():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    duration = float(len(y)) / float(sr)
    assert duration > 0, f"Reference audio duration must be positive, got {duration}."
    return duration


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def test_coarse_entries_have_required_schema(hierarchy):
    coarse = hierarchy["coarse"]
    for idx, seg in enumerate(coarse):
        assert isinstance(seg, dict), (
            f"coarse[{idx}] is not a JSON object, got: {type(seg).__name__}."
        )
        for key in ("index", "start", "end"):
            assert key in seg, (
                f"coarse[{idx}] is missing required key '{key}': {seg!r}."
            )
        assert _is_int(seg["index"]), (
            f"coarse[{idx}] 'index' must be an int, got: {seg['index']!r}."
        )
        assert _is_number(seg["start"]), (
            f"coarse[{idx}] 'start' must be numeric, got: {seg['start']!r}."
        )
        assert _is_number(seg["end"]), (
            f"coarse[{idx}] 'end' must be numeric, got: {seg['end']!r}."
        )
        assert float(seg["start"]) < float(seg["end"]), (
            f"coarse[{idx}] must satisfy start < end, "
            f"got start={seg['start']}, end={seg['end']}."
        )
        assert float(seg["end"]) - float(seg["start"]) > 0.5, (
            f"coarse[{idx}] duration must be > 0.5s, "
            f"got {float(seg['end']) - float(seg['start'])}s."
        )


def test_coarse_indices_are_unique_and_dense(hierarchy):
    coarse = hierarchy["coarse"]
    indices = [seg["index"] for seg in coarse]
    assert len(set(indices)) == len(indices), (
        f"coarse indices must be unique, got duplicates in: {indices}."
    )
    assert set(indices) == set(range(len(coarse))), (
        f"coarse indices must form the set {{0, ..., len(coarse)-1}}, "
        f"got: {sorted(indices)} for len={len(coarse)}."
    )


def test_coarse_count_within_bounds(hierarchy):
    coarse = hierarchy["coarse"]
    assert 4 <= len(coarse) <= 8, (
        f"coarse must contain between 4 and 8 entries, got {len(coarse)}."
    )


def test_coarse_is_sorted_by_start(hierarchy):
    coarse = hierarchy["coarse"]
    starts = [float(seg["start"]) for seg in coarse]
    assert starts == sorted(starts), (
        f"coarse must already be sorted by start, got starts: {starts}."
    )


def test_coarse_covers_audio_without_big_gaps_or_overlaps(hierarchy, audio_duration):
    coarse = sorted(hierarchy["coarse"], key=lambda s: float(s["start"]))

    first_start = float(coarse[0]["start"])
    assert first_start <= 0.3 + 1e-6, (
        f"First coarse segment must start within 0.3s of 0, got start={first_start}."
    )

    last_end = float(coarse[-1]["end"])
    assert abs(last_end - audio_duration) <= 0.5 + 1e-6, (
        f"Last coarse segment end must be within 0.5s of audio duration "
        f"{audio_duration}, got end={last_end}."
    )

    for i in range(len(coarse) - 1):
        end_i = float(coarse[i]["end"])
        start_next = float(coarse[i + 1]["start"])
        gap = start_next - end_i
        assert gap <= 0.3 + 1e-6, (
            f"Gap between coarse[{i}] (end={end_i}) and coarse[{i+1}] "
            f"(start={start_next}) exceeds 0.3s: gap={gap}."
        )
        overlap = end_i - start_next
        assert overlap <= 1e-6, (
            f"Overlap between coarse[{i}] (end={end_i}) and coarse[{i+1}] "
            f"(start={start_next}) exceeds the float-noise tolerance: overlap={overlap}."
        )


def test_fine_entries_have_required_schema(hierarchy):
    fine = hierarchy["fine"]
    for idx, seg in enumerate(fine):
        assert isinstance(seg, dict), (
            f"fine[{idx}] is not a JSON object, got: {type(seg).__name__}."
        )
        for key in ("index", "start", "end", "parent_index"):
            assert key in seg, (
                f"fine[{idx}] is missing required key '{key}': {seg!r}."
            )
        assert _is_int(seg["index"]), (
            f"fine[{idx}] 'index' must be an int, got: {seg['index']!r}."
        )
        assert _is_int(seg["parent_index"]), (
            f"fine[{idx}] 'parent_index' must be an int, got: {seg['parent_index']!r}."
        )
        assert _is_number(seg["start"]), (
            f"fine[{idx}] 'start' must be numeric, got: {seg['start']!r}."
        )
        assert _is_number(seg["end"]), (
            f"fine[{idx}] 'end' must be numeric, got: {seg['end']!r}."
        )
        assert float(seg["start"]) < float(seg["end"]), (
            f"fine[{idx}] must satisfy start < end, "
            f"got start={seg['start']}, end={seg['end']}."
        )
        assert float(seg["end"]) - float(seg["start"]) > 0.1, (
            f"fine[{idx}] duration must be > 0.1s, "
            f"got {float(seg['end']) - float(seg['start'])}s."
        )


def test_fine_indices_are_unique(hierarchy):
    fine = hierarchy["fine"]
    indices = [seg["index"] for seg in fine]
    assert len(set(indices)) == len(indices), (
        f"fine indices must be unique, got duplicates in: {indices}."
    )


def test_fine_is_sorted_by_start(hierarchy):
    fine = hierarchy["fine"]
    starts = [float(seg["start"]) for seg in fine]
    assert starts == sorted(starts), (
        f"fine must already be sorted by start, got starts: {starts}."
    )


def test_fine_parent_indices_are_valid(hierarchy):
    coarse_len = len(hierarchy["coarse"])
    for idx, seg in enumerate(hierarchy["fine"]):
        parent = seg["parent_index"]
        assert 0 <= parent < coarse_len, (
            f"fine[{idx}].parent_index={parent} is out of range "
            f"[0, {coarse_len})."
        )


def test_each_coarse_segment_has_exactly_three_children_that_cover_it(hierarchy):
    coarse_by_index = {seg["index"]: seg for seg in hierarchy["coarse"]}
    children = defaultdict(list)
    for seg in hierarchy["fine"]:
        children[seg["parent_index"]].append(seg)

    for parent_index, parent in coarse_by_index.items():
        kids = children.get(parent_index, [])
        assert len(kids) == 3, (
            f"coarse index {parent_index} must have exactly 3 fine children, "
            f"got {len(kids)}."
        )
        kids_sorted = sorted(kids, key=lambda s: float(s["start"]))

        c_start = float(parent["start"])
        c_end = float(parent["end"])

        first_start = float(kids_sorted[0]["start"])
        assert abs(first_start - c_start) <= 0.05 + 1e-6, (
            f"First fine child of coarse[{parent_index}] must start within "
            f"0.05s of coarse start {c_start}, got {first_start}."
        )

        last_end = float(kids_sorted[-1]["end"])
        assert abs(last_end - c_end) <= 0.05 + 1e-6, (
            f"Last fine child of coarse[{parent_index}] must end within "
            f"0.05s of coarse end {c_end}, got {last_end}."
        )

        for i in range(len(kids_sorted) - 1):
            end_i = float(kids_sorted[i]["end"])
            start_next = float(kids_sorted[i + 1]["start"])
            gap = start_next - end_i
            overlap = end_i - start_next
            assert gap <= 0.05 + 1e-6, (
                f"Gap between fine children {i} and {i+1} of coarse[{parent_index}] "
                f"exceeds 0.05s: gap={gap} (end_i={end_i}, start_next={start_next})."
            )
            assert overlap <= 0.05 + 1e-6, (
                f"Overlap between fine children {i} and {i+1} of coarse[{parent_index}] "
                f"exceeds 0.05s: overlap={overlap} (end_i={end_i}, start_next={start_next})."
            )
