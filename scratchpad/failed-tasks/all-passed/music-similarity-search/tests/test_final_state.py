import glob
import json
import math
import os


WORKSPACE = "/workspace"
LIBRARY_DIR = os.path.join(WORKSPACE, "library")
RESULTS_PATH = os.path.join(WORKSPACE, "results.json")
GROUND_TRUTH_PATH = os.path.join(WORKSPACE, ".ground_truth_match")


def _load_results():
    assert os.path.isfile(RESULTS_PATH), (
        f"Expected results file at {RESULTS_PATH} but it was not produced."
    )
    with open(RESULTS_PATH) as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{RESULTS_PATH} does not contain valid JSON: {exc}"
            )
    return payload


def _expected_library_basenames():
    paths = sorted(glob.glob(os.path.join(LIBRARY_DIR, "*.wav")))
    assert paths, f"No WAV files found in {LIBRARY_DIR}."
    return [os.path.basename(p) for p in paths]


def _ground_truth_match():
    assert os.path.isfile(GROUND_TRUTH_PATH), (
        f"Ground-truth file {GROUND_TRUTH_PATH} is missing; the test image was built incorrectly."
    )
    with open(GROUND_TRUTH_PATH) as f:
        name = f.read().strip()
    assert name, "Ground-truth file is empty."
    return name


def test_results_json_is_valid_json_with_results_list():
    payload = _load_results()
    assert isinstance(payload, dict), (
        f"Top-level JSON in {RESULTS_PATH} must be an object, got {type(payload).__name__}."
    )
    assert "results" in payload, (
        f"Top-level JSON in {RESULTS_PATH} must contain a 'results' key."
    )
    assert isinstance(payload["results"], list), (
        f"'results' must be a list, got {type(payload['results']).__name__}."
    )


def test_results_entries_have_required_schema():
    payload = _load_results()
    for i, entry in enumerate(payload["results"]):
        assert isinstance(entry, dict), (
            f"results[{i}] must be an object, got {type(entry).__name__}."
        )
        assert "filename" in entry, f"results[{i}] is missing 'filename'."
        assert "similarity" in entry, f"results[{i}] is missing 'similarity'."
        assert isinstance(entry["filename"], str), (
            f"results[{i}].filename must be a string."
        )
        assert isinstance(entry["similarity"], (int, float)) and not isinstance(
            entry["similarity"], bool
        ), f"results[{i}].similarity must be numeric, got {type(entry['similarity']).__name__}."
        assert math.isfinite(float(entry["similarity"])), (
            f"results[{i}].similarity must be a finite number."
        )


def test_results_cover_every_library_file_exactly_once():
    payload = _load_results()
    expected = set(_expected_library_basenames())
    got = [entry["filename"] for entry in payload["results"]]
    got_set = set(got)
    assert len(got) == len(got_set), (
        f"Duplicate filenames in results: {[f for f in got if got.count(f) > 1]}."
    )
    missing = expected - got_set
    extra = got_set - expected
    assert not missing, f"Missing library entries in results: {sorted(missing)}."
    assert not extra, f"Unexpected entries in results not present in library: {sorted(extra)}."


def test_filenames_are_basenames_only():
    payload = _load_results()
    for entry in payload["results"]:
        fname = entry["filename"]
        assert os.path.basename(fname) == fname, (
            f"filename {fname!r} must be a basename (no directory components)."
        )


def test_similarities_in_valid_range():
    payload = _load_results()
    for entry in payload["results"]:
        sim = float(entry["similarity"])
        assert -1.0 <= sim <= 1.0, (
            f"similarity for {entry['filename']!r} is {sim}, outside [-1.0, 1.0]."
        )


def test_results_sorted_descending_by_similarity():
    payload = _load_results()
    sims = [float(e["similarity"]) for e in payload["results"]]
    for i in range(len(sims) - 1):
        assert sims[i] >= sims[i + 1], (
            f"results are not sorted descending: index {i} has similarity {sims[i]} "
            f"but next entry has {sims[i + 1]}."
        )


def test_ground_truth_track_is_top_ranked():
    payload = _load_results()
    gt = _ground_truth_match()
    top2 = [e["filename"] for e in payload["results"][:2]]
    assert gt in top2, (
        f"Ground-truth matching track {gt!r} must rank in the top 2, "
        f"but top 2 were {top2}."
    )


def test_ground_truth_similarity_exceeds_threshold():
    payload = _load_results()
    gt = _ground_truth_match()
    matched = [e for e in payload["results"] if e["filename"] == gt]
    assert matched, (
        f"Ground-truth track {gt!r} was not present in results at all."
    )
    sim = float(matched[0]["similarity"])
    assert sim > 0.7, (
        f"Ground-truth track {gt!r} similarity is {sim}, which is not > 0.7 as required."
    )
