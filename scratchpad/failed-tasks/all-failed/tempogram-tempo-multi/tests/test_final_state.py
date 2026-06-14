import json
import os

import pytest


TEMPO_JSON = "/workspace/tempo_candidates.json"
INPUT_WAV = "/workspace/input.wav"


@pytest.fixture(scope="module")
def candidates():
    assert os.path.isfile(TEMPO_JSON), (
        f"Expected output file {TEMPO_JSON} to exist after the task completes."
    )
    with open(TEMPO_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{TEMPO_JSON} is not valid JSON: {exc}")
    assert isinstance(data, list), (
        f"{TEMPO_JSON} must be a JSON array, got: {type(data).__name__}."
    )
    assert len(data) == 3, (
        f"{TEMPO_JSON} must contain exactly 3 candidates, got: {len(data)}."
    )
    return data


@pytest.fixture(scope="module")
def reference_signals():
    import numpy as np
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    assert getattr(y, "size", 0) > 0, (
        f"Reference audio at {INPUT_WAV} is empty."
    )
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return {"y": y, "sr": sr, "onset_env": onset_env, "np": np, "librosa": librosa}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (isinstance(value, (int, float))
            and not isinstance(value, bool))


def test_each_candidate_has_required_schema(candidates):
    for idx, cand in enumerate(candidates):
        assert isinstance(cand, dict), (
            f"Candidate {idx} is not a JSON object, got: {type(cand).__name__}."
        )
        for key in ("tempo_bpm", "salience", "harmonic_rank"):
            assert key in cand, (
                f"Candidate {idx} is missing required key '{key}': {cand!r}."
            )
        assert _is_number(cand["tempo_bpm"]), (
            f"Candidate {idx} 'tempo_bpm' must be numeric, got: {cand['tempo_bpm']!r}."
        )
        assert _is_number(cand["salience"]), (
            f"Candidate {idx} 'salience' must be numeric, got: {cand['salience']!r}."
        )
        assert _is_int(cand["harmonic_rank"]), (
            f"Candidate {idx} 'harmonic_rank' must be an integer, got: "
            f"{cand['harmonic_rank']!r}."
        )


def test_tempo_bpm_within_allowed_range(candidates):
    for idx, cand in enumerate(candidates):
        bpm = float(cand["tempo_bpm"])
        assert 40.0 <= bpm <= 240.0, (
            f"Candidate {idx} 'tempo_bpm'={bpm} must lie within [40.0, 240.0]."
        )


def test_salience_non_negative_and_non_increasing(candidates):
    saliences = [float(c["salience"]) for c in candidates]
    for idx, s in enumerate(saliences):
        assert s >= 0.0, (
            f"Candidate {idx} 'salience'={s} must be non-negative."
        )
    for i in range(len(saliences) - 1):
        assert saliences[i] >= saliences[i + 1] - 1e-12, (
            f"Saliences must be non-increasing; got "
            f"index {i}={saliences[i]} < index {i+1}={saliences[i+1]}."
        )


def test_harmonic_rank_is_permutation_of_1_2_3(candidates):
    ranks = [c["harmonic_rank"] for c in candidates]
    for idx, r in enumerate(ranks):
        assert _is_int(r) and r > 0, (
            f"Candidate {idx} 'harmonic_rank'={r!r} must be a positive integer."
        )
    assert sorted(ranks) == [1, 2, 3], (
        f"'harmonic_rank' values must be a permutation of [1, 2, 3], got: {ranks}."
    )


def test_at_least_one_candidate_matches_reference_tempo(candidates, reference_signals):
    np = reference_signals["np"]
    librosa = reference_signals["librosa"]
    sr = reference_signals["sr"]
    onset_env = reference_signals["onset_env"]

    ref_tempo_arr = librosa.feature.rhythm.tempo(
        onset_envelope=onset_env, sr=sr, aggregate=np.mean
    )
    ref_tempo = float(np.asarray(ref_tempo_arr).reshape(-1)[0])
    diffs = [abs(float(c["tempo_bpm"]) - ref_tempo) for c in candidates]
    min_diff = min(diffs)
    assert min_diff <= 3.0 + 1e-6, (
        f"At least one candidate must be within \u00b13 BPM of the reference "
        f"tempo {ref_tempo:.3f}; min diff was {min_diff:.3f}. "
        f"Candidates: {[c['tempo_bpm'] for c in candidates]}."
    )


def test_top_candidate_matches_beat_track_tempo(candidates, reference_signals):
    np = reference_signals["np"]
    librosa = reference_signals["librosa"]
    y = reference_signals["y"]
    sr = reference_signals["sr"]

    beat_tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
    beat_tempo = float(np.asarray(beat_tempo_raw).reshape(-1)[0])
    top_bpm = float(candidates[0]["tempo_bpm"])
    diff = abs(top_bpm - beat_tempo)
    assert diff <= 5.0 + 1e-6, (
        f"Top candidate tempo {top_bpm:.3f} must be within \u00b15 BPM of "
        f"beat_track tempo {beat_tempo:.3f}; diff={diff:.3f}."
    )
