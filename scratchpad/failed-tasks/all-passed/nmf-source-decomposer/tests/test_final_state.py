import json
import os

import numpy as np
import pytest


WORKSPACE = "/workspace"
MIXTURE_WAV = "/workspace/mixture.wav"
COMPONENTS_JSON = "/workspace/components.json"
COMPONENT_WAVS = [f"/workspace/component_{i}.wav" for i in range(4)]


@pytest.fixture(scope="module")
def mixture():
    import librosa

    assert os.path.isfile(MIXTURE_WAV), (
        f"Expected reference mixture file {MIXTURE_WAV} to exist."
    )
    y, sr = librosa.load(MIXTURE_WAV, sr=None, mono=True)
    assert y.size > 0, f"Reference mixture {MIXTURE_WAV} is empty."
    assert sr and sr > 0, f"Invalid mixture sample rate: {sr!r}."
    mix_rms = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    assert mix_rms > 0, f"Reference mixture RMS must be positive, got {mix_rms}."
    return {"y": np.asarray(y, dtype=np.float64), "sr": int(sr), "N": int(len(y)), "rms": mix_rms}


@pytest.fixture(scope="module")
def component_waveforms(mixture):
    import librosa

    waveforms = []
    for idx, path in enumerate(COMPONENT_WAVS):
        assert os.path.isfile(path), (
            f"Expected component output file {path} to exist after the task completes."
        )
        y, sr = librosa.load(path, sr=None, mono=True)
        assert getattr(y, "ndim", 0) == 1, (
            f"Component file {path} must contain mono (1-D) audio, got ndim={getattr(y, 'ndim', None)}."
        )
        assert int(sr) == mixture["sr"], (
            f"Component file {path} sample rate {sr} does not match mixture sample rate {mixture['sr']}."
        )
        assert abs(int(len(y)) - mixture["N"]) <= 1, (
            f"Component file {path} sample count {len(y)} differs from mixture sample count {mixture['N']} by more than 1."
        )
        waveforms.append(np.asarray(y, dtype=np.float64))
    return waveforms


@pytest.fixture(scope="module")
def components_meta():
    assert os.path.isfile(COMPONENTS_JSON), (
        f"Expected output metadata file {COMPONENTS_JSON} to exist after the task completes."
    )
    with open(COMPONENTS_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{COMPONENTS_JSON} is not valid JSON: {exc}")
    assert isinstance(data, list), (
        f"{COMPONENTS_JSON} must contain a JSON array, got: {type(data).__name__}."
    )
    assert len(data) == 4, (
        f"{COMPONENTS_JSON} must contain exactly 4 component entries, got {len(data)}."
    )
    return data


def test_components_json_schema(components_meta):
    seen_indices = []
    for pos, entry in enumerate(components_meta):
        assert isinstance(entry, dict), (
            f"Entry {pos} of components.json must be a JSON object, got: {type(entry).__name__}."
        )
        for key in ("index", "centroid_hz", "rms"):
            assert key in entry, (
                f"Entry {pos} of components.json is missing required key '{key}': {entry!r}."
            )
        idx = entry["index"]
        assert isinstance(idx, int) and not isinstance(idx, bool), (
            f"Entry {pos} 'index' must be an int, got: {idx!r}."
        )
        seen_indices.append(idx)
        assert isinstance(entry["centroid_hz"], (int, float)) and not isinstance(entry["centroid_hz"], bool), (
            f"Entry {pos} 'centroid_hz' must be numeric, got: {entry['centroid_hz']!r}."
        )
        assert isinstance(entry["rms"], (int, float)) and not isinstance(entry["rms"], bool), (
            f"Entry {pos} 'rms' must be numeric, got: {entry['rms']!r}."
        )
    assert sorted(seen_indices) == [0, 1, 2, 3], (
        f"components.json 'index' values must be exactly the multiset {{0,1,2,3}}, got {sorted(seen_indices)}."
    )


def test_components_centroids_within_nyquist(components_meta, mixture):
    nyquist = mixture["sr"] / 2.0
    for entry in components_meta:
        c = float(entry["centroid_hz"])
        assert 0.0 < c < nyquist, (
            f"Component index={entry['index']} centroid_hz={c} must lie strictly in (0, {nyquist})."
        )


def test_components_centroid_spread(components_meta):
    centroids = [float(entry["centroid_hz"]) for entry in components_meta]
    spread = max(centroids) - min(centroids)
    assert spread > 200.0, (
        f"Spectral centroid spread across components must exceed 200 Hz, got {spread} Hz "
        f"with centroids={centroids}."
    )


def test_components_rms_nonzero(components_meta):
    for entry in components_meta:
        r = float(entry["rms"])
        assert r > 1e-4, (
            f"Component index={entry['index']} rms={r} must exceed 1e-4 (no silent components allowed)."
        )


def test_reconstruction_matches_mixture(component_waveforms, mixture):
    L = min(int(mixture["N"]), min(int(len(w)) for w in component_waveforms))
    summed = np.zeros(L, dtype=np.float64)
    for w in component_waveforms:
        summed += w[:L]
    mae = float(np.mean(np.abs(summed - mixture["y"][:L])))
    tolerance = 5e-2 * mixture["rms"]
    assert mae <= tolerance, (
        f"Sum of component waveforms must reconstruct the mixture: "
        f"mean absolute error {mae} exceeds tolerance {tolerance} "
        f"(5e-2 * mixture RMS {mixture['rms']})."
    )
