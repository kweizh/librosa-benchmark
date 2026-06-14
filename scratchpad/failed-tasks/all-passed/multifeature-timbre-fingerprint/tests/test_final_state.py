import json
import math
import os

import numpy as np
import pytest


FEATURES_JSON = "/workspace/features.json"
INPUT_WAV = "/workspace/input.wav"
CENTROIDS_JSON = "/workspace/centroids.json"

EXPECTED_FEATURE_ORDER = [
    "mfcc_mean",
    "mfcc_std",
    "chroma_mean",
    "chroma_std",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "rolloff_std",
    "zcr_mean",
    "zcr_std",
    "contrast_mean",
    "contrast_std",
]
EXPECTED_LABELS = {"rock", "classical", "jazz"}
EXPECTED_TOP_LEVEL_KEYS = {"vector", "feature_order", "similarities", "predicted_label"}


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _reference_vector():
    import librosa

    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    blocks = [
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(centroid, axis=1),
        np.std(centroid, axis=1),
        np.mean(bandwidth, axis=1),
        np.std(bandwidth, axis=1),
        np.mean(rolloff, axis=1),
        np.std(rolloff, axis=1),
        np.mean(zcr, axis=1),
        np.std(zcr, axis=1),
        np.mean(contrast, axis=1),
        np.std(contrast, axis=1),
    ]
    return np.concatenate([np.asarray(b, dtype=np.float64).reshape(-1) for b in blocks])


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.fixture(scope="module")
def features():
    assert os.path.isfile(FEATURES_JSON), (
        f"Expected output file {FEATURES_JSON} to exist after the task completes."
    )
    with open(FEATURES_JSON, "r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{FEATURES_JSON} is not valid JSON: {exc}"
            )
    assert isinstance(data, dict), (
        f"{FEATURES_JSON} must be a JSON object, got: {type(data).__name__}."
    )
    return data


@pytest.fixture(scope="module")
def reference_vector():
    return _reference_vector()


@pytest.fixture(scope="module")
def centroids():
    with open(CENTROIDS_JSON, "r") as fh:
        data = json.load(fh)
    return {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}


def test_top_level_keys(features):
    assert set(features.keys()) == EXPECTED_TOP_LEVEL_KEYS, (
        f"features.json must contain exactly the keys {sorted(EXPECTED_TOP_LEVEL_KEYS)}, "
        f"got: {sorted(features.keys())}."
    )


def test_feature_order_exact(features):
    assert features["feature_order"] == EXPECTED_FEATURE_ORDER, (
        f"feature_order must equal {EXPECTED_FEATURE_ORDER}, got: {features['feature_order']!r}."
    )


def test_vector_shape_and_finite(features):
    vector = features["vector"]
    assert isinstance(vector, list), (
        f"vector must be a list, got: {type(vector).__name__}."
    )
    assert len(vector) == 72, (
        f"vector must have length 72, got {len(vector)}."
    )
    for idx, value in enumerate(vector):
        assert _is_finite_number(value), (
            f"vector[{idx}] must be a finite float, got: {value!r}."
        )


def test_similarities_keys_and_range(features):
    sims = features["similarities"]
    assert isinstance(sims, dict), (
        f"similarities must be a JSON object, got: {type(sims).__name__}."
    )
    assert set(sims.keys()) == EXPECTED_LABELS, (
        f"similarities keys must be exactly {sorted(EXPECTED_LABELS)}, got: {sorted(sims.keys())}."
    )
    for label, value in sims.items():
        assert _is_finite_number(value), (
            f"similarities[{label!r}] must be a finite float, got: {value!r}."
        )
        v = float(value)
        assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6, (
            f"similarities[{label!r}]={v} is outside the valid cosine range [-1, 1]."
        )


def test_vector_matches_reference(features, reference_vector):
    submitted = np.asarray(features["vector"], dtype=np.float64)
    assert submitted.shape == reference_vector.shape, (
        f"vector shape {submitted.shape} does not match reference shape {reference_vector.shape}."
    )
    assert np.allclose(submitted, reference_vector, rtol=1e-4, atol=1e-4), (
        f"Submitted vector does not match the reference fingerprint within rtol=1e-4, atol=1e-4. "
        f"Max absolute diff = {float(np.max(np.abs(submitted - reference_vector)))}."
    )


def test_predicted_label_is_argmax(features, reference_vector, centroids):
    ref_sims = {label: _cosine(reference_vector, vec) for label, vec in centroids.items()}
    max_sim = max(ref_sims.values())
    tied_labels = sorted([label for label, s in ref_sims.items() if s == max_sim])
    expected_label = tied_labels[0]
    predicted = features["predicted_label"]
    assert isinstance(predicted, str), (
        f"predicted_label must be a string, got: {type(predicted).__name__}."
    )
    assert predicted in EXPECTED_LABELS, (
        f"predicted_label {predicted!r} is not one of {sorted(EXPECTED_LABELS)}."
    )
    submitted_sim = float(features["similarities"][predicted])
    expected_sim = ref_sims[expected_label]
    # Allow tie-breaking against the reference: accept any label whose reference
    # similarity is within numerical tolerance of the maximum.
    assert math.isclose(submitted_sim, expected_sim, rel_tol=1e-4, abs_tol=1e-4) or (
        ref_sims[predicted] >= max_sim - 1e-6
    ), (
        f"predicted_label={predicted!r} does not correspond to the argmax of the reference "
        f"similarities {ref_sims}; expected one of the lexicographically-smallest tied "
        f"labels {tied_labels}."
    )
