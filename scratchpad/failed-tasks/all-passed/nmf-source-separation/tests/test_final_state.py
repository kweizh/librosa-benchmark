import os

import numpy as np
import pytest
import soundfile as sf

WORKSPACE_DIR = "/workspace"
INPUT_AUDIO = os.path.join(WORKSPACE_DIR, "input.wav")
COMPONENT_PATHS = [
    os.path.join(WORKSPACE_DIR, f"component_{i}.wav") for i in range(4)
]
LENGTH_TOLERANCE_SAMPLES = 2048
COSINE_SIMILARITY_THRESHOLD = 0.95
RMS_FLOOR = 1e-5
CENTROID_DIFF_HZ = 50.0


def _load_mono(path):
    data, sr = sf.read(path, always_2d=False)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


@pytest.fixture(scope="module")
def original_audio():
    assert os.path.isfile(INPUT_AUDIO), (
        f"Original input audio {INPUT_AUDIO} is missing; cannot run verification."
    )
    return _load_mono(INPUT_AUDIO)


@pytest.fixture(scope="module")
def components(original_audio):
    loaded = []
    for path in COMPONENT_PATHS:
        assert os.path.isfile(path), (
            f"Expected component output file {path} does not exist."
        )
        try:
            data, sr = _load_mono(path)
        except Exception as exc:
            pytest.fail(f"Could not read component output {path}: {exc}")
        loaded.append((path, data, sr))
    return loaded


def test_all_component_files_exist_and_load(components):
    # The fixture asserts existence and successful load; this test simply
    # ensures the fixture ran for all 4 components.
    assert len(components) == 4, (
        f"Expected 4 component output files, found {len(components)}."
    )


def test_each_component_matches_input_sample_rate(original_audio, components):
    _, sr_orig = original_audio
    for path, _, sr in components:
        assert sr == sr_orig, (
            f"Sample rate mismatch in {path}: expected {sr_orig} Hz, got {sr} Hz."
        )


def test_each_component_length_within_tolerance(original_audio, components):
    y_orig, _ = original_audio
    n_orig = len(y_orig)
    for path, data, _ in components:
        diff = abs(len(data) - n_orig)
        assert diff <= LENGTH_TOLERANCE_SAMPLES, (
            f"Component {path} length ({len(data)}) differs from input length "
            f"({n_orig}) by {diff} samples, which exceeds the allowed "
            f"tolerance of {LENGTH_TOLERANCE_SAMPLES} samples."
        )


def _align_to_length(arr, n):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    out = np.zeros(n, dtype=np.float64)
    out[: len(arr)] = arr
    return out


def test_sum_of_components_is_close_to_input(original_audio, components):
    y_orig, _ = original_audio
    n = len(y_orig)
    y_sum = np.zeros(n, dtype=np.float64)
    for _, data, _ in components:
        y_sum += _align_to_length(data, n)

    y_orig_d = np.asarray(y_orig, dtype=np.float64)
    norm_orig = np.linalg.norm(y_orig_d)
    norm_sum = np.linalg.norm(y_sum)
    assert norm_orig > 0, "Original input audio is all zeros; cannot verify."
    assert norm_sum > 0, (
        "Element-wise sum of component waveforms is all zeros, which cannot "
        "approximate the input mixture."
    )
    cosine = float(np.dot(y_sum, y_orig_d) / (norm_sum * norm_orig))
    assert cosine >= COSINE_SIMILARITY_THRESHOLD, (
        f"Cosine similarity between the sum of component waveforms and the "
        f"original input is {cosine:.4f}, which is below the required "
        f"threshold of {COSINE_SIMILARITY_THRESHOLD}."
    )


def test_each_component_has_audible_energy(components):
    for path, data, _ in components:
        arr = np.asarray(data, dtype=np.float64)
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size > 0 else 0.0
        assert rms > RMS_FLOOR, (
            f"Component {path} has RMS amplitude {rms:.3e}, which is at or "
            f"below the silence threshold {RMS_FLOOR:.0e}."
        )


def test_components_have_distinguishable_spectral_content(components):
    import librosa

    centroids = []
    for path, data, sr in components:
        arr = np.asarray(data, dtype=np.float32)
        if arr.size == 0:
            pytest.fail(f"Component {path} is empty; cannot compute spectral centroid.")
        centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=arr, sr=sr))
        )
        centroids.append((path, centroid))

    found_pair = False
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            if abs(centroids[i][1] - centroids[j][1]) > CENTROID_DIFF_HZ:
                found_pair = True
                break
        if found_pair:
            break

    summary = ", ".join(f"{os.path.basename(p)}={c:.1f}Hz" for p, c in centroids)
    assert found_pair, (
        f"Components do not have distinguishable spectral centroids; "
        f"no pair differs by more than {CENTROID_DIFF_HZ} Hz. "
        f"Observed centroids: {summary}."
    )
