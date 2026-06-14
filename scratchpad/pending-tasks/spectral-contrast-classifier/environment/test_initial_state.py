import os

import pytest


WORKSPACE = "/workspace"
TRAIN_DIR = os.path.join(WORKSPACE, "train")
TRAIN_SPEECH_DIR = os.path.join(TRAIN_DIR, "speech")
TRAIN_MUSIC_DIR = os.path.join(TRAIN_DIR, "music")
TEST_DIR = os.path.join(WORKSPACE, "test")


def test_librosa_importable():
    try:
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import librosa: {exc}")


def test_soundfile_importable():
    try:
        import soundfile  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import soundfile: {exc}")


def test_numpy_importable():
    try:
        import numpy  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import numpy: {exc}")


def test_sklearn_importable():
    try:
        import sklearn  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"Failed to import scikit-learn LogisticRegression: {exc}")


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), (
        f"Expected workspace directory {WORKSPACE} to exist before the task starts."
    )


def test_train_directory_exists():
    assert os.path.isdir(TRAIN_DIR), (
        f"Expected training root directory {TRAIN_DIR} to exist before the task starts."
    )


def test_train_speech_subdir_exists_with_wavs():
    assert os.path.isdir(TRAIN_SPEECH_DIR), (
        f"Expected speech training directory {TRAIN_SPEECH_DIR} to exist."
    )
    wavs = [
        name for name in os.listdir(TRAIN_SPEECH_DIR)
        if name.lower().endswith(".wav")
    ]
    assert len(wavs) >= 2, (
        f"Expected at least 2 WAV files in {TRAIN_SPEECH_DIR}, found {len(wavs)}."
    )


def test_train_music_subdir_exists_with_wavs():
    assert os.path.isdir(TRAIN_MUSIC_DIR), (
        f"Expected music training directory {TRAIN_MUSIC_DIR} to exist."
    )
    wavs = [
        name for name in os.listdir(TRAIN_MUSIC_DIR)
        if name.lower().endswith(".wav")
    ]
    assert len(wavs) >= 2, (
        f"Expected at least 2 WAV files in {TRAIN_MUSIC_DIR}, found {len(wavs)}."
    )


def test_test_directory_exists_with_wavs():
    assert os.path.isdir(TEST_DIR), (
        f"Expected test directory {TEST_DIR} to exist before the task starts."
    )
    wavs = [name for name in os.listdir(TEST_DIR) if name.lower().endswith(".wav")]
    assert len(wavs) >= 2, (
        f"Expected at least 2 WAV files in {TEST_DIR}, found {len(wavs)}."
    )


def test_train_wavs_are_readable():
    import soundfile as sf

    for class_dir in (TRAIN_SPEECH_DIR, TRAIN_MUSIC_DIR):
        for name in sorted(os.listdir(class_dir)):
            if not name.lower().endswith(".wav"):
                continue
            path = os.path.join(class_dir, name)
            try:
                with sf.SoundFile(path) as f:
                    assert f.samplerate > 0, (
                        f"Training WAV {path} has a non-positive sample rate."
                    )
                    assert len(f) > 0, (
                        f"Training WAV {path} contains no audio samples."
                    )
            except Exception as exc:
                pytest.fail(f"Failed to read training WAV {path}: {exc}")


def test_test_wavs_are_readable():
    import soundfile as sf

    for name in sorted(os.listdir(TEST_DIR)):
        if not name.lower().endswith(".wav"):
            continue
        path = os.path.join(TEST_DIR, name)
        try:
            with sf.SoundFile(path) as f:
                assert f.samplerate > 0, (
                    f"Test WAV {path} has a non-positive sample rate."
                )
                assert len(f) > 0, (
                    f"Test WAV {path} contains no audio samples."
                )
        except Exception as exc:
            pytest.fail(f"Failed to read test WAV {path}: {exc}")


def test_predictions_file_not_present_initially():
    predictions_path = os.path.join(WORKSPACE, "predictions.csv")
    assert not os.path.exists(predictions_path), (
        f"Predictions file {predictions_path} should not exist before the executor runs."
    )
