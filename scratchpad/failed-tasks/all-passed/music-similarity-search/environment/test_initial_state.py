import glob
import importlib
import os


WORKSPACE = "/workspace"
LIBRARY_DIR = os.path.join(WORKSPACE, "library")
QUERY_PATH = os.path.join(WORKSPACE, "query.wav")
GROUND_TRUTH_PATH = os.path.join(WORKSPACE, ".ground_truth_match")


def test_librosa_importable():
    module = importlib.import_module("librosa")
    assert module is not None, "librosa is not importable in the environment."


def test_librosa_version_is_0_11_0():
    import librosa

    assert librosa.__version__ == "0.11.0", (
        f"Expected librosa==0.11.0 but found {librosa.__version__}."
    )


def test_soundfile_importable():
    module = importlib.import_module("soundfile")
    assert module is not None, "soundfile is not importable in the environment."


def test_numpy_importable():
    module = importlib.import_module("numpy")
    assert module is not None, "numpy is not importable in the environment."


def test_workspace_dir_exists():
    assert os.path.isdir(WORKSPACE), f"Workspace directory {WORKSPACE} does not exist."


def test_library_dir_exists():
    assert os.path.isdir(LIBRARY_DIR), (
        f"Library directory {LIBRARY_DIR} does not exist."
    )


def test_library_has_wav_files():
    wavs = sorted(glob.glob(os.path.join(LIBRARY_DIR, "*.wav")))
    assert len(wavs) >= 2, (
        f"Library directory must contain at least 2 WAV files, found {len(wavs)}."
    )


def test_query_file_exists():
    assert os.path.isfile(QUERY_PATH), f"Query file {QUERY_PATH} does not exist."


def test_query_is_readable_wav():
    import soundfile as sf

    info = sf.info(QUERY_PATH)
    assert info.frames > 0, f"Query file {QUERY_PATH} contains no audio frames."


def test_ground_truth_file_exists():
    assert os.path.isfile(GROUND_TRUTH_PATH), (
        f"Ground-truth match file {GROUND_TRUTH_PATH} is missing."
    )


def test_ground_truth_points_to_library_file():
    with open(GROUND_TRUTH_PATH) as f:
        gt = f.read().strip()
    assert gt, "Ground-truth match file is empty."
    candidate = os.path.join(LIBRARY_DIR, gt)
    assert os.path.isfile(candidate), (
        f"Ground-truth basename {gt!r} does not exist in {LIBRARY_DIR}."
    )
