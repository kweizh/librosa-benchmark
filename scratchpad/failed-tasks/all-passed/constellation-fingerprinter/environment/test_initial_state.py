import os

import pytest


WORKSPACE = "/workspace"
DATABASE_WAV = os.path.join(WORKSPACE, "database.wav")
QUERY_WAV = os.path.join(WORKSPACE, "query.wav")
MATCH_JSON = os.path.join(WORKSPACE, "match.json")


def test_librosa_importable():
    try:
        import librosa  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment failure
        pytest.fail(f"librosa is not importable in the task environment: {exc}")


def test_librosa_version_pinned():
    import librosa

    assert librosa.__version__ == "0.11.0", (
        f"Expected librosa 0.11.0 to be installed, but found {librosa.__version__}."
    )


def test_supporting_libraries_importable():
    missing = []
    for mod in ("numpy", "scipy", "scipy.ndimage", "soundfile"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    assert not missing, (
        f"The following supporting Python modules are missing from the task "
        f"environment: {missing}"
    )


def test_workspace_directory_exists():
    assert os.path.isdir(WORKSPACE), (
        f"Workspace directory {WORKSPACE} does not exist."
    )


def test_database_wav_exists():
    assert os.path.isfile(DATABASE_WAV), (
        f"Expected pre-built database file {DATABASE_WAV} to exist."
    )


def test_query_wav_exists():
    assert os.path.isfile(QUERY_WAV), (
        f"Expected pre-built query file {QUERY_WAV} to exist."
    )


def test_database_is_longer_than_query():
    import soundfile as sf

    db_info = sf.info(DATABASE_WAV)
    q_info = sf.info(QUERY_WAV)
    assert db_info.duration > q_info.duration, (
        "database.wav must be longer than query.wav for the matching task "
        f"(database duration={db_info.duration:.2f}s, "
        f"query duration={q_info.duration:.2f}s)."
    )


def test_query_is_approximately_five_seconds():
    import soundfile as sf

    q_info = sf.info(QUERY_WAV)
    assert 4.5 <= q_info.duration <= 5.5, (
        f"query.wav should be ~5 seconds long, but it is {q_info.duration:.2f}s."
    )


def test_match_json_not_yet_present():
    assert not os.path.exists(MATCH_JSON), (
        f"{MATCH_JSON} must not exist before the agent runs; the agent is "
        f"expected to create it."
    )
