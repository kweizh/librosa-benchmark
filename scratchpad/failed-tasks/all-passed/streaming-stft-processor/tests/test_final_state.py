import csv
import os

import numpy as np
import pytest


WORKSPACE = "/workspace"
INPUT_PATH = os.path.join(WORKSPACE, "input.wav")
OUTPUT_PATH = os.path.join(WORKSPACE, "rms_stream.csv")
EXPECTED_HEADER = ["block_index", "start_time_sec", "end_time_sec", "rms"]


def _read_rows():
    with open(OUTPUT_PATH, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows, f"{OUTPUT_PATH} is empty."
    header = rows[0]
    data = rows[1:]
    return header, data


def test_output_csv_exists():
    assert os.path.isfile(OUTPUT_PATH), (
        f"Expected output CSV at {OUTPUT_PATH}, but it does not exist."
    )


def test_csv_header_and_row_count():
    header, data = _read_rows()
    assert header == EXPECTED_HEADER, (
        f"CSV header must be exactly {EXPECTED_HEADER}, got {header}."
    )
    assert len(data) >= 2, (
        f"CSV must contain at least 2 data rows, found {len(data)}."
    )


def test_block_index_starts_at_zero_and_strictly_increases():
    _, data = _read_rows()
    indices = [int(row[0]) for row in data]
    assert indices[0] == 0, (
        f"First block_index must be 0, got {indices[0]}."
    )
    for prev, curr in zip(indices, indices[1:]):
        assert curr > prev, (
            f"block_index column must be strictly increasing; "
            f"got {prev} followed by {curr}."
        )


def test_time_columns_are_monotonic_and_start_at_zero():
    _, data = _read_rows()
    starts = [float(row[1]) for row in data]
    ends = [float(row[2]) for row in data]
    assert abs(starts[0]) < 1e-3, (
        f"First start_time_sec must be within 0.001 s of 0, got {starts[0]}."
    )
    for prev, curr in zip(starts, starts[1:]):
        assert curr > prev, (
            f"start_time_sec must be strictly increasing; "
            f"got {prev} followed by {curr}."
        )
    for prev, curr in zip(ends, ends[1:]):
        assert curr > prev, (
            f"end_time_sec must be strictly increasing; "
            f"got {prev} followed by {curr}."
        )
    for s, e in zip(starts, ends):
        assert e > s, (
            f"Each row must satisfy end_time_sec > start_time_sec; "
            f"got start={s}, end={e}."
        )


def test_weighted_rms_matches_global_rms():
    import librosa

    _, data = _read_rows()
    starts = np.array([float(row[1]) for row in data], dtype=np.float64)
    ends = np.array([float(row[2]) for row in data], dtype=np.float64)
    rms_vals = np.array([float(row[3]) for row in data], dtype=np.float64)
    durations = ends - starts

    assert np.all(durations > 0), (
        "All block durations (end_time_sec - start_time_sec) must be positive."
    )
    assert np.all(rms_vals >= 0), (
        "RMS values must be non-negative."
    )

    weighted_power = float(np.sum(rms_vals ** 2 * durations) / np.sum(durations))
    weighted_rms = float(np.sqrt(weighted_power))

    y, sr = librosa.load(INPUT_PATH, sr=None, mono=True)
    global_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))

    assert global_rms > 0, (
        f"Global RMS of {INPUT_PATH} is 0, cannot compare ratios."
    )
    rel_err = abs(weighted_rms - global_rms) / global_rms
    assert rel_err < 0.05, (
        f"Duration-weighted RMS reconstructed from the CSV "
        f"({weighted_rms:.6f}) deviates by {rel_err * 100:.2f}% from the "
        f"global RMS ({global_rms:.6f}); expected within 5%."
    )


def test_last_end_time_close_to_file_duration():
    import librosa

    _, data = _read_rows()
    last_end = float(data[-1][2])
    duration = float(librosa.get_duration(path=INPUT_PATH))
    diff = abs(last_end - duration)
    assert diff < 0.5, (
        f"Last end_time_sec ({last_end:.3f}) must be within 0.5 s of file "
        f"duration ({duration:.3f}); difference was {diff:.3f}."
    )
