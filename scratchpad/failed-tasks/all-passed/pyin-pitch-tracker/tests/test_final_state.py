import csv
import math
import os

import pytest


WORKSPACE = "/workspace"
OUTPUT_CSV = os.path.join(WORKSPACE, "pitch.csv")

# Ground-truth sweep parameters baked into the Dockerfile-synthesized input.wav.
# Linear sweep from F_START Hz at t=0 to F_END Hz at t=SWEEP_DURATION_SEC,
# sampled at SAMPLE_RATE Hz, written to /workspace/input.wav.
SAMPLE_RATE = 22050
SWEEP_DURATION_SEC = 5.0
F_START = 200.0
F_END = 800.0

# A frame-tolerance for the very first time stamp. pyin defaults to
# frame_length=2048, hop_length=frame_length//4=512, so one frame at sr=22050
# is ~23.2 ms. We allow at most one full default frame_length of tolerance.
FIRST_TIME_TOLERANCE_SEC = 2048.0 / SAMPLE_RATE

VALID_TRUE_TOKENS = {"true", "True", "TRUE", "1"}
VALID_FALSE_TOKENS = {"false", "False", "FALSE", "0"}


def _ground_truth_freq(t: float) -> float:
    """Instantaneous frequency of a linear sweep clamped to the sweep window."""
    t_clamped = max(0.0, min(SWEEP_DURATION_SEC, t))
    return F_START + (F_END - F_START) * (t_clamped / SWEEP_DURATION_SEC)


def _read_csv_rows():
    with open(OUTPUT_CSV, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows


def _parse_voiced(token: str) -> bool:
    t = token.strip()
    if t in VALID_TRUE_TOKENS:
        return True
    if t in VALID_FALSE_TOKENS:
        return False
    raise ValueError(f"unrecognized voiced token: {token!r}")


def _parse_freq(token: str):
    t = token.strip()
    if t == "" or t.lower() == "nan":
        return float("nan")
    return float(t)


@pytest.fixture(scope="module")
def csv_rows():
    assert os.path.isfile(OUTPUT_CSV), (
        f"Expected output CSV at {OUTPUT_CSV}, but it does not exist."
    )
    rows = _read_csv_rows()
    assert len(rows) >= 1, f"{OUTPUT_CSV} is empty (no header row)."
    return rows


@pytest.fixture(scope="module")
def parsed(csv_rows):
    header = [h.strip() for h in csv_rows[0]]
    data = csv_rows[1:]
    return header, data


def test_output_csv_exists():
    assert os.path.isfile(OUTPUT_CSV), (
        f"Expected output CSV at {OUTPUT_CSV}, but it does not exist."
    )


def test_csv_has_required_header(parsed):
    header, _ = parsed
    expected = ["time_sec", "frequency_hz", "voiced"]
    assert header == expected, (
        f"Expected CSV header {expected}, got {header}."
    )


def test_csv_has_minimum_row_count(parsed):
    _, data = parsed
    assert len(data) >= 50, (
        f"Expected at least 50 data rows in {OUTPUT_CSV}, got {len(data)}."
    )


def test_csv_rows_have_three_columns(parsed):
    _, data = parsed
    for i, row in enumerate(data):
        assert len(row) == 3, (
            f"Row {i} of {OUTPUT_CSV} has {len(row)} columns, expected 3 (got {row!r})."
        )


def test_time_sec_starts_near_zero_and_is_monotonic(parsed):
    _, data = parsed
    assert len(data) >= 2, "Need at least two rows to verify time monotonicity."
    times = [float(row[0]) for row in data]
    assert times[0] >= -1e-9, f"First time_sec is negative: {times[0]}."
    assert times[0] <= FIRST_TIME_TOLERANCE_SEC, (
        f"First time_sec {times[0]:.6f} is not within {FIRST_TIME_TOLERANCE_SEC:.6f}s of 0."
    )
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1] - 1e-9, (
            f"time_sec is not monotonically non-decreasing at row {i}: "
            f"{times[i - 1]} -> {times[i]}."
        )


def test_voiced_tokens_are_consistent_and_recognized(parsed):
    _, data = parsed
    tokens_seen = {row[2].strip() for row in data}
    assert tokens_seen, "No voiced tokens were parsed."
    if tokens_seen.issubset(VALID_TRUE_TOKENS | VALID_FALSE_TOKENS):
        # Check style consistency: either {true/false}, {True/False}, {TRUE/FALSE}, or {1/0}.
        styles = [
            {"true", "false"},
            {"True", "False"},
            {"TRUE", "FALSE"},
            {"1", "0"},
        ]
        ok = any(tokens_seen.issubset(s) for s in styles)
        assert ok, (
            f"voiced column mixes incompatible encodings: {sorted(tokens_seen)}."
        )
    else:
        unknown = tokens_seen - (VALID_TRUE_TOKENS | VALID_FALSE_TOKENS)
        pytest.fail(f"voiced column contains unrecognized tokens: {sorted(unknown)}.")


def test_voiced_fraction_within_sweep(parsed):
    _, data = parsed
    in_window = 0
    voiced_in_window = 0
    for row in data:
        t = float(row[0])
        if 0.0 <= t <= SWEEP_DURATION_SEC:
            in_window += 1
            if _parse_voiced(row[2]):
                voiced_in_window += 1
    assert in_window > 0, "No CSV rows fall inside the [0, 5s] sweep window."
    fraction = voiced_in_window / in_window
    assert fraction >= 0.8, (
        f"Only {fraction:.2%} of frames in the sweep window are voiced; expected >= 80%."
    )


def test_voiced_frequencies_are_in_range(parsed):
    _, data = parsed
    for i, row in enumerate(data):
        if not _parse_voiced(row[2]):
            continue
        freq = _parse_freq(row[1])
        assert math.isfinite(freq), (
            f"Row {i} is marked voiced but frequency_hz is not finite: {row[1]!r}."
        )
        assert 50.0 <= freq <= 2000.0, (
            f"Row {i} voiced frequency {freq} Hz is outside [50, 2000] Hz."
        )


def test_unvoiced_frequencies_are_nan_empty_or_zero(parsed):
    _, data = parsed
    for i, row in enumerate(data):
        if _parse_voiced(row[2]):
            continue
        token = row[1].strip()
        if token == "" or token.lower() == "nan":
            continue
        try:
            value = float(token)
        except ValueError:
            pytest.fail(
                f"Row {i} is unvoiced but frequency_hz {token!r} is not NaN, empty, or numeric 0."
            )
        assert value == 0.0 or math.isnan(value), (
            f"Row {i} is unvoiced but frequency_hz={value} is neither 0 nor NaN."
        )


def test_voiced_frequency_median_error_below_5pct(parsed):
    _, data = parsed
    errors = []
    for row in data:
        t = float(row[0])
        if not (0.0 <= t <= SWEEP_DURATION_SEC):
            continue
        if not _parse_voiced(row[2]):
            continue
        f_pred = _parse_freq(row[1])
        if not math.isfinite(f_pred):
            continue
        f_true = _ground_truth_freq(t)
        if f_true <= 0:
            continue
        errors.append(abs(f_pred - f_true) / f_true)
    assert len(errors) >= 25, (
        f"Not enough voiced frames inside the sweep window to evaluate accuracy "
        f"(got {len(errors)}, expected >= 25)."
    )
    errors.sort()
    n = len(errors)
    if n % 2 == 1:
        median = errors[n // 2]
    else:
        median = 0.5 * (errors[n // 2 - 1] + errors[n // 2])
    assert median < 0.05, (
        f"Median relative pitch error {median:.4f} exceeds the 5% threshold."
    )
