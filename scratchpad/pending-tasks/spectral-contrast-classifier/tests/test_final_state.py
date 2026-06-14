import csv
import os

import pytest


WORKSPACE = "/workspace"
TEST_DIR = os.path.join(WORKSPACE, "test")
PREDICTIONS_PATH = os.path.join(WORKSPACE, "predictions.csv")

VALID_LABELS = {"speech", "music"}

# Ground-truth labels for the deterministic test set baked into the Docker image.
GROUND_TRUTH = {
    "speech_test_0.wav": "speech",
    "speech_test_1.wav": "speech",
    "music_test_0.wav": "music",
    "music_test_1.wav": "music",
}

MIN_ACCURACY = 0.80


def _read_predictions(path):
    """Return (header, rows) where rows is a list of dicts keyed by header columns."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None, []
        rows = [row for row in reader if row]  # skip blank lines
    return header, rows


@pytest.fixture(scope="module")
def predictions():
    assert os.path.isfile(PREDICTIONS_PATH), (
        f"Predictions file {PREDICTIONS_PATH} was not created by the executor."
    )
    header, rows = _read_predictions(PREDICTIONS_PATH)
    assert header is not None, (
        f"Predictions file {PREDICTIONS_PATH} is empty (no header line)."
    )
    return header, rows


def test_predictions_file_exists():
    assert os.path.isfile(PREDICTIONS_PATH), (
        f"Expected predictions file at {PREDICTIONS_PATH}."
    )


def test_predictions_header_is_exact(predictions):
    header, _ = predictions
    assert header == ["filename", "label"], (
        f"Predictions CSV header must be exactly ['filename', 'label']; got {header!r}."
    )


def test_predictions_rows_have_two_columns(predictions):
    _, rows = predictions
    for i, row in enumerate(rows, start=2):  # data rows start on line 2
        assert len(row) == 2, (
            f"Row {i} of {PREDICTIONS_PATH} has {len(row)} columns, expected 2; row={row!r}."
        )


def test_predictions_cover_exactly_test_set(predictions):
    _, rows = predictions
    predicted_files = [row[0] for row in rows]
    predicted_set = set(predicted_files)

    test_files_on_disk = {
        name for name in os.listdir(TEST_DIR) if name.lower().endswith(".wav")
    }

    missing = test_files_on_disk - predicted_set
    extra = predicted_set - test_files_on_disk
    assert not missing, (
        f"Predictions CSV is missing rows for these test files: {sorted(missing)}."
    )
    assert not extra, (
        f"Predictions CSV contains rows for files not in {TEST_DIR}: {sorted(extra)}."
    )


def test_no_duplicate_filenames(predictions):
    _, rows = predictions
    seen = {}
    for i, row in enumerate(rows, start=2):
        fname = row[0]
        if fname in seen:
            pytest.fail(
                f"Duplicate filename {fname!r} in {PREDICTIONS_PATH}: "
                f"first seen on line {seen[fname]}, again on line {i}."
            )
        seen[fname] = i


def test_all_labels_are_valid(predictions):
    _, rows = predictions
    for i, row in enumerate(rows, start=2):
        label = row[1]
        assert label in VALID_LABELS, (
            f"Row {i} of {PREDICTIONS_PATH} has invalid label {label!r}; "
            f"expected one of {sorted(VALID_LABELS)}."
        )


def test_classification_accuracy_meets_threshold(predictions):
    _, rows = predictions
    predicted = {row[0]: row[1] for row in rows}

    # Restrict scoring to filenames we have ground truth for. The other coverage
    # tests already guarantee that the predictions exactly cover the test set,
    # which itself equals GROUND_TRUTH.keys() because the Dockerfile generated it.
    relevant = [name for name in GROUND_TRUTH if name in predicted]
    assert relevant, (
        "No predicted rows match the ground-truth test set; cannot score accuracy."
    )

    correct = sum(1 for name in relevant if predicted[name] == GROUND_TRUTH[name])
    accuracy = correct / len(GROUND_TRUTH)

    assert accuracy >= MIN_ACCURACY, (
        f"Classification accuracy {accuracy:.2f} is below the required "
        f"threshold of {MIN_ACCURACY:.2f}. Per-file predictions: "
        f"{[(n, predicted.get(n), GROUND_TRUTH[n]) for n in GROUND_TRUTH]}."
    )
