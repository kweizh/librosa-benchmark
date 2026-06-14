# Multi-Feature Timbral Fingerprint and Centroid Classification

## Background
Build a fixed-length timbral fingerprint from a short audio clip using `librosa` and classify it against three precomputed reference centroids by cosine similarity. The fingerprint concatenates the temporal mean and standard deviation of multiple complementary spectral and temporal feature streams into a single 72-dimensional vector.

## Requirements
- Load the input WAV from `/workspace/input.wav` (mono, 22050 Hz, ~8 seconds).
- Load the reference centroids from `/workspace/centroids.json`. The file is a JSON object whose keys are the candidate labels `rock`, `classical`, `jazz` and whose values are 72-element float arrays.
- Compute the 72-dimensional fingerprint by concatenating, in the exact order specified below, the per-frame mean and standard deviation (over the time axis) of each feature stream.
- Score the fingerprint against each centroid with cosine similarity and pick the label with the highest similarity.
- Write the result to `/workspace/features.json`.

## Feature Ordering Specification
The 72-dim `vector` is the concatenation of these blocks, in this exact order:

| # | Block | Dim | Feature source |
| - | ----- | --- | -------------- |
| 1 | `mfcc_mean` | 13 | `librosa.feature.mfcc` with `n_mfcc=13` |
| 2 | `mfcc_std`  | 13 | same MFCC matrix, temporal std |
| 3 | `chroma_mean` | 12 | `librosa.feature.chroma_stft` (default `n_chroma=12`) |
| 4 | `chroma_std`  | 12 | same chroma matrix, temporal std |
| 5 | `centroid_mean` | 1 | `librosa.feature.spectral_centroid` |
| 6 | `centroid_std`  | 1 | same |
| 7 | `bandwidth_mean` | 1 | `librosa.feature.spectral_bandwidth` |
| 8 | `bandwidth_std`  | 1 | same |
| 9 | `rolloff_mean` | 1 | `librosa.feature.spectral_rolloff` |
| 10 | `rolloff_std`  | 1 | same |
| 11 | `zcr_mean` | 1 | `librosa.feature.zero_crossing_rate` |
| 12 | `zcr_std`  | 1 | same |
| 13 | `contrast_mean` | 7 | `librosa.feature.spectral_contrast` (default `n_bands=6`, which yields 7 sub-bands) |
| 14 | `contrast_std`  | 7 | same |

Total: 26 + 24 + 8 + 14 = 72.

## Implementation Hints
- All librosa feature extractors must be called with the audio loaded from `/workspace/input.wav` and the sample rate returned by `librosa.load`, with every other parameter left at its 0.11.0 default.
- Temporal mean/std means reduction along the time (frame) axis of each feature matrix.
- Cosine similarity between two vectors `a` and `b` is `dot(a, b) / (norm(a) * norm(b))`.
- The predicted label is the centroid key whose cosine similarity is the largest. Ties must be broken by Python's natural string ordering of the tied keys (i.e., the lexicographically smallest key wins on a tie).
- The verifier will recompute the fingerprint from the same input file using the documented order and call `numpy.allclose(rtol=1e-4, atol=1e-4)` element-wise.

## Acceptance Criteria
- Project path: /workspace
- Ensure the fingerprint pipeline is executed and the output artifact exists.
- Output file: `/workspace/features.json`
- The output file must be a JSON object with exactly these top-level keys:
  - `vector`: list of 72 finite floats.
  - `feature_order`: list of strings equal to `["mfcc_mean", "mfcc_std", "chroma_mean", "chroma_std", "centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std", "rolloff_mean", "rolloff_std", "zcr_mean", "zcr_std", "contrast_mean", "contrast_std"]`.
  - `similarities`: JSON object with exactly the 3 keys `rock`, `classical`, `jazz`; every value is a float in the closed range `[-1.0, 1.0]`.
  - `predicted_label`: string equal to the centroid key with the maximum similarity (ties broken by the lexicographically smallest key).
- The `vector` values must match a reference computation against `/workspace/input.wav` using `numpy.allclose(rtol=1e-4, atol=1e-4)` element-wise.

