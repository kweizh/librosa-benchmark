# Laplacian Structural Segmentation

## Background
Given a music track, you must identify its structural sections (intro, verse, chorus, bridge, etc.) using a Laplacian-based spectral clustering pipeline driven by `librosa`.

## Task
1. Read the input audio file from `/workspace/input.wav`.
2. Compute a structural segmentation in which beat-synchronous spectral features are converted into a recurrence-based graph whose normalized Laplacian is spectrally decomposed and then clustered to produce contiguous time intervals (“segments”).
3. Write the resulting segment boundaries to `/workspace/segments.json` as a JSON array, sorted by start time, where each element has the shape:
   ```json
   {"start": <float seconds>, "end": <float seconds>}
   ```

## Acceptance Criteria
- Project path: `/workspace`
- The pipeline is executed end-to-end and the artifact `/workspace/segments.json` exists.
- `/workspace/segments.json` is valid JSON: a list of objects, each with float keys `start` and `end` in seconds.
- Segments are sorted by `start`, contain no negative durations, and every duration is greater than 1.0 second.
- Segments cover the full audio: the first segment starts at (approximately) 0 seconds and the last segment ends at (approximately) the audio duration.
- Adjacent segments are contiguous (no gap larger than 0.5 s and no overlap larger than 0.1 s).
- The total number of segments is between 3 and 15 inclusive.

## Hints
- The agent is free to choose the implementation details (feature type, hop length, neighborhood width, number of eigenvectors, clustering linkage, etc.) but the approach must combine: beat tracking, beat-synchronous spectral features, a recurrence/affinity matrix, a normalized graph Laplacian eigendecomposition, and agglomerative clustering on the leading eigenvectors.
- The official `librosa` Laplacian Segmentation gallery example is a useful reference.

