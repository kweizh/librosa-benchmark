# Laplacian Structural Music Segmentation

## Background
Use `librosa` 0.11.0 to build a structural music segmentation pipeline that recovers song-section boundaries and re-uses cluster identities (A/B/A...) across repeating sections. The pipeline must beat-track the input, extract a beat-synchronous tonal feature, build a weighted recurrence graph that mixes path-enhanced repetition with sequential affinity, embed frames via the symmetric normalized graph Laplacian, and assign one section label per frame using temporally-constrained agglomerative clustering before mapping the labels back to absolute time intervals.

## Requirements
- Read the input WAV file from `/workspace/input.wav`.
- Detect beats, then aggregate a tonal feature (CQT magnitude or `chroma_cqt`) into beat-synchronous frames using `librosa.util.sync`.
- Construct a recurrence-based affinity matrix and enhance its diagonal paths with `librosa.segment.path_enhance`.
- Combine the path-enhanced repetition graph with a sequential (local) affinity term into a single symmetric weighted adjacency.
- Compute the symmetric normalized graph Laplacian, take its bottom eigenvectors, normalize the chosen eigenvector embedding, and cluster the per-beat embeddings into a small number of section types.
- Map cluster ids back to contiguous time intervals and write them to `/workspace/segments.json`.

## Implementation Hints
- All `librosa` 0.11.0 feature/segment APIs are keyword-only except for primary data arrays. Verify signatures before calling.
- You may use any of: `librosa.load`, `librosa.beat.beat_track`, `librosa.cqt`, `librosa.feature.chroma_cqt`, `librosa.util.sync`, `librosa.util.fix_frames`, `librosa.segment.recurrence_matrix`, `librosa.segment.path_enhance`, `librosa.segment.agglomerative`, `librosa.frames_to_time`, plus NumPy / SciPy / scikit-learn helpers for eigendecomposition and clustering.
- Choose the Laplacian normalization, the number of eigenvectors `K`, the number of clusters, and the relative weighting of repetition vs. sequential affinity yourself; defaults that work on the librosa segmentation gallery example are a reasonable starting point.
- The audio at `/workspace/input.wav` is mono at 22050 Hz and has a clear A/B/A-style structure, so the output must demonstrate that at least one section label is re-used (proving repetition recovery, not just temporal partitioning).

## Acceptance Criteria
- Project path: /workspace
- Ensure the segmentation pipeline is executed and the output artifact exists.
- Output file: `/workspace/segments.json`
- The output file must be a JSON array. Each element must be an object with the following schema:

  ```json
  {
    "start": number,
    "end": number,
    "label": string
  }
  ```

  - `start` and `end` are floating-point timestamps in seconds with `start < end`.
  - `label` must be a non-empty string matching the regex `^[A-Z]$` (a single uppercase letter such as `A`, `B`, `C`, ...).
- Segments must be sorted by `start`, cover the audio from near 0s through the audio duration, and have neither significant overlaps nor gaps between adjacent segments.
- The decoded segmentation must contain at least 2 distinct labels, at least 3 segments total, every segment must have duration greater than 0.5 seconds, and at least one label must appear in more than one segment (re-used section identity).

