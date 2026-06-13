# Music Similarity Search Engine

## Background
Build a music similarity search engine using librosa 0.11.0.

## Inputs
- `/workspace/library/`: a directory of N library `.wav` tracks.
- `/workspace/query.wav`: a single query clip.

## Requirements
- For the query and for every library track, derive a single fixed-dimensional, track-level embedding vector.
- The embedding MUST be L2-normalized (unit Euclidean norm).
- Rank EVERY library track in `/workspace/library/` by cosine similarity between its embedding and the query embedding.
- Emit `/workspace/results.json` with the following exact schema:

  ```json
  {
    "results": [
      { "filename": "<basename>.wav", "similarity": <float in [-1.0, 1.0]> }
    ]
  }
  ```

## Acceptance Criteria
- Project path: /workspace
- `/workspace/results.json` exists and is valid JSON.
- The `results` list contains exactly one entry for each `*.wav` file in `/workspace/library/` (no duplicates, no extras).
- `filename` is the basename of the library file (no directory components).
- `similarity` is a finite float in `[-1.0, 1.0]`.
- The list is sorted strictly in descending order of `similarity`.
- The track from which the query was derived must rank near the top with a clearly high similarity.

