Music synchronization requires aligning two different performances of the same piece using robust, pitch-invariant audio features.

You need to implement a script `align_audio.py` that loads two distinct audio segments (`ref.wav` and `comp.wav`), computes Chroma Energy Normalized Statistics (CENS) for both sequences, and utilizes Dynamic Time Warping (DTW) to calculate the optimal warping path between the two performances in an offline analysis pipeline. Output the total accumulated alignment cost value to a text file named `alignment_cost.txt`.

**Constraints:**
- Use `librosa.feature.chroma_cens` to generate the features and `librosa.sequence.dtw` to compute the alignment.
- Positional arguments are restricted; pass all configuration parameters (like `metric`, `subseq`, and `backtrack`) as strictly keyword-only arguments.