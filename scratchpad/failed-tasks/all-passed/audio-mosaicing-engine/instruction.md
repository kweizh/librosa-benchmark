# Concatenative Audio Mosaicing

## Background
Audio mosaicing is a concatenative synthesis technique that repaints a target recording using grains borrowed from a separate source recording.

## Inputs
- `/workspace/target.wav` — the audio to repaint (22050 Hz mono).
- `/workspace/source.wav` — the audio whose grains may be borrowed (22050 Hz mono).

## Output
- `/workspace/mosaic.wav` — the mosaiced audio at the target sample rate.

## Requirements
- Split the target into fixed, non-overlapping frames of length 2048 samples (frame_length=2048, hop_length=2048).
- For each target frame, select the most timbrally similar source frame by cosine-similarity nearest neighbor over MFCC vectors computed on the same 2048-sample non-overlapping framing of the source.
- Concatenate the selected source frames back together in target-frame order to produce the mosaic, and write it to `/workspace/mosaic.wav` at the target sample rate.

