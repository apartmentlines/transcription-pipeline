* Look at https://huggingface.co/superwhisper/s1-mini for cleaning up transcripts

## Qwen shared engine follow-up

* Tune recording limits 2/4/8 and chunk batches 8/16/24/32 at the current
  45-second chunk default. Change one setting at a time, then test promising
  combinations. Compare quality, retries/failures, throughput, per-recording
  latency, and peak VRAM/RAM.
* Repeat benchmarks through production images on the same 100 recordings,
  with exclusive GPU access and matching timing boundaries for WhisperX.
  Separate startup, ASR/alignment/output, and shutdown, and repeat runs to
  establish variability. Save results under `benchmarks/`.
* Measure startup with empty and populated compilation caches. Determine how
  RunPod can persist caches across pods and GPU types; retain compilation
  fallback for incompatible hardware/configurations. Validate other deployment
  GPUs/MIG slices and CUDA 13 driver compatibility.

## Testing framework

### If sticking with mono audio

* Implement SRT block timestamp editing in website review
* Review first 14 transcriptions, fix timestamps as needed
* Finish remaining transcriptions reviews
* Port unlabeled and labeled reference SRTs to srt-labeler, re-run labeling accuracy tests

### If switching to stereo audio

* Figure out best native file format, and how best to prep stereo files for mono processing
