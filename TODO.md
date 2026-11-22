* Look at https://huggingface.co/superwhisper/s1-mini for cleaning up transcripts

## Qwen shared engine follow-up

* Tune concurrency and chunking from the current defaults: four recordings,
  16 active chunks, 45-second maximum chunks. Try recording limits 2/4/8,
  chunk batches 8/16/24/32. Chunk durations 30/45/60 have been compared above;
  rebuild deployment images to pick up the new default. Change one
  setting at a time, then test promising combinations. Compare text, timing,
  retries/failures, throughput, per-recording latency, and peak VRAM/RAM.
  Larger batches improve scheduling; longer chunks provide more audio context.
* Repeat benchmarks through production images on the same 100 recordings,
  with exclusive GPU access and matching timing boundaries for WhisperX.
  Separate startup, ASR/alignment/output, and shutdown. Current Qwen baseline:
  100/100 successful, 83.1 s processing, 163.3 s total, 16.25 GiB peak VRAM
  on the RTX 3090. Repeat runs to establish variability.
* Measure startup with empty and populated compilation caches. Determine how
  RunPod can persist caches across pods and GPU types; retain compilation
  fallback for incompatible hardware/configurations. Validate other deployment
  GPUs/MIG slices and CUDA 13 driver compatibility; only the 3090 is measured.

## Testing framework

### If sticking with mono audio

* Implement SRT block timestamp editing in website review
* Review first 14 transcriptions, fix timestamps as needed
* Finish remaining transcriptions reviews
* Port unlabeled and labeled reference SRTs to srt-labeler, re-run labeling accuracy tests

### If switching to stereo audio

* Figure out best native file format, and how best to prep stereo files for mono processing
