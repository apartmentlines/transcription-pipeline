* Look at https://huggingface.co/superwhisper/s1-mini for cleaning up transcripts

## Qwen shared engine follow-up

* Audio investigation completed for all 100 recordings, with focused review of
  the tiny cues, warning intervals, gaps, and repetition cases. Findings and
  pending human-feedback records are in
  [the saved review](test-data/quality-review/2026-11-18/REVIEW.md).
  Human adjudication remains; model assessments are not reference transcripts.
* Context/realignment experiments completed with production unchanged:
  [results and saved evidence](test-data/quality-experiments/2026-11-18-context-alignment/RESULTS.md).
  Two 100-call runs per setting: 45-second chunks were ~11% faster than 30,
  with mixed recognition changes and no demonstrated net quality loss. Revised
  recommendation adopted: 45 is now the default; minor name
  errors do not justify preferring 30. Overall ASR superiority is unproven.
  60 seconds was slower and
  echoed the ASR prompt. Selective realignment fixed some stretched words
  but introduced collapses and neighbor displacement, even targeting only
  long words. No tested realignment policy is ready to ship.
* Next quality experiments: compare minimal/no ASR context against the current
  prompt on leakage, bilingual, name, and repetition cases, then the corpus.
  Judge timing issues by downstream speaker-label accuracy with full audio plus
  SRT, not subtitle playback. Compare original versus corrected timing with
  identical text/blocks before investing in realignment; mixed-speaker blocks
  matter more when the labeler must preserve blocks and assign one speaker.
  Use the pending feedback records for later human review.
  Do not infer speakers, invent timings, or treat model assessments as truth.
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
* Audit removable vLLM dependencies for an ASR/alignment-only image. Keep both
  bundled checkpoints; no duplicate model weights were found. Validate any
  pruning with offline startup and corpus inference before adopting it.
