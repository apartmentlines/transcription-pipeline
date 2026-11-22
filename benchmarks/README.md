# Performance measurements

The [performance summary](../docs/performance.md) presents the final comparison.
These files preserve measured evidence; they are not automated test fixtures.
No inference was rerun when collecting them here on 22 November 2026.

## Saved results

| File | Measurement |
| --- | --- |
| [qwen-context-comparison.json](results/qwen-context-comparison.json) | 18 November: two runs each at 30/45/60-second maximum chunks, using the installed Qwen CLI/shared-engine path, four recordings and 16 active chunks. Includes startup, processing, shutdown, memory, median recording latency, retries and structural audits. |
| [worker-comparison.json](results/worker-comparison.json) | 18 November: Qwen single/shared/multiple-engine configurations and two-worker WhisperX, one run per configuration. This is the source of the WhisperX summary row. Earlier Qwen configurations had incomplete-generation failures; their speed must be read alongside success counts. |
| [qwen-production-30s.json](results/qwen-production-30s.json) | Production-container Qwen baseline at 30-second chunks, including exact transcriber arguments and image ID. All 100 outputs completed; processing took 83.1 seconds and the measured total was 163.3 seconds. Predates the 45-second default. |
| [transformers-baseline.json](results/transformers-baseline.json) | Earlier sequential Transformers Qwen versus WhisperX. Qwen completed 96/100 recordings; WhisperX completed 100/100 but diagnostics export failed. This does not describe the current Qwen engine. |

## Workload and measurement boundaries

The workload is the original 100 operator recordings from 5–6 November 2026,
totaling 11,021.16 seconds. Their filenames are retained in the production
command. Stable audio now lives at
`~/git/apartmentlines/testing-references/audio/operator-recordings/`.
Configurations ran sequentially on one RTX 3090 without competing inference.

Processing time includes transcription, forced alignment and output writing.
Throughput is audio duration divided by processing wall time; real-time factor
is its reciprocal. For runs with failures, attempted-audio throughput does not
mean all that audio was successfully transcribed. Summed concurrent stage or
recording times are not wall time.

The worker experiment starts its timer before sequential worker initialization;
processing starts after workers are ready. Its total includes worker shutdown,
but excludes Docker launch and imports before the harness timer. Compilation
artifacts were cached. WhisperX's stage totals of zero mean those stages were
not separately measured, not that ASR/alignment took no time. The generic
`kv_cache_mib` setting in its configuration is unused by WhisperX.

The context experiment also excludes Docker launch/pulls and Python imports
before its timer. It uses a populated compilation cache. Startup includes
subsequent imports, model loading and engine initialization. Result/timestamp
capture adds diagnostic I/O (about 0.75 seconds of native-result writes per
run). The earlier production 30-second baseline used a cold compilation cache;
its total must not be treated as a matched comparison with these warm runs.

Context-run memory was sampled every 0.5 seconds. GPU memory is device-wide;
host memory is summed process RSS and can count shared pages more than once.
Divide MiB by 1,024 for GiB. Two context repetitions provide an observed range,
not a confidence interval. The records do not provide a consistent p95 latency
measurement across backends or an accuracy ranking.

## Build identity and provenance

The production baseline and context experiment used bundled-model image
`sha256:857441d583ace9cb423eba77648d37b0ce0c369b1a3697dce751b746f4791994`
(`transcription-pipeline-qwen3:shared-engine`). The worker logs identify Qwen ASR
as `Qwen/Qwen3-ASR-1.7B-hf` revision
`bcd2b5b7f32b480ab5790554cfa8347f246a14f3`, and WhisperX ASR as
`Systran/faster-whisper-large-v2` revision
`f0fe81560cb8b68660e564f55dd99207059c092e`.
Exact Git commits, the WhisperX image digest, complete runtime versions and an
audio-content manifest were not captured in these preserved summaries; they
must not be inferred from the current checkout or installed Docker tags.

The baseline, worker and production JSON files are unchanged copies of
`/tmp/call-output-audit/benchmark-summary.json`,
`/tmp/call-output-audit/worker-experiment/summary.json`, and
`/tmp/call-output-audit/shared-engine-production/benchmark.json`, respectively.
The context JSON preserves the complete `context_runs` array from
`test-data/quality-experiments/2026-11-18-context-alignment/experiment-summary.json`;
unrelated realignment and audio-assessment sections are excluded. Its method
description comes from that experiment's `RESULTS.md`.
Paths inside saved JSON describe the original runs and are not portable input
locations. The copied measurements can be read without those working files.

## Recording future runs

Add compact results under `results/`, recording the date, Git commit, image
digest, model/runtime versions, GPU/driver, audio manifest, exact command,
worker/recording limit, batch and chunk sizes, and compilation-cache state.
Separate startup, processing and shutdown; include throughput, median/p95
recording latency, sampled peak VRAM/RAM, successes, failures and retries.
Record sampling/timing boundaries and repeat runs to show variability. Keep
large logs and generated transcripts in ignored `test-data/`. Outstanding
performance experiments remain in [TODO.md](../TODO.md).
