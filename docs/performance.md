# Backend performance

The shared Qwen vLLM engine processed the reference workload substantially
faster than two-worker WhisperX in the saved RTX 3090 measurements. All runs
below completed the same 100 recordings, totaling 3.061 hours of audio.

| Configuration | Processing | Startup/shutdown overhead | Total | Peak GPU memory | Completed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen shared engine: 4 recordings, 16 active chunks, 45-second chunks; two warm-cache runs | 74.4–75.1 s | 47.6–48.0 s | 122.4–122.7 s | 12.21–14.97 GiB | 100/100 each |
| WhisperX: 2 workers, batch size 16; one run | 246.8 s | 10.0 s | 256.8 s | 17.61 GiB | 100/100 |

Overhead is each run's measured total minus its processing time.

Qwen's measured processing throughput was approximately 147–148 times real
time, versus 44.7 for WhisperX: about 3.3 times faster. Qwen has substantially
more initialization overhead, so the advantage is smaller for short jobs.

These are separate experiments with different measurement harnesses and cache
conditions, not a matched end-to-end deployment benchmark. Peak memory is
sampled usage, not a guaranteed capacity requirement; completion does not
establish transcription accuracy. Results apply to this corpus and GPU.

See [benchmarks](../benchmarks/README.md) for saved measurements, configurations,
timing definitions, provenance, and earlier experiments.
