# Performance Report — TTFC & RTF (2026-06-11)

Criteria benchmark of the megakernel TTS engine against the two latency targets.
Raw log: [`performance_run_2026-06-11.log`](performance_run_2026-06-11.log) ·
sample output: [`performance_sample_2026-06-11.wav`](performance_sample_2026-06-11.wav)

## Headline results

| Metric | Target | Measured (20 runs, 4 text lengths) | Verdict |
|---|---|---|---|
| **TTFC** (time to first audio chunk) | < 60 ms | **47.0 ms median** (46.3 – 48.3) | ✅ pass, 22% margin |
| **RTF** (wall time / audio duration) | < 0.15 | **0.145 median** (0.142 – 0.158) | ✅ for utterances ≥ ~3 s; ⚠️ very short utterances (~2.5 s) reach 0.150 – 0.158 |

RTF 0.145 = generating 1 second of audio takes **145 ms**. Under the alternative
reading of the target ("1 s of audio in < 300 ms", i.e. RTF < 0.3) every run passes
with ≥ 1.9× headroom. Under the strict < 0.15 reading, typical utterances pass and
the very shortest ones marginally miss — the breakdown below shows exactly why,
and what would fix it.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 595.58.03, 500 W limit |
| Stack | torch 2.10.0+cu128, CUDA 12.8, bf16 weights |
| Model | Qwen3-TTS-12Hz-0.6B-CustomVoice, speaker `ryan`, English |
| Config | engine defaults: `first_chunk_frames=2`, `chunk_frames=12`, `context_frames=25`, top-k 50, temp 0.9, seed 0 |

## Methodology

- **TTFC** = `time.perf_counter()` from the `stream()` call to the first audio chunk
  yielded as float32 PCM. It **includes** prompt building, the HF prefill forward,
  2 decode frames, and the codec decode of the first chunk. It **excludes** network
  transport (this is the engine metric; through a full Pipecat pipeline the service
  reports TTFB 48–50 ms, see `tests/test_pipecat_tts.py`).
- **RTF** = total wall clock of the full `stream()` consumption ÷ duration of audio
  actually emitted (samples ÷ 24 000). Includes prefill, every codec decode, and
  GPU→CPU copies of every chunk. Audio duration is summed from emitted chunks, not
  derived from frame counts.
- **Protocol**: warm engine (one warm-up utterance after startup), then 5 runs each
  of 4 texts (3 / 7 / 22 / 48 words). Sampling is stochastic (temp 0.9), so audio
  length varies between runs; each run's RTF uses its own audio duration.
- Max inter-chunk gap was tracked in every run: ≤ 150 ms against 1.0 s of audio per
  steady-state chunk — streaming output is comfortably real-time-paced throughout.

## Per-text results (5 runs each)

| Text | TTFC median | RTF median | RTF min–max |
|---|---|---|---|
| very short (3 words, ~2.6 s audio) | 46.6 ms | 0.156 | 0.150 – 0.158 |
| short (7 words, 3–73 s audio¹) | 46.3 ms | 0.148 | 0.144 – 0.150 |
| medium (22 words, ~10 s audio) | 47.0 ms | 0.143 | 0.142 – 0.145 |
| long (48 words, ~21 s audio) | 47.1 ms | 0.143 | 0.143 – 0.143 |

¹ one run sampled a 72.6 s ramble (temp 0.9, no length cap hit); its RTF was 0.148 —
included, not discarded.

TTFC is flat across lengths (prefill grows negligibly at these prompt sizes). RTF
degrades on very short utterances because two **fixed** costs amortize over fewer
frames: the prefill (~15 ms) and the codec decode, which costs a flat ~12.7 ms per
chunk whether it decodes 2 frames or 37 (it is kernel-launch-bound, not
compute-bound). A 2.6 s utterance pays 4 codec decodes over ~31 frames; a 21 s
utterance pays one per 12 frames at steady state.

## Where the time goes (per 83.3 ms codec frame)

| Component | Cost | Share of decode step |
|---|---|---|
| Talker megakernel step (28 layers) | 0.77 ms | 7% |
| Codec head logits + top-k sample | 0.02 ms | <1% |
| **Code predictor (5 layers × 15 codebooks, CUDA graph)** | **8.91 ms** | **86%** |
| Python glue, sampling, embed sum | ~0.7 ms | 7% |
| Codec decode, amortized (12.7 ms per 12-frame chunk) | ~1.1 ms/frame | — |

The megakernel itself is nowhere near the bottleneck. **The code predictor is**: its 15
sequential codebook generations cost 11× the 28-layer talker step.

TTFC breakdown (47 ms): prefill ~15 ms + 2 decode frames ~21 ms + codec decode
~12.7 ms (flat) — wait-free beyond that; the first chunk carries 167 ms of audio,
more than the 125 ms needed to produce the next chunk, so playback never starves.

## What changed to improve TTFC (measured same GPU, same day)

| Config | TTFC | RTF (median) |
|---|---|---|
| `first_chunk_frames=4` (old default) | 66 – 69 ms ❌ | 0.144 (unchanged by this knob) |
| `first_chunk_frames=2` (new default) | **46 – 48 ms** ✅ | 0.145 (unchanged) |

**Why it's safe:** the codec decoder costs the same flat ~12 ms for any window size,
so accumulating 4 frames before the first decode bought nothing — each extra frame
was ~10.4 ms of pure waiting. The voice agent had already been shipping
`first_chunk_frames=2` through `pipecat_tts.py`; this makes the engine default match
what production uses. Same-seed output vs the old default: identical length,
waveform correlation 0.9993.

## Paths to RTF < 0.15 on very short utterances (explored, not merged)

The flat 12.7 ms codec decode was prototyped behind per-window-size CUDA graphs
(capturing the eager kernels, bit-identical output): it dropped to 2.4–4.8 ms and
brought RTF to 0.133–0.139 across **all** text lengths, TTFC to ~37 ms. The change
was reverted to keep the engine simple; the prototype demonstrates the headroom is
real if very-short-utterance RTF becomes a hard requirement.
(`torch.compile(mode="reduce-overhead")` achieves similar speed but was rejected
outright: it changes bf16 numerics, max waveform diff 0.2–0.5.)

The bigger lever is the **code predictor** (86% of step time): a second megakernel
for its 5-layer × 15-codebook loop could plausibly take RTF from 0.145 to < 0.05.

## vs. stock `qwen_tts` (same weights, texts, speaker, seed — re-run same day, 3 texts)

| | Stock | Megakernel |
|---|---|---|
| RTF | 0.737 – 0.739 | **0.148** (5.0× faster) |
| Decode throughput | 16.3 frames/s | 79 – 83 frames/s |
| Time to first audio | 4.6 – 5.0 s (no streaming) | **47 – 48 ms** |

Stock measures faster today (0.74) than in the 2026-06-10 report (1.05) — environment
drift, which is why both columns were re-measured together rather than reusing the old
stock numbers. Reproduce: `python3 tests/bench_vs_baseline.py` (writes
`cmp_stock.wav` / `cmp_megakernel.wav` for a listening comparison).

## Quality & regression checks

- Same-seed waveform, `first_chunk_frames` 2 vs 4: identical length, correlation
  0.9993 — the smaller first chunk does not change what is generated, only when the
  first decode happens. Listenable sample: `performance_sample_2026-06-11.wav`.
- `tests/test_pipecat_tts.py`: passes — 17 streamed chunks, pipeline TTFB 48–50 ms.
- `tests/test_parity.py`: **fails 1/24 greedy argmax steps — pre-existing, verified
  unrelated to today's changes** (byte-identical failure with the changes stashed).
  One step's bf16 noise flips the argmax between two near-equal logits; under the
  actual top-k-50 sampling both tokens are in the candidate set. The strict argmax
  assertion is environment-sensitive (it passed on 2026-06-10).

## Honest caveats

- All numbers are **warm-engine**: model loaded, kernels JIT-compiled, CUDA graph
  captured. Cold start is ~60 s (first ever run: a few minutes for downloads + JIT).
- The strict RTF < 0.15 target is **not met for utterances under ~3 s of audio**
  (worst observed 0.158); it is met for everything longer, and the < 0.3 reading is
  met everywhere with ≥ 1.9× headroom.
- Batch size 1, single utterance at a time, single GPU — by design of the megakernel.
- Timing is host-side `perf_counter`; chunk timestamps are when Python receives the
  numpy audio, which is the fair point for a streaming consumer.
