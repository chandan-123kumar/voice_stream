# Performance history — how the numbers evolved

Companion to [`performance.md`](performance.md) (current numbers + how to verify them).
All measurements on the same RTX 5090 (sm_120), bf16.

## 2026-06-09 — first working megakernel TTS

Qwen3-TTS talker decode running on the repurposed qwen_megakernel: **1018 tok/s**
(0.98 ms/step), TTFC ~89 ms, RTF ~0.18. Found and fixed the entry-barrier race in
`ldg_decode_kernel_direct` (deadlocked under CUDA-graph-interleaved launches).
Identified the code predictor — not the megakernel — as the bottleneck (~81% of step).

## 2026-06-10 — criteria benchmark + voice-agent latency reduction

- Criteria run: TTFC **89 ms**, RTF **0.163–0.166**, streaming verified frame-by-frame
  (max inter-frame gap 77 ms), audio quality clean (0 clipped samples, longest
  zero-run 4.9 ms). vs stock that day: RTF 1.048 vs 0.187 (5.6×).
- Voice agent: log-timeline analysis showed the dominant E2E cost was waiting ~1.4 s
  for hosted STT finals, not turn-stop. Moved STT to local faster-whisper on the same
  GPU: speech-end → first reply audio went **~3.9 s (segmented STT) → ~2.2 s (realtime
  STT) → 0.7–1.1 s (local Whisper)**. Details: [`latency_reduction_2026-06-10.md`](latency_reduction_2026-06-10.md),
  [`voice_agent_bench.md`](voice_agent_bench.md).

## 2026-06-11 — TTFC under the 60 ms target

- Profiling insight: the 12 Hz codec decoder costs a **flat ~12 ms for any window size
  (1–37 frames)** — it is kernel-launch-bound, not compute-bound. So accumulating 4
  frames before the first codec decode bought nothing; each extra frame was ~10.4 ms
  of pure waiting before first audio.
- Shipped: engine default `first_chunk_frames` 4 → 2 (the voice agent's Pipecat service
  had already been running 2). **TTFC 66–69 ms → 47.0 ms median.** Same-seed output
  identical in length, waveform correlation 0.9993 vs the old default. RTF unchanged
  (0.145 median); stock re-measured same day at 0.738 (5.0×). Pipecat pipeline TTFB
  82–87 ms → 48–50 ms; live-agent streaming re-verified (190 frames over 7.52 s for
  7.60 s of audio, barge-in halts the stream within ~0.9 s of speech onset).

### Explored, not merged: CUDA-graphing the codec decoder

Per-window-size CUDA graphs (capturing the exact eager kernels) cut the codec decode
from 12.9 ms to 2.4–4.8 ms with **bit-identical** output: TTFC 37.5 ms, RTF 0.133–0.139
across all text lengths — including the very short utterances that currently reach
0.150–0.158. Reverted to keep the engine simple; re-apply if very-short-utterance RTF
becomes a hard requirement (prewarm window sizes 2, 14, 26–37; capture lazily otherwise).
`torch.compile(mode="reduce-overhead")` was rejected outright: similar speed but it
changes bf16 numerics (max waveform diff 0.2–0.5).

### Known issue

`tests/test_parity.py` fails 1/24 greedy argmax steps as of 2026-06-11 — verified
pre-existing (byte-identical failure with all of today's changes stashed). One step's
bf16 noise flips the argmax between two near-equal logits; both tokens are inside the
actual top-k-50 sampling set. The strict argmax assertion is environment-sensitive
(it passed on 2026-06-10).
