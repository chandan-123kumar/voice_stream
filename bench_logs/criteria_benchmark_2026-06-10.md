# Benchmark vs. assignment criteria — fresh run 2026-06-10

Hardware: RTX 5090 (32 GB, sm_120), bf16, single process per test, GPU otherwise idle.
All numbers below were re-measured today (not copied from earlier logs), except the
stock-`qwen_tts` baseline which is from `bench_logs/bench_vs.log` (same machine, same day).

## Scorecard

| Criterion | Target | Measured | Verdict |
|---|---|---|---|
| Talker decode rate | — (blog: ~1000 tok/s) | **1020 tok/s** (0.981 ms/step) | ✅ |
| TTFC | < 60 ms ref / < 90 ms deliverable | **89 ms** (Pipecat service) | ✅ deliverable, ❌ 60 ms ref — see below |
| RTF | < 0.15 ref / < 0.3 deliverable | **0.163 – 0.166** | ✅ deliverable, ~9% above 0.15 ref — see below |
| Streaming (not buffered) | required | ✅ verified at 2 layers (below) | ✅ |
| Audio quality | no glitches/drops | peak 0.79, 0 clipped samples, longest zero-run 4.9 ms | ✅ |
| E2E latency (speech-end → first reply audio) | report | **2.45 / 2.64 / 3.00 s** (min/median/max, 3 turns) | reported; ~95% is hosted OpenAI STT/LLM + turn-stop |

## 1. Raw megakernel decode (`tests/bench.py`)

```
talker megakernel: 500 steps in 490.4 ms -> 0.981 ms/step = 1020 tok/s
```

Per-component cost inside one 12 Hz frame (83.3 ms of audio):

| Component | ms |
|---|---|
| talker megakernel step | 0.785 |
| codec head logits + sample | 0.044 |
| code predictor (CUDA graph) | 8.934 |
| codec decode, 25 frames (amortized ~1.0/frame) | 25.831 |

The megakernel is **8%** of frame cost; the 5-layer code predictor
(15 sequential codebooks/frame) is **~80%** — it is the bottleneck, not the kernel.

## 2. Engine streaming TTFC / RTF (`tests/bench.py`, default `first_chunk_frames=4`)

```
run 0: TTFC 105.0 ms | audio 11.44s | wall 1.89s | RTF 0.166
run 1: TTFC 100.0 ms | audio 11.36s | wall 1.88s | RTF 0.165
run 2: TTFC  99.3 ms | audio 10.40s | wall 1.70s | RTF 0.163
```

## 3. Pipecat TTS service (`tests/test_pipecat_tts.py`, service default `first_chunk_frames=2`)

```
OK: 28 streamed chunks, first frame after 89 ms, 25.0s audio @ 24000 Hz
```

This is the TTFC that matters for the voice agent — the service emits its first
PCM frame into the pipeline **89 ms** after `TTSSpeakFrame`. `first_chunk_frames=1`
gets 72–75 ms (see `knowledge/ttfb-optimization.md`).

## 4. Voice agent end-to-end (`tests/bench_e2e.py` variant, question wav from `synthesize.py`)

Question: "What is the capital of France?" (2.32 s), played as paced 32 ms packets.

```
turn 0: speech-end -> first reply audio = 2638 ms
turn 1: speech-end -> first reply audio = 2448 ms
turn 2: speech-end -> first reply audio = 3003 ms
```

Per-service TTFB (Pipecat instrumentation, same turns):

| Stage | Range |
|---|---|
| OpenAI realtime STT final | 0.84 – 1.01 s |
| OpenAI LLM first token (`gpt-4o-mini`) | 0.70 – 1.31 s |
| **Megakernel TTS first audio (local)** | **0.109 – 0.137 s** |

Plus ~1 s turn-stop decision (Silero VAD + Smart Turn v3), partially overlapped.
Local TTS is ~5% of perceived latency; the rest is hosted-API round-trips.

## 5. Streaming verification (no buffer-then-send)

Transport-level pacing from the same 3 turns — buffered audio would arrive in one burst
(span ≈ 0); instead the arrival span tracks audio duration:

```
turn 0: 138 frames over  5.44s for  5.52s of audio (max inter-frame gap 75 ms)
turn 1: 580 frames over 23.12s for 23.20s of audio (max inter-frame gap 74 ms)
turn 2:  82 frames over  3.20s for  3.28s of audio (max inter-frame gap 77 ms)
```

Service-level: 28 incremental `TTSAudioRawFrame` chunks per utterance (section 3).

## 6. Stock baseline (context, from `bench_vs.log` same day/machine)

stock `qwen_tts` RTF **1.048** (slower than real time, no streaming, first audio = full
wall 6.3–7.8 s) vs megakernel RTF **0.187** → 5.6× faster, first audio 90 ms.

## Why the strictest reference targets are missed (honest accounting)

- **TTFC 89 ms vs 60 ms ref:** the 12 Hz codec decoder needs a minimum context of
  codec frames before it can emit audio. First chunk = prefill (1 HF forward) +
  2 frames × ~9.8 ms decode + first codec decode + service hop. `first_chunk_frames=1`
  reaches 72–75 ms; going below ~60 ms needs a faster code predictor, not a faster talker.
- **RTF 0.163 vs 0.15 ref:** per-frame floor is ~9.8 ms compute + ~1.0 ms amortized
  codec ≈ 10.8 ms per 83.3 ms of audio → RTF ≈ 0.13 theoretical; measured 0.163 adds
  prefill + chunk-boundary codec bursts. The fix with headroom is a second megakernel
  (or fused kernel) for the 5-layer code predictor — projected RTF < 0.05, since the
  talker megakernel itself runs at RTF ≈ 0.01 (0.785 ms / 83.3 ms).

## Notes on this run

- First e2e attempt stalled: the test question wav (cut from a previous TTS output)
  transcribed as "Hello. Um..." and OpenAI realtime STT never committed a transcript.
  Regenerating a clean question via `synthesize.py` fixed it — the agent itself had no
  errors. Recorded here because hosted-STT silence is a real failure mode the agent
  currently has no timeout/retry for.
- The bench client printed `heard: [] / bot said: []` — the server no longer forwards
  transcription/text frames to this client; latency numbers are unaffected (they key on
  audio frames).
