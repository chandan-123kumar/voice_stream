# Voice agent latency reduction — 2026-06-10

Goal: cut speech-end → first-reply-audio (measured 2.45–3.00 s in
`criteria_benchmark_2026-06-10.md`). Same hardware (RTX 5090), same method
(`tests/bench_e2e.py`, 2.32 s spoken question, 3 turns per config).

## What the log timeline actually showed

A timestamp reconstruction of a real turn corrected the assumed breakdown — the
turn-stop decision was *not* the ~1 s cost previously reported:

| Stage (turn at 13:23, 2.64 s total) | Measured |
|---|---|
| VAD (stop_secs 0.2) + Smart Turn v3 verdict | ~0.2–0.3 s |
| **wait for OpenAI realtime STT final transcript** (turn-stop 15.215 → LLM start 16.677) | **~1.46 s** |
| LLM first token | 0.81 s |
| TTS sentence aggregation + first audio | 0.30 s |

The aggregator cannot run the LLM until the final transcript arrives, so the
hosted STT round-trip sat directly on the critical path.

## The fix: local Whisper STT on the same GPU

`STT_BACKEND=whisper` (now the default) runs faster-whisper
`large-v3-turbo` (ct2, fp16) in-process. One process-wide model, loaded and
JIT-warmed at server startup (first CUDA inference compiles ctranslate2
kernels, ~10 s once; warmed transcription of the 2.3 s question takes
**0.15–0.18 s** standalone, 0.29–0.32 s in-pipeline including segment
assembly). `STT_BACKEND=openai` keeps the hosted path as a fallback.

ctranslate2 4.8.0 runs fine on sm_120 — no Blackwell issues beyond the
one-time JIT (the fallback ladder to int8/CPU was not needed).

## Results (speech-end → first reply audio, ms)

| Config | min | median | max | STT TTFB | LLM TTFB | TTS TTFB |
|---|---|---|---|---|---|---|
| openai STT (same code, fallback path) | 2181 | **2293** | 2402 | 860–1064 | 499–633 | 103–133 |
| **whisper STT (default)** | 864 | **1117** | 1411 | 286–315 | 580–999 | 95–132 |
| whisper STT, earlier run (lower LLM variance) | 648 | **716** | 997 | 302–314 | 400–635 | 84–137 |

**Median cut from ~2.3–2.6 s to ~0.7–1.1 s (2–3.7×).** The spread between the
two whisper runs is entirely OpenAI LLM first-token variance (0.40–1.00 s),
which is now ~60 % of the remaining latency and the next bottleneck.
Transcripts were verified correct in the agent log on every turn; streaming
stayed real-time paced (max inter-frame gap 74–82 ms).

## Tried and rejected: `first_chunk_frames=1`

TTS TTFB drops to ~75 ms, but the first chunk carries only 83 ms of audio
while the next chunk takes ~133 ms to compute — playout stalls ~70 ms at every
utterance start (observed as 151–155 ms inter-frame gaps vs 74–82 ms at the
default of 2). A 15 ms TTFB win is not worth an audible hiccup; the service
default (2) stays.

## Remaining levers (not done)

- **LLM first token (0.4–1.0 s, hosted)** — now the dominant cost. A/B faster
  hosted models via `OPENAI_LLM_MODEL`, or go local (decided against for now:
  quality tradeoff).
- Turn-stop (~0.2–0.3 s) and whisper (~0.3 s) are small; a streaming local STT
  (transcribe while speaking) could hide the 0.3 s but adds real complexity.

## Reproduce

```bash
python3 voice_agent.py                       # STT_BACKEND=whisper is the default
python3 synthesize.py "What is the capital of France?" --speaker serena --out /tmp/q.wav
python3 tests/bench_e2e.py 3 --question-wav /tmp/q.wav --agent-log <server log>
STT_BACKEND=openai python3 voice_agent.py    # hosted comparison
```
