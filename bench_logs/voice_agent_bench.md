# Voice agent end-to-end benchmark — 2026-06-10

Setup: RTX 5090 (vast.ai), pipeline entirely in one process on the GPU box
(`voice_agent.py`), benchmark client (`tests/bench_e2e.py`) on the same box —
network latency to OpenAI is real, client↔server latency is ~0.

Method: the client connects to the agent's WebSocket, plays a synthesized
spoken question ("What is the capital of France?", 2.4 s) paced in 32 ms
packets like a live mic, and measures **speech-end → first reply audio frame**
— the silence a human would hear. Metrics per service come from Pipecat's
TTFB instrumentation in the server log.

## Headline numbers

| Configuration | speech-end → first reply audio |
|---|---|
| segmented STT (`gpt-4o-mini-transcribe`) | 3.7 – 4.0 s |
| **realtime STT (`gpt-realtime-whisper`), best run** | **2.0 – 2.6 s** |
| realtime STT, worst observed run | 3.8 – 5.3 s |

The spread between best and worst realtime-STT runs is OpenAI-side variance,
not local: in the worst run one LLM time-to-first-token spiked to 2.28 s
while local TTS TTFB stayed at 93–130 ms in every single turn.

## Where the time goes (per turn, realtime STT)

| Stage | Typical | Notes |
|---|---|---|
| Turn-stop decision (Silero VAD + Smart Turn v3) | ~1 s | local, tunable |
| STT final transcript after commit | 0.87 – 1.23 s | OpenAI realtime API |
| LLM time-to-first-token (`gpt-4o-mini`) | 0.37 – 2.28 s | OpenAI, high variance |
| **TTS time-to-first-audio (megakernel, local)** | **0.09 – 0.18 s** | constant |

The local TTS is ~5% of the response time; the rest is hosted-API and
turn-taking cost. (Stages overlap partially, so they do not sum exactly to
the headline number.)

## Streaming verification (no buffering)

Reply audio is paced frame-by-frame by the output transport; if it were
buffered-then-sent all frames would arrive in one burst:

```
turn 0: 116 frames over 4.56s for 4.64s of audio (max inter-frame gap 79 ms)
turn 1:  90 frames over 3.52s for 3.60s of audio (max inter-frame gap 75 ms)
turn 2:  78 frames over 3.04s for 3.12s of audio (max inter-frame gap 76 ms)
```

## In-process TTS service TTFB (Pipecat-measured)

| `first_chunk_frames` | TTFB |
|---|---|
| 4 (engine default) | ~102 ms |
| **2 (service default)** | **82–87 ms** |
| 1 | 72–75 ms |

Required two fixes documented in `knowledge/ttfb-optimization.md`: a
persistent warmed generation thread (a fresh thread per utterance costs
~35 ms of per-thread CUDA/cuBLAS init) and warming the codec decode with the
production chunk shape.

## Sample artifact

`assets/sample_agent_turn.wav` is a real bot turn recorded from a live
conversation; `assets/sample_agent_turn.json` is its saved metrics:
response 1465 ms = STT 1005 + LLM 762 + TTS 130 (+ turn-stop, overlapped).

## Reproduce

```bash
python3 voice_agent.py                 # GPU box, needs OPENAI_API_KEY in .env
python3 tests/bench_e2e.py 3           # measures 3 turns, prints profile
grep TTFB <server log>                 # per-service breakdown
```
