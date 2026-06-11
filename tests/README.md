# Tests & benchmarks — reviewer guide

Six scripts: three benchmarks that back the numbers in
[`bench_logs/performance.md`](../bench_logs/performance.md), three tests for
correctness and integration. All run from the repo root on the GPU server.
First-ever run downloads the model and JIT-compiles the kernel (a few minutes);
after that each script starts in ~60 s (CUDA graph capture).

```bash
pip install -r requirements.txt
bash scripts/fix_torchaudio_stub.sh   # see torchaudio note in the main README
```

## Benchmarks (the criteria numbers)

### `bench.py` — TTFC, RTF, per-component costs
```bash
python3 tests/bench.py
```
No external services needed. Expect: raw talker decode **~1030 tok/s**; a
per-component table (code predictor ≈ 8.9 ms — the bottleneck at 86% of step
time; megakernel step ≈ 0.77 ms; codec decode ≈ 12.7 ms flat); three streaming
runs ending like `TTFC 47.8 ms | audio 10.40s | wall 1.50s | RTF 0.144`.
Targets: TTFC < 60 ms, RTF < 0.15 (utterances under ~3 s of audio reach
0.150–0.158 — explained in `performance.md`).

### `bench_vs_baseline.py` — megakernel vs stock `qwen_tts`
```bash
python3 tests/bench_vs_baseline.py
```
Same weights, texts, speaker, seed. Expect: stock RTF ~0.74 (no streaming,
first audio after ~4.6–5.0 s) vs megakernel **~0.148 (5.0×), first audio ~47 ms**.
Writes `cmp_stock.wav` / `cmp_megakernel.wav` to the repo root for a
listening comparison — the quality should be indistinguishable.

### `bench_e2e.py` — voice agent latency + streaming proof
```bash
python3 voice_agent.py &      # needs OPENAI_API_KEY in .env; wait ~60 s
python3 synthesize.py "What is the capital of France?" --speaker serena --out /tmp/q.wav
python3 tests/bench_e2e.py 2 --question-wav /tmp/q.wav
```
Connects to the agent's WebSocket like a browser mic, paces the question in
real time, and timestamps every reply frame. Expect speech-end → first reply
audio of ~0.4–1.1 s (the spread is OpenAI LLM first-token variance) and a
per-turn streaming profile like:
```
streaming profile: 190 frames over 7.52s for 7.60s of audio (max inter-frame gap 56 ms)
```
Arrival span ≈ audio duration proves audio streams **frame-by-frame** —
buffered-then-sent would arrive as one burst (span ≈ 0). Add
`--agent-log <file>` for the per-stage breakdown (local Whisper STT ~0.3 s,
LLM 0.4–1.0 s, megakernel TTS ~0.06 s).

## Tests

### `test_parity.py` — megakernel vs HF numerics
Feeds identical inputs to the megakernel and the HF talker (sdpa), compares
post-norm hidden states and codec-head logits per step.
> **Known issue (2026-06-11):** fails 1/24 greedy argmax steps. Verified
> pre-existing (identical failure with all recent changes reverted): one
> step's bf16 noise flips the argmax between two near-equal logits; both
> tokens are inside the actual top-k-50 sampling set. The strict argmax
> assertion is environment-sensitive — it passed on 2026-06-10. Details in
> [`bench_logs/performance_history.md`](../bench_logs/performance_history.md).

### `test_pipecat_tts.py` — Pipecat TTS service integration
Runs `MegakernelTTSService` inside a real Pipecat pipeline (no API key
needed). Expect: PASS with ~20 streamed chunks and **pipeline TTFB 48–50 ms**.

### `test_e2e.py` — full synthesis sanity
Generates a complete utterance through the streaming engine plus an HF
baseline for comparison; writes wavs for listening.
