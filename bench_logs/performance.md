# Performance — TTFC & RTF

Measured 2026-06-11 on an RTX 5090 (sm_120, driver 595.58.03), torch 2.10.0+cu128, bf16,
warm engine, engine defaults (`first_chunk_frames=2`, `chunk_frames=12`, top-k 50, temp 0.9).
Raw log: [`performance_run_2026-06-11.log`](performance_run_2026-06-11.log) ·
sample: [`performance_sample_2026-06-11.wav`](performance_sample_2026-06-11.wav) ·
how we got here: [`performance_history.md`](performance_history.md)

| Metric | Target | Measured (20 runs, 4 text lengths) | Verdict |
|---|---|---|---|
| **TTFC** — `stream()` call → first audio chunk (incl. prompt build, prefill, 2 decode frames, codec decode) | < 60 ms | **47.0 ms median** (46.3 – 48.3) | ✅ |
| **RTF** — total wall ÷ emitted audio duration (incl. prefill and every codec decode) | < 0.15 | **0.145 median**; 0.142 – 0.148 for utterances ≥ 3 s; up to **0.158** on ~2.5 s ones | ✅ for ≥ 3 s; ⚠️ very short utterances miss — see below |
| vs stock `qwen_tts` (same weights/texts/seed, same day) | — | RTF 0.738 → **0.148**, time-to-first-audio 4.6–5.0 s → **47 ms** | 5.0× |

**Where the time goes** (per 83.3 ms codec frame): code predictor 8.9 ms (**86%** — the
bottleneck; a second megakernel for it is the next lever) · talker megakernel step 0.77 ms
(7%) · codec decode ~12.7 ms flat per 12-frame chunk (~1.1 ms/frame amortized) · sampling
\+ glue ~0.7 ms. The flat codec-decode and prefill (~15 ms) costs are why utterances under
~3 s of audio reach RTF 0.150–0.158; a validated fix exists but is not merged
(see [history](performance_history.md)).

## Verify the numbers yourself

```bash
# one-time setup (RTX 5090 / sm_120, CUDA 12.8+)
pip install -r requirements.txt
bash scripts/fix_torchaudio_stub.sh
```

**1. TTFC + RTF + per-component costs** (~3 min after first-run model download/JIT):

```bash
python3 tests/bench.py
```

Expect: `~1030 tok/s` raw decode; per-component table (code predictor ≈ 8.9 ms,
megakernel step ≈ 0.77 ms, codec decode ≈ 12.7 ms); three streaming runs ending in
lines like `TTFC 47.8 ms | audio 10.40s | wall 1.50s | RTF 0.144`.

**2. vs stock baseline** (same seed, writes wavs for a listening comparison):

```bash
python3 tests/bench_vs_baseline.py
```

Expect: `stock qwen_tts : RTF ~0.74`, `megakernel : RTF ~0.148 -> 5.0x faster`,
megakernel TTFC ~47 ms vs stock first-audio ~4.6–5.0 s.

**3. Streaming (frame-by-frame, not buffered) through the full voice agent** —
needs `OPENAI_API_KEY` in `.env`:

```bash
python3 voice_agent.py &                       # wait ~60 s for startup
python3 synthesize.py "What is the capital of France?" --speaker serena --out /tmp/q.wav
python3 tests/bench_e2e.py 2 --question-wav /tmp/q.wav
```

Expect a streaming profile per turn like `190 frames over 7.52s for 7.60s of audio
(max inter-frame gap 56 ms)` — arrival span ≈ audio duration proves real-time-paced
frames (buffered-then-sent would arrive as one burst), and `MegakernelTTSService
TTFB ~58 ms` in the agent log.

All numbers are warm-engine (cold start ~60 s), batch 1, single GPU, host-side
`perf_counter` timing at the point a streaming consumer receives audio.
