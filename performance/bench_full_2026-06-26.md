# bench_full.py — Performance Report

**Date:** 2026-06-26  
**Script:** `tests/bench_full.py` (5 runs × 4 text lengths)  
**Config:** `first_chunk_frames=2`, `chunk_frames=12`

---

## Environment

| Field | Value |
|-------|-------|
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5090 |
| Driver | 595.71.05 |
| Power limit | 600 W |

---

## Raw Talker Decode

| Metric | Value |
|--------|-------|
| Step latency | 0.977 ms/step |
| Throughput | 1023 tok/s |

(500-step benchmark, 20-step warmup)

---

## TTFC / RTF by Text Length

### Very short (3 words) — "Hello, voice agent."

| Run | TTFC (ms) | Audio (s) | Wall (s) | RTF | Max gap (ms) |
|-----|-----------|-----------|----------|-----|-------------|
| 0 | 65.4 | 2.88 | 0.50 | 0.174 | 146.0 |
| 1 | 61.9 | 19.60 | 3.09 | 0.157 | 150.5 |
| 2 | 64.3 | 2.24 | 0.40 | 0.177 | 145.3 |
| 3 | 62.1 | 2.88 | 0.48 | 0.166 | 145.0 |
| 4 | 61.6 | 2.72 | 0.48 | 0.175 | 145.2 |
| **Median** | **62.1** | — | — | **0.174** | — |

Prefill median: 20.6 ms | RTF range: 0.157 – 0.177

---

### Short (7 words) — "Performance numbers should be measured honestly today."

| Run | TTFC (ms) | Audio (s) | Wall (s) | RTF | Max gap (ms) |
|-----|-----------|-----------|----------|-----|-------------|
| 0 | 62.3 | 4.56 | 0.76 | 0.166 | 145.6 |
| 1 | 62.4 | 4.00 | 0.64 | 0.161 | 145.9 |
| 2 | 62.1 | 3.76 | 0.64 | 0.169 | 146.7 |
| 3 | 62.2 | 7.04 | 1.12 | 0.159 | 145.7 |
| 4 | 61.7 | 5.92 | 0.93 | 0.158 | 146.0 |
| **Median** | **62.2** | — | — | **0.161** | — |

Prefill median: 20.2 ms | RTF range: 0.158 – 0.169

---

### Medium (22 words)

| Run | TTFC (ms) | Audio (s) | Wall (s) | RTF | Max gap (ms) |
|-----|-----------|-----------|----------|-----|-------------|
| 0 | 63.1 | 10.16 | 1.61 | 0.159 | 147.6 |
| 1 | 63.2 | 9.84 | 1.55 | 0.158 | 147.1 |
| 2 | 62.7 | 9.84 | 1.55 | 0.158 | 146.8 |
| 3 | 62.6 | 12.64 | 1.96 | 0.155 | 147.3 |
| 4 | 62.1 | 9.20 | 1.46 | 0.158 | 149.1 |
| **Median** | **62.7** | — | — | **0.158** | — |

Prefill median: 21.0 ms | RTF range: 0.155 – 0.159

---

### Long (48 words)

| Run | TTFC (ms) | Audio (s) | Wall (s) | RTF | Max gap (ms) |
|-----|-----------|-----------|----------|-----|-------------|
| 0 | 63.6 | 28.64 | 4.47 | 0.156 | 152.6 |
| 1 | 65.2 | 25.20 | 3.95 | 0.157 | 152.1 |
| 2 | 62.4 | 23.76 | 3.69 | 0.155 | 149.7 |
| 3 | 62.4 | 21.84 | 3.40 | 0.156 | 154.4 |
| 4 | 62.6 | 22.88 | 3.56 | 0.155 | 149.8 |
| **Median** | **62.6** | — | — | **0.156** | — |

Prefill median: 20.8 ms | RTF range: 0.155 – 0.157

---

## Overall (20 runs)

| Metric | Median | Min | Max |
|--------|--------|-----|-----|
| TTFC (ms) | **62.4** | 61.6 | 65.4 |
| RTF | **0.158** | 0.155 | 0.177 |

---

## Per-Component Step Cost (200 iterations)

| Component | Cost (ms) |
|-----------|-----------|
| Talker megakernel step | 0.784 |
| Codec head logits + sample | 0.023 |
| Code predictor (CUDA graph, 15 codebooks) | 8.896 |
| Codec decode — 2-frame window | 20.039 |
| Codec decode — 37-frame window | 21.278 |

---

## Quality: fcf=2 vs fcf=4 Waveform Correlation

Same seed, medium text (22 words):

| Metric | Value |
|--------|-------|
| fcf=2 audio length | 10.16 s |
| fcf=4 audio length | 10.16 s |
| Same length | ✓ |
| Waveform correlation | **0.999165** |
| Max absolute diff | 0.073181 |

The new default (`first_chunk_frames=2`) is bit-similar to the old default (`first_chunk_frames=4`), confirming no quality regression from the latency change.
