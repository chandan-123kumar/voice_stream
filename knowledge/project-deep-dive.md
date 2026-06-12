# Project deep-dive — how every piece works and why

A topic-by-topic explainer of the whole system, with the numbers and reasoning
behind each decision. Companion to the README (what it is) and
[`bench_logs/performance.md`](../bench_logs/performance.md) (current numbers +
how to verify them).

## 1. The megakernel

- One **persistent CUDA kernel** — 128 thread blocks × 512 threads, launched once —
  executes all 28 transformer layers per decode step internally, instead of hundreds
  of separate kernel launches per token. Blocks synchronize between layers with
  grid-wide barriers (atomics + sense flags), since there is no launch boundary to
  act as one.
- **Why decode is bandwidth-bound**: batch-1 autoregressive decode reads every weight
  once per token with almost no compute reuse, so tok/s ≈ memory bandwidth ÷ model
  bytes. That is why the kernel targets ~71% of GDDR7 bandwidth, why it stays bf16
  (quantization is a different trade, explicitly out of scope), and why ~1,000 tok/s
  is the right ballpark for 0.6B on an RTX 5090. We measure **~1,030 tok/s**
  (0.97 ms/step).
- Built for sm_120 (Blackwell); compiled with `-arch=sm_120a`.

## 2. Qwen3-TTS architecture — the three networks

| Network | Size/shape | Role | Per-frame cost here |
|---|---|---|---|
| **Talker decoder** | 0.6B, 28 layers, hidden 1024, 16Q/8KV, head_dim 128 — dimensionally identical to Qwen3-0.6B | generates **codebook 0**, one token per 12 Hz frame, autoregressively over the utterance | **0.77 ms** (megakernel) |
| **Code predictor** ("codebook generator") | small 5-layer transformer, 15 lm-heads/embeddings | generates **codebooks 1–15** within each frame, sequentially, conditioned on talker hidden + codebook-0 embed | 8.9 ms (stock HF wrapped in a CUDA graph) |
| **12 Hz codec decoder** | conv + transformer vocoder | 16-token frames → 24 kHz audio (2,000 samples/frame) | ~12.7 ms flat per chunk |

One frame = 83.3 ms of audio = 16 codec tokens. The assignment targets the talker
(the Qwen3-shaped LLM decode loop) and explicitly scopes out the codebook generator —
which still must run to produce audio, just not on the megakernel.

## 3. Adapting the kernel to the talker — what actually changed

- **Kernel source**: one change — `LDG_VOCAB_SIZE` became a macro (3,072-entry untied
  codec head vs the LLM vocab). Everything else fits because the talker backbone has
  exactly Qwen3-0.6B dimensions.
- **rope_theta 1e6 (vs 1e4)** → cos/sin tables rebuilt in Python; no kernel change.
- **Interleaved M-RoPE [24,20,20]** → reduces exactly to standard rotate-half RoPE
  during decode, because all three position streams advance together (M-RoPE only
  differentiates positions for multimodal prefill); verified numerically,
  rope_deltas = 0.
- **Embedding inputs**: the talker consumes embeddings (Σ of 16 codec-group embeds +
  text hidden), not token ids → fed through a 1-row staging "embedding table" with
  `token_id=0`.
- **Prefill**: variable-length prompt runs through one HF forward; its KV cache is
  copied into the kernel's cache layout. The kernel only ever does decode.
- **Sampling**: the kernel natively does argmax; TTS needs top-k-50 / temp 0.9 /
  repetition penalty 1.05 → codec-head logits are computed by a torch matvec from
  the kernel's final-norm hidden, sampled HF-style on device.

## 4. The kernel bug found and fixed (entry-barrier race → deadlock)

`ldg_decode_kernel_direct` reset its grid-barrier words from **inside** the kernel
(block 0), racing blocks already publishing their sense flag at the entry barrier —
a late block could spin forever. Upstream's back-to-back launch benchmark rarely hit
it; our decode loop interleaves CUDA-graph replays (code predictor) and memsets
between launches, changing timing enough to deadlock within a few steps, reliably.
**Fix**: zero the barrier/flag words with `cudaMemsetAsync` from the host before each
launch and drop the in-kernel reset + entry barrier. Also saves a few µs/step.
See `csrc/kernel.cu` (`launch_ldg_decode_direct`).

## 5. How we know the kernel output is correct

- Per-layer outputs match HF (sdpa) to bf16 noise; after 28 layers the hidden-state
  cosine is ≈ 0.98 — **tighter than HF's own sdpa-vs-eager noise floor of 0.952**.
  Same model, within numerical noise, by HF's own internal disagreement standard.
- Top-k codec logit sets match → equivalent under top-k sampling.
- Known flake: `tests/test_parity.py` asserts strict greedy-argmax equality and fails
  1/24 steps as of 2026-06-11 (bf16 noise flips two near-equal logits; both tokens are
  inside the real top-k-50 set). Verified pre-existing — details in
  [`bench_logs/performance_history.md`](../bench_logs/performance_history.md).

## 6. Performance: numbers, definitions, budget

- **TTFC 47.0 ms median** (target < 60), **RTF 0.145 median** (target < 0.15;
  utterances under ~3 s reach 0.150–0.158 — see below), **~1,030 tok/s**.
  vs stock `qwen_tts` same-day: RTF 0.738 → 0.148 (**5.0×**), first audio
  4.6–5.0 s → 47 ms.
- **Definitions** (deliberately inclusive): TTFC = `stream()` call → first PCM chunk,
  including prompt build, HF prefill, 2 decode frames, and the codec decode. RTF =
  total wall ÷ emitted audio duration, including prefill and every codec decode.
  Warm engine, 5 runs × 4 text lengths, host-side `perf_counter`.
- **Per-frame budget** (83.3 ms of audio): code predictor 8.9 ms (**86%**) ·
  megakernel step 0.77 ms (7%) · codec decode ~1.1 ms amortized (12.7 ms flat per
  12-frame chunk) · sampling/glue ~0.7 ms.
- **The TTFC optimization**: profiling showed the codec decoder costs a flat ~12 ms
  at any window size 1–37 (launch-bound, not compute-bound) — so accumulating 4
  frames before the first decode was pure waiting. Default `first_chunk_frames`
  4 → 2 cut TTFC 66–69 → 47 ms; same-seed audio is identical in length, waveform
  correlation 0.9993.
- **Why short utterances miss 0.15**: fixed costs over few frames. A ~2.6 s
  utterance (~31 frames) pays ~15 ms prefill + 4 codec decodes × ~12.7 ms; a 21 s
  one pays one decode per 12 frames at steady state.
- **Explored, not merged**: per-window-size CUDA graphs for the codec decoder cut it
  to 2.4–4.8 ms with bit-identical output → RTF 0.133–0.139 at every length, TTFC
  37.5 ms. Reverted for simplicity. `torch.compile(mode="reduce-overhead")` was
  rejected outright: similar speed but it changes bf16 numerics (max waveform diff
  0.2–0.5) — verifiability beat speed.
- The step-4 stretch target RTF < 0.1 is not met: 86% of decode time is the code
  predictor, the component scoped out of the megakernel work. A second megakernel
  for its 5-layer × 15-codebook loop is the documented path to RTF < 0.05.

## 7. Streaming — design and proof

- Chunking: first audio after 2 frames (~167 ms), then every 12 frames (~1 s), with
  a 25-frame sliding context for codec continuity; `samples_per_frame` calibrates
  once ≥ 25 frames are decoded. A 1-frame first chunk would *hurt*: its 83 ms of
  audio drains before the next chunk lands (~133 ms) — audible gap.
- **Proof it is not buffered-then-sent** (measured at the WebSocket client):
  190 frames over 7.52 s for 7.60 s of audio — arrival span ≈ audio duration, max
  inter-frame gap 56 ms. Buffered audio would arrive as one burst (span ≈ 0).
  **Barge-in** is the second proof: speaking over the bot halts the stream ~0.9 s
  after speech onset (VAD detection + chunk boundary + queued transport frames) —
  impossible if the reply had already been shipped.
- Reproduce: `tests/bench_e2e.py` prints the streaming profile per turn.

## 8. The voice agent (Pipecat)

- Pipeline, all in one process: browser mic → WebSocket (protobuf) →
  `FastAPIWebsocketTransport` → Silero VAD + Smart Turn v3 → **local faster-whisper
  large-v3-turbo STT (same GPU)** → OpenAI LLM (gpt-4o-mini) → `MegakernelTTSService`
  (in-process) → transport out. 9 switchable voices, per-turn recordings + metrics.
- **WebSocket, not WebRTC**: the demo rides an SSH tunnel (TCP only); WebRTC/Opus is
  the stated production path. Raw PCM16 ≈ 384 kbps — fine on a tunnel.
- **In-process TTS, not a TTS server**: no network hop to the GPU; blocking decode on
  one persistent warmed thread (a fresh thread costs ~35 ms of cuBLAS/thread-local
  init); a single engine lock serializes utterances (megakernel is batch-1 by design).
- **E2E latency**: speech-end → first reply audio **0.7–1.1 s median**. History:
  ~3.9 s (segmented hosted STT) → ~2.2 s (hosted realtime STT) → 0.7–1.1 s (local
  Whisper). The unlock was a log-timeline analysis showing the dominant cost was
  waiting ~1.4 s for the hosted STT final transcript — the LLM cannot start without
  it. **Current bottleneck: OpenAI LLM first token (0.4–1.0 s, high variance)**;
  a gpt-4.1-nano A/B landed within gpt-4o-mini's variance, so the default stayed.


