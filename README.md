# Qwen3-TTS on the RTX 5090 Decode Megakernel

[AlpinDale's qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) (single
persistent CUDA kernel decoding Qwen3-0.6B at ~1000 tok/s on an RTX 5090) repurposed
as the **talker decoder** of [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice),
streaming real-time speech — plus a full **Pipecat voice agent** (mic → STT → LLM →
megakernel TTS → speaker) built on top of it.

Quick tour:
- `qwen_tts_megakernel/` — the streaming TTS engine (megakernel talker decode)
- `qwen_tts_megakernel/pipecat_tts.py` — in-process Pipecat `TTSService`
- `voice_agent.py` — WebSocket voice agent demo + browser client
- `app.py` — TTS-only web demo (type text, hear it stream)
- `bench_logs/voice_agent_bench.md` — honest end-to-end latency numbers
- `knowledge/ttfb-optimization.md` — how TTS TTFB got under 90 ms
- `assets/sample_agent_turn.wav` — a real recorded bot turn (+ metrics json)
- `conversation/` — the full Claude Code session that built this

## Results (RTX 5090, bf16)

| Metric | Value | Target |
|---|---|---|
| Talker decode rate | **1018 tok/s** (0.98 ms/step) | ~1000 (blog) |
| TTFC (time to first audio chunk) | **~89 ms** | < 90 ms |
| RTF (wall / audio duration) | **~0.18** | < 0.3 |
| Streaming | chunk-by-chunk, ~1 s chunks, 333 ms first chunk | required |

Per-iteration cost breakdown (one 12 Hz codec frame):

| Component | ms |
|---|---|
| Talker megakernel step (28 layers) | 0.79 |
| Codec-head logits + top-k sample | 0.02 |
| Code predictor, 15 codebooks (CUDA graph) | 10.75 |
| 12 Hz codec decode (amortized, 17.9 ms / 12 frames) | ~1.5 |

The megakernel is **not** the bottleneck — the 5-layer code predictor (15 sequential
sub-steps per frame, vendored from
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) as a CUDA graph)
dominates at ~81% of step time.

### vs. stock qwen_tts (same weights, text, speaker, seed)

| | Stock `qwen_tts` | Megakernel engine |
|---|---|---|
| RTF | 1.048 (slower than real time) | **0.187** (5.6× faster) |
| Decode throughput | 11.5 frames/s | ~64 frames/s |
| Time to first audio | 6.3–7.8 s (no streaming) | **90 ms** |

Reproduce with `python3 tests/bench_vs_baseline.py` (writes `cmp_stock.wav` /
`cmp_megakernel.wav` for a listening comparison).

## Why almost no kernel changes were needed

The 0.6B talker backbone is **dimensionally identical** to Qwen3-0.6B: hidden 1024,
intermediate 3072, 28 layers, 16 Q / 8 KV heads, head_dim 128, RMS eps 1e-6. The real
differences are handled outside the kernel:

| Difference | Where handled |
|---|---|
| `rope_theta` 1e6 (vs 1e4) | cos/sin tables rebuilt in Python (`talker_megakernel.py`) |
| Interleaved M-RoPE `[24,20,20]` | reduces *exactly* to standard rotate-half RoPE during decode because all 3 position streams are equal (verified numerically); kernel RoPE used as-is |
| Inputs are embeddings, not token ids (codec-embedding sums + projected text hiddens) | kernel reads `embed_weight + token_id*H`; we pass a 1-row staging buffer + `token_id=0` |
| Codec head: 3072-entry vocab, untied | `LDG_VOCAB_SIZE` made a compile-time macro (the only kernel source change for adaptation); top-k sampling via a trivial torch matvec on the kernel's final-norm output |
| Variable-length prompt prefill | one HF forward; its DynamicCache (post-RoPE K) is copied into the kernel's `[layer, kv_head, pos, head_dim]` cache layout |

## Kernel bug found & fixed: entry-barrier race → deadlock

`ldg_decode_kernel_direct` reset its grid-barrier words (`barrier_counter`,
`barrier_sense`) from *inside* the kernel (block 0), racing against other blocks
already arriving at the entry barrier. If block 0's reset of `barrier_sense` landed
after another block had published `sense=1`, late blocks spun on `sense==0` forever.
Back-to-back identical launches (the upstream benchmark pattern) rarely trigger it,
but interleaving CUDA-graph replays and memsets between launches (our decode loop)
skews block start times and deadlocked reliably within a few steps.

Fix: zero the barrier/flag words with stream-ordered `cudaMemsetAsync` on the host
before each launch and drop the in-kernel reset *and* the entry barrier entirely
(the first `grid.sync()` with `local_gen=0` is the rendezvous). This also shaves a
few µs per step. See `csrc/kernel.cu` (`launch_ldg_decode_direct`).

## Numerical parity

Compared against the HF talker (sdpa) on identical prefill + inputs:
- per-layer outputs match to bf16 noise (layer-0 maxdiff 0.003);
- after 28 layers, hidden-state cosine ≈ 0.98 — **tighter than HF's own
  sdpa-vs-eager noise floor (cosine 0.952)** on the same step;
- top-5 codec logits are the same set. Under temperature-0.9 top-k sampling this is
  numerically equivalent. (`tests/test_parity.py`, `tests/debug_prefill.py`)

## Architecture

```
text ─ qwen_tts prompt build ─► HF prefill (1 forward) ──► KV → kernel cache
                                        │
            ┌── per frame ──────────────▼──────────────────────────────┐
            │ megakernel step (0.79ms) → final-norm hidden             │
            │   → codec-head matvec → repetition penalty → top-k sample│
            │   → code predictor CUDA graph → 15 codebooks             │
            │   → next input embed = Σ 16 codec embeds + text hidden   │
            └──────────────┬───────────────────────────────────────────┘
                           ▼ every 4 (first) / 12 frames
            12 Hz codec decoder (25-frame sliding context) → audio chunk
```

## Voice agent (Pipecat integration)

`voice_agent.py` runs the complete conversational pipeline **in one process** —
the TTS service calls the megakernel engine directly, no network hop between
the pipeline and the GPU:

```
browser mic ──┐                                                  ┌── browser speaker
              │  WebSocket: protobuf frames over the SSH tunnel  │
              ▼                                                  │
   FastAPIWebsocketTransport.input()                 transport.output()
              │                                                  ▲
              ▼                                                  │
   ControlChannel (speaker switch msgs)                   TurnRecorder ── async wav+metrics
              │                                                  ▲          to voices/
              ▼                                                  │
   OpenAI Realtime STT (gpt-realtime-whisper, streams while you speak)
              │                                                  │
              ▼                                                  │
   user context aggregator (Silero VAD + Smart Turn v3 turn-taking)
              │                                                  │
              ▼                                                  │
   OpenAI LLM (gpt-4o-mini) ──► MegakernelTTSService (in-process, local GPU)
```

- **Transport: WebSocket, not WebRTC** — the demo rides an SSH local-forward,
  which can't carry WebRTC's UDP media. WebRTC is the production path for
  clients on the open internet; over a tunnel WebSocket is strictly simpler.
- **In-process TTS** (`qwen_tts_megakernel/pipecat_tts.py`): a Pipecat
  `TTSService` running the blocking decode loop on a persistent warmed thread,
  handing PCM to the pipeline as each codec chunk decodes. TTFB 82–87 ms
  (`first_chunk_frames=2`). Barge-in interruption stops decode at the next
  chunk boundary.
- **Frame-by-frame streaming, verified**: reply audio reaches the browser as
  real-time-paced WebSocket frames (e.g. 116 frames over 4.56 s for 4.64 s of
  audio, max gap 79 ms) — never buffered-then-sent.
- **UI**: speaker dropdown (9 Qwen voices, switchable mid-conversation), live
  playback, and a "Get session recordings" button that lists every bot turn
  of the session with its saved audio and metrics (response ms, STT/LLM/TTS
  TTFB) served from disk.

### End-to-end latency (RTX 5090, full numbers in `bench_logs/voice_agent_bench.md`)

| Metric | Value |
|---|---|
| speech-end → first reply audio | 2.0–2.6 s typical (3.8–5.3 s on bad OpenAI days) |
| of which local megakernel TTS | **0.09–0.18 s** (~5%) |
| of which OpenAI STT-final + LLM first token | ~1.3–3.5 s |
| of which turn-stop decision (VAD + Smart Turn) | ~1 s |

The local TTS is a rounding error in the response time; the latency budget is
dominated by hosted-API round-trips and turn-taking — switching from segmented
to realtime STT cut the total from ~3.9 s to ~2.2 s (44%).

## Setup

```bash
# Hardware/stack: RTX 5090 (sm_120), CUDA 12.8+, torch with sm_120 support
pip install -r requirements.txt
bash scripts/fix_torchaudio_stub.sh    # see torchaudio note below

# Voice agent only: OpenAI key for STT + LLM
cp .env.example .env                   # then fill in OPENAI_API_KEY

# On the GPU server
python3 app.py                         # TTS-only demo  -> :8000
python3 voice_agent.py                 # voice agent    -> :8001

# From your machine (both UIs ride an SSH tunnel)
ssh -L 8000:localhost:8000 -L 8001:localhost:8001 -p <port> root@<server>
# open http://localhost:8000 (TTS demo) / http://localhost:8001 (voice agent)
```

First start downloads the model from HuggingFace and JIT-compiles the CUDA
kernel (a few minutes); after that, startup is ~60 s (CUDA graph capture).

```bash
# CLI synthesis without any server:
python3 synthesize.py "Hello from the megakernel." --speaker ryan --out hello.wav

# streaming API
python3 - <<'PY'
from qwen_tts_megakernel.engine import MegakernelTTS
eng = MegakernelTTS()
for audio, sr, timing in eng.stream("Streaming speech!", speaker="ryan", language="English"):
    ...  # play / forward each float32 chunk as it arrives
PY

# tests & benchmarks
python3 tests/test_parity.py     # numerical parity vs HF
python3 tests/test_e2e.py        # end-to-end wav + HF baseline comparison
python3 tests/bench.py           # tok/s, component costs, TTFC, RTF
python3 tests/test_pipecat_tts.py  # Pipecat TTS service: streaming + TTFB
python3 tests/bench_e2e.py 3     # voice agent: speech-end -> reply latency
```

Note: in this container the stock `torchaudio` wheel is ABI-incompatible with the
NVIDIA torch build; a stub satisfying `qwen_tts`'s unused 25Hz-tokenizer import is
installed instead (the 12Hz path never calls it). If anything fails to start with
`OSError: libtorchaudio.abi3.so: undefined symbol: ...`, a pip install has pulled
the real torchaudio back in — run `bash scripts/fix_torchaudio_stub.sh` to fix.

## Honest limitations / next steps

- The code predictor (10.8 ms/frame) is the obvious optimization target — a second
  megakernel or a leaner handwritten loop could plausibly take RTF from 0.18 to <0.05.
- Greedy argmax from the kernel's fused LM head is available "for free" but unused;
  TTS quality needs top-k sampling (done in torch, 0.02 ms).
- Batch size 1 only, single utterance at a time (matches the megakernel's design).
  The voice agent serializes TTS across connections behind one engine lock.
- `max_seq_len` 4096 (≈5.5 min of audio incl. prompt) — KV cache is preallocated.
- Voice agent latency is dominated by OpenAI round-trips (STT final ~1 s, LLM
  first token 0.4–2.3 s) and the ~1 s turn-stop window — none of it local. Next
  levers: tune VAD/turn-stop timing, or move STT/LLM local (whisper + a small
  LLM on the same GPU would trade quality for ~1.5 s).
- The browser client speaks raw PCM16 over WebSocket (~384 kbps) — fine over a
  tunnel, wasteful over the internet (where WebRTC/Opus is the answer anyway).
