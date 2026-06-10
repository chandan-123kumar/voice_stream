# Qwen3-TTS on the RTX 5090 Decode Megakernel

Real-time streaming TTS + a full voice agent, built by repurposing
[AlpinDale's qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
(one persistent CUDA kernel, ~1000 tok/s Qwen3-0.6B decode on an RTX 5090) as the
**talker decoder** of [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice).

**What's here:**
- `qwen_tts_megakernel/` — streaming TTS engine (megakernel talker decode)
- `qwen_tts_megakernel/pipecat_tts.py` — in-process Pipecat `TTSService`
- `voice_agent.py` — WebSocket voice agent (mic → STT → LLM → megakernel TTS → speaker) + browser client
- `app.py` — TTS-only web demo
- `bench_logs/voice_agent_bench.md` — honest end-to-end latency numbers
- `knowledge/ttfb-optimization.md` — how TTS TTFB got under 90 ms
- `assets/sample_agent_turn.wav` — a real recorded bot turn (+ metrics json)
- `conversation/` — the full Claude Code session that built this

## Results (RTX 5090, bf16 — re-measured 2026-06-10, full run in `bench_logs/criteria_benchmark_2026-06-10.md`)

| Metric | Value | Target |
|---|---|---|
| Talker decode rate | **1020 tok/s** (0.98 ms/step) | ~1000 (blog) |
| TTFC (time to first audio chunk) | **89 ms** (Pipecat service, 28 streamed chunks) | < 90 ms |
| RTF (wall / audio duration) | **0.163 – 0.166** | < 0.3 |
| Streaming | verified frame-by-frame (max inter-frame gap 77 ms) | required |

**vs. stock `qwen_tts`** (same weights, text, speaker, seed):

| | Stock | Megakernel |
|---|---|---|
| RTF | 1.048 (slower than real time) | **0.187** (5.6× faster) |
| Decode throughput | 11.5 frames/s | ~64 frames/s |
| Time to first audio | 6.3–7.8 s (no streaming) | **90 ms** |

Per-frame cost: megakernel step 0.79 ms · codec-head sample 0.04 ms ·
code predictor 8.9 ms · codec decode ~1.0 ms (amortized).
The megakernel is **not** the bottleneck — the 5-layer code predictor
(15 sequential codebooks/frame, vendored from
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) as a CUDA graph)
is ~80% of step time.

Reproduce: `python3 tests/bench_vs_baseline.py` (writes wavs for a listening comparison).

## Why almost no kernel changes were needed

The 0.6B talker backbone is **dimensionally identical** to Qwen3-0.6B
(hidden 1024, 28 layers, 16 Q / 8 KV heads, head_dim 128). The differences live outside the kernel:

| Difference | Where handled |
|---|---|
| `rope_theta` 1e6 (vs 1e4) | cos/sin tables rebuilt in Python |
| Interleaved M-RoPE `[24,20,20]` | reduces exactly to standard RoPE during decode (all 3 position streams equal, verified numerically) |
| Inputs are embeddings, not token ids | 1-row staging buffer + `token_id=0` |
| Codec head: 3072-entry vocab, untied | `LDG_VOCAB_SIZE` macro (the only kernel source change); top-k sampling via torch matvec |
| Variable-length prompt prefill | one HF forward; its KV cache copied into the kernel layout |

## Kernel bug found & fixed: entry-barrier race → deadlock

- `ldg_decode_kernel_direct` reset its grid-barrier words from *inside* the kernel (block 0), racing blocks already at the entry barrier — late blocks could spin forever.
- Rare with back-to-back launches (upstream's benchmark pattern); our decode loop (CUDA-graph replays + memsets between launches) deadlocked reliably within a few steps.
- **Fix:** zero barrier/flag words with `cudaMemsetAsync` on the host before each launch; drop the in-kernel reset and the entry barrier. Also shaves a few µs/step. See `csrc/kernel.cu` (`launch_ldg_decode_direct`).

## Numerical parity

- Per-layer outputs match HF (sdpa) to bf16 noise (layer-0 maxdiff 0.003).
- After 28 layers, hidden-state cosine ≈ 0.98 — **tighter than HF's own sdpa-vs-eager noise floor (0.952)**.
- Top-5 codec logits are the same set → numerically equivalent under top-k sampling. (`tests/test_parity.py`)

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

`voice_agent.py` runs the whole conversational pipeline **in one process** — the
TTS service calls the megakernel engine directly, no network hop to the GPU:

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

- **WebSocket, not WebRTC** — the demo rides an SSH tunnel, which can't carry WebRTC's UDP media. WebRTC is the production path on the open internet.
- **In-process TTS** — blocking decode loop on a persistent warmed thread; TTFB 82–87 ms. Barge-in stops decode at the next chunk boundary.
- **True streaming, verified** — reply audio reaches the browser as real-time-paced frames (116 frames over 4.56 s for 4.64 s of audio), never buffered-then-sent.
- **UI** — 9 Qwen voices switchable mid-conversation, live playback, per-turn session recordings with metrics (response ms, STT/LLM/TTS TTFB).

### End-to-end latency (full numbers in `bench_logs/voice_agent_bench.md`)

| Metric | Value |
|---|---|
| speech-end → first reply audio | 2.0–3.0 s typical (latest 3-turn run: 2.45/2.64/3.00 s) |
| of which local megakernel TTS | **0.09–0.18 s** (~5%) |
| of which OpenAI STT-final + LLM first token | ~1.3–3.5 s |
| of which turn-stop decision (VAD + Smart Turn) | ~1 s |

Local TTS is a rounding error; hosted-API round-trips and turn-taking dominate.
Switching from segmented to realtime STT cut the total from ~3.9 s to ~2.2 s (44%).

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

First start downloads the model and JIT-compiles the kernel (a few minutes);
after that, startup is ~60 s (CUDA graph capture).

```bash
# CLI synthesis, no server
python3 synthesize.py "Hello from the megakernel." --speaker ryan --out hello.wav

# streaming API
python3 - <<'PY'
from qwen_tts_megakernel.engine import MegakernelTTS
eng = MegakernelTTS()
for audio, sr, timing in eng.stream("Streaming speech!", speaker="ryan", language="English"):
    ...  # play / forward each float32 chunk as it arrives
PY

# tests & benchmarks
python3 tests/test_parity.py       # numerical parity vs HF
python3 tests/test_e2e.py          # end-to-end wav + HF baseline comparison
python3 tests/bench.py             # tok/s, component costs, TTFC, RTF
python3 tests/test_pipecat_tts.py  # Pipecat TTS service: streaming + TTFB
python3 tests/bench_e2e.py 3       # voice agent: speech-end -> reply latency
```

> **torchaudio note:** the stock wheel is ABI-incompatible with this container's
> NVIDIA torch build; a stub satisfying `qwen_tts`'s unused 25Hz-tokenizer import
> is installed instead. If startup fails with `OSError: libtorchaudio.abi3.so:
> undefined symbol`, rerun `bash scripts/fix_torchaudio_stub.sh`.

## Honest limitations / next steps

- **Code predictor is the optimization target** (10.8 ms/frame) — a second megakernel could plausibly take RTF from 0.18 to <0.05.
- **Batch size 1**, single utterance at a time (matches the megakernel's design); the voice agent serializes TTS behind one engine lock.
- **`max_seq_len` 4096** (≈5.5 min of audio incl. prompt) — KV cache is preallocated.
- **Agent latency is all remote** — OpenAI STT/LLM round-trips + the ~1 s turn-stop window. Next levers: tune VAD/turn-stop timing, or go fully local (whisper + small LLM on the same GPU, trading quality for ~1.5 s).
- **Raw PCM16 over WebSocket** (~384 kbps) — fine over a tunnel, wasteful on the internet (use WebRTC/Opus there).
