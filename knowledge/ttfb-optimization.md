# Reducing TTFB for the in-process Pipecat TTS service

How we got time-to-first-audio through the Pipecat pipeline from **~124 ms down to ~75–87 ms**
(target: < 90 ms), measured on an RTX 5090 with `tests/test_pipecat_tts.py`.

## Baseline

The first working version of `MegakernelTTSService` (qwen_tts_megakernel/pipecat_tts.py)
spawned a fresh `threading.Thread` per utterance to run the blocking
`MegakernelTTS.stream()` loop, with the engine's default `first_chunk_frames=4`.

- Pipecat-measured TTFB: **~164 ms** (cold), **~115–124 ms** warm.
- Raw engine TTFA in a plain main-thread benchmark: **~62 ms** warm.

So ~50–60 ms was being lost somewhere between the engine and the pipeline.

## Finding 1 — warm up the codec decode for the chunk shape you use

The first generation after changing `first_chunk_frames` is ~40 ms slower: the
speech-tokenizer decode path is cold for the new first-chunk shape. The warmup
generation must use the **same `first_chunk_frames`** as production traffic
(see `_warm_thread()` in the service, and the warmup in the test).

## Finding 2 (the big one) — never use a fresh thread per utterance

Instrumentation showed the entire remaining overhead was *inside* the worker
thread: even the HF prefill forward ran ~10 ms slower than in the main thread.

Isolation benchmark (same engine, same text, warm):

| execution context                  | TTFA |
|------------------------------------|----------|
| main thread                        | ~66 ms |
| fresh `threading.Thread` each time | ~98–108 ms |
| reused pool thread (`asyncio.to_thread`) | ~68–72 ms after first use |

**Root cause:** PyTorch keeps CUDA state per-thread (cuBLAS handles and other
thread-local init). A brand-new thread pays ~35 ms one-time setup on its first
kernels — every utterance, if you spawn a thread every utterance.

**Fix:** the service owns a persistent `ThreadPoolExecutor(max_workers=1)`;
all generations run on that one thread, and a short warmup generation at init
(`warmup=True`) pre-pays the per-thread CUDA init before the first real
utterance. This also naturally serializes utterances (single KV cache).

## Finding 3 — smaller first chunk

With the thread fixed, shrinking the first codec chunk gives the rest:

| `first_chunk_frames` | Pipecat TTFB |
|----------------------|--------------|
| 4 (engine default)   | ~102 ms |
| 2 (service default)  | ~82–87 ms ✓ |
| 1                    | ~72–75 ms ✓ |

Each codec frame is ~83 ms of audio at 12 Hz and decode runs well below
real-time (RTF ≈ 0.18), so the playback buffer never underruns even with a
1-frame first chunk. The service defaults to 2 as a quality margin; pass
`gen_params={"first_chunk_frames": 1}` for the lowest latency.

## Reproduce

```bash
python3 tests/test_pipecat_tts.py      # service default (first_chunk_frames=2)
python3 tests/test_pipecat_tts.py 1    # override first_chunk_frames
```

The test runs a minimal `Pipeline([MegakernelTTSService, collector])`, pushes
two `TTSSpeakFrame`s, asserts chunks stream incrementally, prints TTFB, and
writes the audio to `/tmp/pipecat_tts_test.wav`.
