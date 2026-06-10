---
name: qwen-tts-megakernel-project
description: Take-home project state — Qwen3-TTS talker decode via AlpinDale megakernel on RTX 5090
metadata: 
  node_type: memory
  type: project
  originSessionId: da043312-2e5c-4cbb-a40a-7d5de024b3fc
---

Take-home (started 2026-06-09): run Qwen3-TTS talker decoder on AlpinDale's qwen_megakernel, stream audio. User said: focus on voice generation only — skip Pipecat/server unless asked.

Key facts established:
- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (public; `Qwen/Qwen3-TTS` is a wrong/404 repo id). Talker backbone == Qwen3-0.6B dims exactly (1024 hidden, 3072 inter, 28 layers, 16Q/8KV, head_dim 128, vocab 3072, rope_theta 1e6).
- Interleaved M-RoPE [24,20,20] reduces to standard rotate-half RoPE during decode (all 3 position streams equal); rope_deltas = 0.
- Kernel needs no dim changes; only LDG_VOCAB_SIZE made a macro. Input embeddings fed via 1-row staging "embed table" + token_id=0 trick.
- Project at /root/qwen_tts_megakernel (engine.py = streaming TTS; talker_megakernel.py = kernel wrapper; vendored predictor_graph/sampling from andimarafioti/faster-qwen3-tts).
- Prefill via HF talker.forward → DynamicCache copied into kernel cache layout; decode steps via megakernel; codec head + top-k sampling in torch; 5-layer code predictor as CUDA graph; 12Hz codec decodes chunks with 25-frame sliding context.
- Parity: mk-vs-sdpa cos 0.98 at layer 28; noise floor sdpa-vs-eager is cos 0.952 — kernel within numerical noise.
- DONE (2026-06-09): voice generation works. Results: 1018 tok/s talker decode, TTFC ~89ms, RTF ~0.18. Bottleneck = code predictor CUDA graph (10.8ms/frame, 81% of step). Output at /root/qwen_tts_megakernel/out_megakernel.wav.
- Found+fixed real kernel bug: entry-barrier race in ldg_decode_kernel_direct (in-kernel reset of barrier_sense races blocks already publishing sense=1 → deadlock when launches interleave with CUDA-graph replays). Fix: host-side cudaMemsetAsync of barrier words + removed in-kernel reset/entry barrier, AtomicGridSync local_gen=0.
- Env gotchas: container torch is custom NVIDIA build (2.10.0a0+nv26.1) — stock torchaudio wheels ABI-incompatible; torchaudio is STUBBED at /usr/local/lib/python3.12/dist-packages/torchaudio (only kaldi import needed, unused by 12Hz path). Build flag -arch=sm_120a.
