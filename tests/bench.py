"""Benchmarks: raw talker megakernel tok/s, per-component step costs, TTFC, RTF."""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts_megakernel.engine import MegakernelTTS

TEXT = ("The quick brown fox jumps over the lazy dog while the persistent "
        "megakernel streams audio frames in real time on a single GPU.")


def bench_talker_steps(eng, n=500):
    """Raw megakernel decode rate (talker backbone only)."""
    mk = eng.mk
    mk.reset()
    emb = torch.randn(1024, dtype=torch.bfloat16, device="cuda") * 0.02
    for _ in range(20):
        mk.step(emb)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mk.step(emb)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"talker megakernel: {n} steps in {dt*1000:.1f} ms -> "
          f"{dt/n*1000:.3f} ms/step = {n/dt:.0f} tok/s")
    mk.reset()


def bench_components(eng, n=200):
    """Per-component cost inside one decode iteration."""
    mk = eng.mk
    mk.reset()
    emb = torch.randn(1, 1, 1024, dtype=torch.bfloat16, device="cuda") * 0.02
    hidden = mk.step(emb.view(-1)).clone()
    past_hidden = hidden.to(torch.bfloat16).view(1, 1, -1)
    token = torch.tensor([100], device="cuda")
    codec_embed = eng.talker.get_input_embeddings()

    def timeit(label, fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        print(f"  {label:28s} {(time.perf_counter()-t0)/n*1000:7.3f} ms")

    timeit("talker megakernel step", lambda: (mk.reset(), mk.step(emb.view(-1))))
    timeit("codec head logits + sample", lambda: eng.mk.logits_from_hidden(hidden))
    pred_input = torch.cat((past_hidden, codec_embed(token.unsqueeze(1))), dim=1)
    timeit("code predictor (CUDA graph)", lambda: eng.predictor_graph.run(pred_input))
    codes = torch.randint(0, 2048, (25, 16), device="cuda")
    timeit("codec decode 25 frames", lambda: eng.speech_tokenizer.decode(
        {"audio_codes": codes.unsqueeze(0)}))
    mk.reset()


def bench_e2e(eng, n_runs=3):
    print("\nend-to-end streaming runs:")
    for run in range(n_runs):
        t0 = time.perf_counter()
        ttfc = None
        total_samples = 0
        sr = eng.sample_rate
        for audio, sr, timing in eng.stream(TEXT, speaker="ryan", language="English"):
            if ttfc is None:
                ttfc = (time.perf_counter() - t0) * 1000
            total_samples += len(audio)
        wall = time.perf_counter() - t0
        dur = total_samples / sr
        print(f"  run {run}: TTFC {ttfc:7.1f} ms | audio {dur:5.2f}s | wall {wall:5.2f}s "
              f"| RTF {wall/dur:.3f}")


def main():
    torch.manual_seed(0)
    eng = MegakernelTTS()
    for _ in eng.stream("Warm up run.", speaker="ryan", language="English", max_new_tokens=40):
        pass

    print("\n=== raw talker decode ===")
    bench_talker_steps(eng)

    print("\n=== per-component ===")
    bench_components(eng)

    bench_e2e(eng)


if __name__ == "__main__":
    main()
