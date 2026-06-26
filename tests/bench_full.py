"""Reproduce bench_logs/performance_run_2026-06-11.log.

20 runs × 4 text lengths with median TTFC, RTF, prefill, max inter-frame gap,
plus per-component costs (talker step, codec head, predictor, codec decode at
2-frame and 37-frame windows) and a same-seed fcf=2 vs fcf=4 waveform
correlation check.

This is the extended harness behind the criteria report; `tests/bench.py` is
the slimmer 3-runs-of-one-text version.

Usage:
    python3 tests/bench_full.py                  # full matrix + components + quality
    python3 tests/bench_full.py --quick          # 2 runs per text instead of 5
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts_megakernel.engine import MegakernelTTS

# Four text lengths spanning the expected production distribution.
# Word counts match the labels (approximate).
TEXTS = {
    "very short (3 words)":
        "Hello, voice agent.",
    "short (7 words)":
        "Performance numbers should be measured honestly today.",
    "medium (22 words)":
        "The quick brown fox jumps over the lazy dog while the persistent "
        "megakernel streams audio frames in real time on a single GPU.",
    "long (48 words)":
        "The persistent megakernel approach to small autoregressive model "
        "decoding eliminates the per-step CUDA kernel launch overhead that "
        "dominates batch one inference, dropping the talker decode latency "
        "to under a millisecond per token while leaving the rest of the "
        "speech synthesis pipeline running through standard PyTorch operators.",
}
SEED = 0


def median(xs):
    return statistics.median(xs) if xs else 0.0


def bench_text(eng, label, text, n_runs):
    """Stream the text n_runs times; return per-run metrics."""
    rows = []
    for run in range(n_runs):
        torch.manual_seed(SEED + run)
        t0 = time.perf_counter()
        ttfc = None
        prefill_ms = None
        total_samples = 0
        sr = eng.sample_rate
        last_yield = None
        max_gap_ms = 0.0
        for audio, sr, timing in eng.stream(text, speaker="ryan", language="English"):
            now = time.perf_counter()
            if ttfc is None:
                ttfc = (now - t0) * 1000
                prefill_ms = timing.get("prefill_ms", 0)
            else:
                gap_ms = (now - last_yield) * 1000
                if gap_ms > max_gap_ms:
                    max_gap_ms = gap_ms
            last_yield = now
            total_samples += len(audio)
        wall = time.perf_counter() - t0
        dur = total_samples / sr if total_samples else 0.0
        rtf = wall / dur if dur > 0 else 0.0
        rows.append({
            "ttfc": ttfc or 0.0, "audio": dur, "wall": wall, "rtf": rtf,
            "max_gap": max_gap_ms, "prefill": prefill_ms or 0.0,
        })
        print(f"  {label:22s} run {run}: TTFC {ttfc:7.1f} ms | audio {dur:5.2f}s | "
              f"wall {wall:5.2f}s | RTF {rtf:.3f} | max gap {max_gap_ms:.1f} ms")

    ttfcs = [r["ttfc"] for r in rows]
    rtfs = [r["rtf"] for r in rows]
    prefills = [r["prefill"] for r in rows]
    print(f"  {label:22s} -> TTFC median {median(ttfcs):.1f} ms | "
          f"RTF median {median(rtfs):.3f} (min {min(rtfs):.3f} max {max(rtfs):.3f}) | "
          f"prefill median {median(prefills):.1f} ms")
    return rows


def bench_components(eng, n=200):
    """Per-component cost. Mirrors the report's component table."""
    print("\n=== per-component step cost (200 iters) ===")
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
        print(f"  {label:44s} {(time.perf_counter() - t0) / n * 1000:7.3f} ms")

    timeit("talker megakernel step",
           lambda: (mk.reset(), mk.step(emb.view(-1))))
    timeit("codec head logits + sample",
           lambda: eng.mk.logits_from_hidden(hidden))
    pred_input = torch.cat((past_hidden, codec_embed(token.unsqueeze(1))), dim=1)
    timeit("code predictor (CUDA graph, 15 codebooks)",
           lambda: eng.predictor_graph.run(pred_input))
    codes_2 = torch.randint(0, 2048, (2, 16), device="cuda")
    codes_37 = torch.randint(0, 2048, (37, 16), device="cuda")
    timeit("codec decode 2-frame window",
           lambda: eng.speech_tokenizer.decode({"audio_codes": codes_2.unsqueeze(0)}))
    timeit("codec decode 37-frame window",
           lambda: eng.speech_tokenizer.decode({"audio_codes": codes_37.unsqueeze(0)}))
    mk.reset()


def bench_quality(eng):
    """Same-seed fcf=2 vs fcf=4: prove the default change is bit-similar."""
    print("\n=== quality: same-seed waveform, first_chunk_frames 2 (new default) vs 4 (old) ===")
    text = TEXTS["medium (22 words)"]
    sr = eng.sample_rate

    torch.manual_seed(SEED)
    chunks_2 = [a for a, _, _ in eng.stream(
        text, speaker="ryan", language="English", first_chunk_frames=2)]
    audio_2 = np.concatenate(chunks_2).astype(np.float32)

    torch.manual_seed(SEED)
    chunks_4 = [a for a, _, _ in eng.stream(
        text, speaker="ryan", language="English", first_chunk_frames=4)]
    audio_4 = np.concatenate(chunks_4).astype(np.float32)

    same_length = len(audio_2) == len(audio_4)
    print(f"  fcf=2: {len(audio_2)/sr:.2f}s | fcf=4: {len(audio_4)/sr:.2f}s | "
          f"same length: {same_length}")
    n = min(len(audio_2), len(audio_4))
    if n > 0:
        a2, a4 = audio_2[:n], audio_4[:n]
        denom = (np.std(a2) * np.std(a4))
        corr = float(np.mean((a2 - a2.mean()) * (a4 - a4.mean())) / denom) if denom > 0 else 0.0
        max_diff = float(np.max(np.abs(a2 - a4)))
        print(f"  waveform correlation: {corr:.6f} | max abs diff: {max_diff:.6f}")


def print_env():
    print("=== environment ===")
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            out = ""
        print(f"{name}{(', ' + out) if out else ''}")


def bench_raw_talker(eng, n=500):
    print(f"\n=== raw talker decode ({n} steps) ===")
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
    print(f"{dt/n*1000:.3f} ms/step = {n/dt:.0f} tok/s")
    mk.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="2 runs per text instead of 5 (8 total runs)")
    args = ap.parse_args()
    n_runs = 2 if args.quick else 5

    print_env()
    eng = MegakernelTTS()

    print("\nwarming up...")
    for _ in eng.stream("Warm up run.", speaker="ryan", language="English",
                        max_new_tokens=40):
        pass

    bench_raw_talker(eng)

    print(f"\n=== TTFC / RTF, default config "
          f"(first_chunk_frames=2, chunk_frames=12), {n_runs} runs each ===")
    all_rows = []
    for label, text in TEXTS.items():
        rows = bench_text(eng, label, text, n_runs)
        all_rows.extend(rows)

    total = len(all_rows)
    all_ttfc = [r["ttfc"] for r in all_rows]
    all_rtf = [r["rtf"] for r in all_rows]
    print(f"\nOVERALL ({total} runs): TTFC median {median(all_ttfc):.1f} ms "
          f"(min {min(all_ttfc):.1f}, max {max(all_ttfc):.1f}) | "
          f"RTF median {median(all_rtf):.3f} "
          f"(min {min(all_rtf):.3f}, max {max(all_rtf):.3f})")

    bench_components(eng)
    bench_quality(eng)


if __name__ == "__main__":
    main()
