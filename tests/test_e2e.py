"""End-to-end: megakernel-driven Qwen3-TTS streaming synthesis to wav."""

import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts_megakernel.engine import MegakernelTTS

TEXT = ("Hello! This is the Qwen3 text to speech model, running its talker "
        "decoder inside a single persistent CUDA megakernel on an RTX 5090.")


def main():
    torch.manual_seed(0)
    eng = MegakernelTTS()

    # Warmup run (JIT, allocator, codec)
    for _ in eng.stream("Warm up.", speaker="ryan", language="English", max_new_tokens=32):
        pass

    print("\n=== streaming synthesis ===")
    t0 = time.perf_counter()
    chunks = []
    sr = eng.sample_rate
    ttfc = None
    for audio, sr, timing in eng.stream(TEXT, speaker="ryan", language="English"):
        now = time.perf_counter()
        if ttfc is None:
            ttfc = (now - t0) * 1000
        chunks.append(audio)
        print(f"chunk {timing['chunk_index']:3d}: {len(audio):6d} samples "
              f"({len(audio)/sr*1000:6.1f} ms audio) at t={timing['elapsed_ms']:8.1f} ms "
              f"frames={timing['frames']}")
    total = time.perf_counter() - t0

    full = np.concatenate(chunks)
    dur = len(full) / sr
    print(f"\nTTFC: {ttfc:.1f} ms")
    print(f"audio: {dur:.2f}s, wall: {total:.2f}s, RTF: {total/dur:.3f}")
    print(f"peak amplitude: {np.abs(full).max():.3f}, rms: {np.sqrt((full**2).mean()):.4f}")
    sf.write("/root/qwen_tts_megakernel/out_megakernel.wav", full, sr)
    print("wrote out_megakernel.wav")

    # HF baseline for the same text (quality reference)
    print("\n=== HF baseline (qwen_tts library) ===")
    t0 = time.perf_counter()
    wavs, bsr = eng.base.generate_custom_voice(text=TEXT, speaker="ryan", language="English")
    bt = time.perf_counter() - t0
    bw = wavs[0] if isinstance(wavs, list) else wavs
    bw = bw.flatten() if hasattr(bw, "flatten") else np.asarray(bw).flatten()
    if torch.is_tensor(bw):
        bw = bw.float().cpu().numpy()
    print(f"baseline: {len(bw)/bsr:.2f}s audio in {bt:.2f}s (RTF {bt/(len(bw)/bsr):.3f})")
    sf.write("/root/qwen_tts_megakernel/out_baseline.wav", bw, bsr)
    print("wrote out_baseline.wav")


if __name__ == "__main__":
    main()
