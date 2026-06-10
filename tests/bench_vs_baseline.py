"""Head-to-head: stock qwen_tts (normal TTS) vs megakernel engine.

Same model weights, same text, same speaker. The stock path is the official
Qwen3TTSModel.generate_custom_voice (HF generate loop, dynamic cache, sdpa).
"""

import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts_megakernel.engine import MegakernelTTS

TEXTS = [
    "Hello! This is a benchmark comparing the standard implementation with the megakernel.",
    "The quick brown fox jumps over the lazy dog while streaming audio in real time.",
    "Performance numbers should be measured honestly, with warmup and repeated runs.",
]
SPEAKER, LANG = "ryan", "English"


def to_np(w):
    if torch.is_tensor(w):
        return w.flatten().float().cpu().numpy()
    return np.asarray(w).flatten()


def main():
    torch.manual_seed(0)
    eng = MegakernelTTS()

    # ---- warmup both paths ----
    for _ in eng.stream("Warm up.", speaker=SPEAKER, language=LANG, max_new_tokens=30):
        pass
    eng.base.generate_custom_voice(text="Warm up.", speaker=SPEAKER, language=LANG,
                                   max_new_tokens=30)
    torch.cuda.synchronize()

    rows = []
    for i, text in enumerate(TEXTS):
        # ---- stock qwen_tts ----
        torch.manual_seed(i)
        t0 = time.perf_counter()
        wavs, sr_b = eng.base.generate_custom_voice(
            text=text, speaker=SPEAKER, language=LANG, max_new_tokens=512)
        torch.cuda.synchronize()
        wall_b = time.perf_counter() - t0
        audio_b = to_np(wavs[0])
        dur_b = len(audio_b) / sr_b
        frames_b = round(dur_b * 12)

        # ---- megakernel ----
        torch.manual_seed(i)
        t0 = time.perf_counter()
        ttfc = None
        chunks = []
        sr_m = eng.sample_rate
        for audio, sr_m, timing in eng.stream(text, speaker=SPEAKER, language=LANG,
                                              max_new_tokens=512):
            if ttfc is None:
                ttfc = (time.perf_counter() - t0) * 1000
            chunks.append(audio)
        wall_m = time.perf_counter() - t0
        audio_m = np.concatenate(chunks)
        dur_m = len(audio_m) / sr_m
        frames_m = round(dur_m * 12)

        rows.append((i, dur_b, wall_b, frames_b, dur_m, wall_m, frames_m, ttfc))
        print(f"[{i}] stock:      {dur_b:5.2f}s audio | wall {wall_b:6.2f}s | "
              f"RTF {wall_b/dur_b:5.3f} | {frames_b/wall_b:5.1f} frames/s")
        print(f"[{i}] megakernel: {dur_m:5.2f}s audio | wall {wall_m:6.2f}s | "
              f"RTF {wall_m/dur_m:5.3f} | {frames_m/wall_m:5.1f} frames/s | "
              f"TTFC {ttfc:.0f} ms (stock TTFC = full wall {wall_b*1000:.0f} ms, no streaming)")
        if i == 0:
            sf.write("/root/qwen_tts_megakernel/cmp_stock.wav", audio_b, sr_b)
            sf.write("/root/qwen_tts_megakernel/cmp_megakernel.wav", audio_m, sr_m)

    rtf_b = sum(r[2] for r in rows) / sum(r[1] for r in rows)
    rtf_m = sum(r[5] for r in rows) / sum(r[4] for r in rows)
    print("\n=== aggregate ===")
    print(f"stock qwen_tts : RTF {rtf_b:.3f}")
    print(f"megakernel     : RTF {rtf_m:.3f}  -> {rtf_b/rtf_m:.1f}x faster")
    print("wrote cmp_stock.wav / cmp_megakernel.wav for listening comparison")


if __name__ == "__main__":
    main()
