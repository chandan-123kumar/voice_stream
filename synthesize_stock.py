#!/usr/bin/env python3
"""Generate speech with the STOCK qwen_tts implementation only.

No megakernel, no CUDA-kernel compile — just the official Qwen3TTSModel
generate loop. Useful as a quality/performance reference.

Usage:
    python3 synthesize_stock.py "Text to speak" [--speaker ryan] [--language English] [--out out_stock.wav]
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--speaker", default="ryan",
                   help="serena vivian uncle_fu ryan aiden ono_anna sohee eric dylan")
    p.add_argument("--language", default="English")
    p.add_argument("--out", default="out_stock.wav")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text=args.text, speaker=args.speaker, language=args.language)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    audio = wavs[0]
    if torch.is_tensor(audio):
        audio = audio.flatten().float().cpu().numpy()
    else:
        audio = np.asarray(audio).flatten()
    dur = len(audio) / sr
    sf.write(args.out, audio, sr)
    print(f"{dur:.2f}s audio in {wall:.2f}s (RTF {wall/dur:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
