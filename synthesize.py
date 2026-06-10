#!/usr/bin/env python3
"""Generate speech with the megakernel-backed Qwen3-TTS engine.

Usage:
    python3 synthesize.py "Text to speak" [--speaker ryan] [--language English] [--out out.wav]
"""

import argparse
import os
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_tts_megakernel.engine import MegakernelTTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--speaker", default="ryan",
                   help="serena vivian uncle_fu ryan aiden ono_anna sohee eric dylan")
    p.add_argument("--language", default="English")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    eng = MegakernelTTS()
    for _ in eng.stream("Warm up.", speaker=args.speaker, language=args.language,
                        max_new_tokens=30):
        pass

    t0 = time.perf_counter()
    audio, sr, timing = eng.synthesize(args.text, speaker=args.speaker,
                                       language=args.language)
    wall = time.perf_counter() - t0
    dur = len(audio) / sr
    sf.write(args.out, audio, sr)
    print(f"{dur:.2f}s audio in {wall:.2f}s (RTF {wall/dur:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
