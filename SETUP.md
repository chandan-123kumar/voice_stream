# Reproducing the environment on a fresh RTX 5090 box

1. Base: any CUDA 12.8+ box with an RTX 5090. Stock PyPI `torch==2.10.0`
   (cu128 wheel) supports sm_120 and is pinned in requirements.txt — the
   NVIDIA container's custom torch build is no longer required. (History:
   the container build worked too, but pip's resolver treats its prerelease
   version `2.10.0a0+nv26.1` as unsatisfying and silently replaced it with
   the stock wheel on 2026-06-10; the stock wheel was then verified to run
   the megakernel at full speed.)
2. `pip install -r requirements.txt && bash scripts/fix_torchaudio_stub.sh`
   (the fix script is mandatory: qwen-tts itself depends on torchaudio, so
   every install of it pulls in the broken stock wheel — see step 3)
3. torchaudio gotcha: if the container's torch is a custom NVIDIA build, stock
   torchaudio wheels fail at import with
   `OSError: libtorchaudio.abi3.so: undefined symbol: torch_dtype_float4_e2m1fn_x2`
   (symbol name may vary by version). qwen_tts only imports
   `torchaudio.compliance.kaldi` (unused by the 12Hz path), so a stub suffices.
   Fix with one command:

       bash scripts/fix_torchaudio_stub.sh

   **This can recur**: any `pip install` that pulls torchaudio back in as a
   dependency overwrites the stub and the app stops starting with the error
   above. Just re-run the script. (Happened once on 2026-06-10 — app.py
   crashed on startup after a stray torchaudio 2.11.0 install.)
4. torchvision gotcha: if a torchvision built against a different CUDA than
   torch is present, app.py dies at startup with
   `ModuleNotFoundError: Could not import module 'AutoProcessor'`
   (the real error underneath: "PyTorch and torchvision were compiled with
   different CUDA major versions" — transformers auto-imports torchvision
   whenever it's installed). Nothing in this project uses torchvision; the
   same `bash scripts/fix_torchaudio_stub.sh` detects and removes a broken
   one. (Happened on 2026-06-10 when pip replaced the container's torch but
   left the container's CUDA-13.1 torchvision behind.)
5. Model downloads automatically on first run:
   Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (public, no HF token needed).
6. Kernel JIT-compiles on first import (-arch=sm_120a), ~1-2 min.
7. Smoke test: `python3 synthesize.py "Hello." --out hello.wav`
