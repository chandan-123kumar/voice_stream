#!/usr/bin/env bash
# Fix the torchaudio / torchvision ABI-mismatch crashes.
#
# Symptom 1 (on `python3 app.py` / `synthesize.py` / anything importing qwen_tts):
#   OSError: .../torchaudio/lib/libtorchaudio.abi3.so:
#   undefined symbol: torch_dtype_float4_e2m1fn_x2
#
# Symptom 2 (seen 2026-06-10 after pip replaced the container's torch):
#   ModuleNotFoundError: Could not import module 'AutoProcessor'.
#   Underneath it: RuntimeError: Detected that PyTorch and torchvision were
#   compiled with different CUDA major versions.
#   transformers auto-imports torchvision if installed; a CUDA-mismatched
#   torchvision therefore poisons AutoProcessor. Nothing in this project uses
#   torchvision, so the fix is simply to remove it.
#
# Cause (both): pip mixing wheels built against different torch/CUDA versions.
# pip may silently (re)install torchaudio/torchvision as a dependency of
# something else (qwen-tts itself depends on torchaudio). qwen_tts only needs
# `import torchaudio.compliance.kaldi` to succeed (the kaldi path is used by
# the 25Hz tokenizer only — the 12Hz models we use never call it), so a
# 3-file stub is sufficient.
#
# Usage: bash scripts/fix_torchaudio_stub.sh
set -euo pipefail

SITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
TA="$SITE/torchaudio"

# Remove torchvision if it is installed but broken (CUDA mismatch with torch).
if ! python3 -c "import torchvision" 2>/dev/null; then
    if pip show torchvision > /dev/null 2>&1; then
        echo "removing torchvision (installed but fails to import — CUDA/ABI mismatch)"
        pip uninstall -y torchvision 2>/dev/null || true
    fi
fi

pip uninstall -y torchaudio 2>/dev/null || true
rm -rf "$TA"
mkdir -p "$TA/compliance"

cat > "$TA/__init__.py" << 'EOF'
"""Stub torchaudio — the real wheel is ABI-incompatible with this container's torch build.
qwen_tts only needs torchaudio.compliance.kaldi for the 25Hz tokenizer path, which the
12Hz models don't use. This stub satisfies the import without the broken C extension."""
__version__ = "0.0.0-stub"
from . import compliance  # noqa: F401
EOF

cat > "$TA/compliance/__init__.py" << 'EOF'
from . import kaldi  # noqa: F401
EOF

cat > "$TA/compliance/kaldi.py" << 'EOF'
"""Stub of torchaudio.compliance.kaldi — raises if actually called."""


def fbank(*args, **kwargs):
    raise NotImplementedError(
        "torchaudio is stubbed in this environment (ABI mismatch with the "
        "container's torch build). kaldi.fbank is only needed for the 25Hz "
        "tokenizer's x-vector extraction, which this project does not use."
    )
EOF

python3 -c "import torchaudio; print('torchaudio stub OK:', torchaudio.__version__)"
