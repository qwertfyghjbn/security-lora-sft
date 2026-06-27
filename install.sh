#!/usr/bin/env bash
# Full environment setup for security_lora_sft
# GPU  : RTX 4060 (CUDA 12.7 driver, cu124 wheels)
# Usage: bash install.sh

set -euo pipefail

PYTHON=/usr/bin/python3.10
VENV=.venv

echo "==> Creating virtual environment at $VENV"
uv venv "$VENV" --python "$PYTHON"

UVPIP="uv pip install --python $VENV/bin/python"

# ── 1. PyTorch with CUDA 12.4 wheels (compatible with CUDA 12.7 driver) ──────
echo "==> Installing PyTorch (cu124) — this may take a few minutes"
$UVPIP torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

# ── 2. Unsloth (auto-detects torch/CUDA after step 1) ────────────────────────
echo "==> Installing Unsloth"
$UVPIP unsloth

# ── 3. Remaining dependencies ─────────────────────────────────────────────────
# torchao is intentionally NOT installed:
#   - torchao 0.9.0: compatible with torch 2.6.0 but PEFT 0.19+ raises on version check
#   - torchao >=0.16.0: requires torch>=2.7.0 (not available for cu124)
#   Both transformers and PEFT safely return False when torchao is absent.
echo "==> Installing project dependencies"
$UVPIP -r requirements.txt

# ── 4. Smoke test ─────────────────────────────────────────────────────────────
echo "==> Smoke test"
$VENV/bin/python - <<'EOF'
import torch
from unsloth import FastLanguageModel  # noqa: F401
print(f"  torch      : {torch.__version__}")
print(f"  CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF

echo ""
echo "Done. Activate with:  source $VENV/bin/activate"
