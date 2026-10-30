#!/usr/bin/env bash
# Idempotent local patches for the `verl-omni` conda env (Python 3.10) used by stage3_mcq_rl.
# Run once after creating/reinstalling the env (CPU only, no GPU needed):
#     conda activate verl-omni && bash env_patches/apply_env_patches.sh
# Background and alternatives: ../ENV_NOTES.md
set -euo pipefail

SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
PYVER=$(python -c "import sys; print('%d.%d' % sys.version_info[:2])")
echo "python ${PYVER}, site-packages: ${SITE}"

# --- Patch 1: flashinfer comm/fd_exchange.py uses `array.array[int]` in an annotation -------
# Evaluated at import time; only valid on Python >= 3.12. vLLM imports flashinfer.comm
# unconditionally when building its TP communicator, so every rollout worker dies with
# "TypeError: 'type' object is not subscriptable" (smoke job 390038).
FD="${SITE}/flashinfer/comm/fd_exchange.py"
if [ ! -f "$FD" ]; then
    echo "[patch1] $FD not found (flashinfer layout changed?) -- check ENV_NOTES.md" >&2
    exit 1
elif python - "$FD" <<'EOF'
import sys
src = open(sys.argv[1], encoding="utf-8").read()
sys.exit(0 if "from __future__ import annotations" in src else 1)
EOF
then
    echo "[patch1] already applied: $FD"
elif python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo "[patch1] not needed on Python ${PYVER}"
else
    python - "$FD" <<'EOF'
import sys
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
marker = "\nimport array\n"
assert marker in src, "unexpected fd_exchange.py layout; patch by hand (see ENV_NOTES.md)"
patch = ("\n# Local patch (tool_use_LALM stage3_mcq_rl/ENV_NOTES.md): postponed annotations so\n"
         "# `array.array[int]` works on Python < 3.12.\nfrom __future__ import annotations\n")
open(path, "w", encoding="utf-8").write(src.replace(marker, patch + marker, 1))
print(f"[patch1] applied: {path}")
EOF
fi

# --- Verify (import only; touches no GPU) -------------------------------------------------
python -c "import flashinfer.comm; print('[verify] flashinfer.comm imports OK')"
