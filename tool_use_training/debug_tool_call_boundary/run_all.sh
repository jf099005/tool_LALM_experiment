#!/usr/bin/env bash
# Runs the three isolation probes (baseline, no-audio, no-adapter, fp32) each
# in a fresh process -- a fresh process per variant matters here since dtype
# and adapter-loading aren't cleanly reversible on an already-loaded model --
# then prints a side-by-side summary. Must run under the `ms-swift` conda env.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p out

echo "=== [1/4] baseline: adapter + audio + bf16 ==="
python probe_first_token.py --out out/baseline.json

echo "=== [2/4] no-audio: adapter + text-only + bf16 ==="
python probe_first_token.py --no-audio --out out/no_audio.json

echo "=== [3/4] no-adapter: base checkpoint + audio + bf16 ==="
python probe_first_token.py --no-adapter --out out/no_adapter.json

echo "=== [4/4] fp32: adapter + audio + fp32 ==="
python probe_first_token.py --dtype fp32 --out out/fp32.json

python - <<'EOF'
import json
from pathlib import Path

variants = ["baseline", "no_audio", "no_adapter", "fp32"]
print("\n=== summary ===")
print(f"{'variant':<12} {'first_id':>8} {'has_ufffd':>10} {'tc_rank':>8} {'tc_logit':>10} {'spread':>8}")
for name in variants:
    path = Path("out") / f"{name}.json"
    if not path.exists():
        continue
    r = json.loads(path.read_text())
    fl = r.get("first_token_logits", {})
    tc = fl.get("tool_call_token", {})
    first_id = r["token_ids_head"][0] if r["token_ids_head"] else None
    print(
        f"{name:<12} {first_id!s:>8} {str(r['has_replacement_char']):>10} "
        f"{tc.get('rank', '?'):>8} {tc.get('logit', float('nan')):>10.4f} "
        f"{fl.get('spread_top_k', float('nan')):>8.4f}"
    )
EOF
