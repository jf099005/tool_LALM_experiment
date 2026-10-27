#!/bin/bash
# 用 scp 把跑 tool-use benchmark 所需的程式碼、benchmark.json，以及
# benchmark.json 內引用到的 audio 檔案，複製到目標工作站。
#
# 用法 (三個參數都可省略，會套用預設值):
#   ./deploy_to_remote.sh [-H user@host] [-c remote_code_dir] [-a remote_audio_dir]
#
#   -H  目標工作站 (ssh 格式 user@host，需已設定好金鑰登入)
#       預設: u1501463@nano5.nchc.org.tw
#   -c  遠端「程式碼 + benchmark.json」要放的目錄 (repo root)。
#       interface/、official_system_prompt.py、testing_tool_use_benchmark/
#       會依原本的相對結構放在這底下 -- run_eval.py 是用這個相對結構
#       (REPO_ROOT = 自己所在目錄的上一層) 找 interface.* 跟
#       official_system_prompt，路徑對不上就會 import 失敗。
#       預設: 跟本地一樣的路徑 (repo root)
#   -a  遠端 audio 檔案要放的目錄，跟 -c 分開、可自由指定。
#       腳本會自動產生一份改寫過路徑的 benchmark.json (把寫死的
#       /work/.../tool_use_benchmark_audio 換成這裡指定的目錄) 一併複製過去，
#       所以遠端 audio 目錄不需要跟本地路徑相同。
#       預設: /work/u1501463/ 底下跟本地同名的目錄 (即跟本地路徑相同)
#
# 不包含模型權重 (model_path / adapter_path，通常在共用的 /work 儲存空間上，
# 已經在目標工作站上存在)；如果目標工作站看不到那個路徑，要另外處理。
#
# 密碼只會被問一次：整支腳本共用同一條 ssh ControlMaster 連線 (下面的
# SSH_OPTS)，之後所有 ssh/scp 都掛在這條已認證的連線上；同時把 audio
# 目錄底下上百個 bench_id 子目錄合併成一次 scp 呼叫，而不是每個目錄各跑一次
# scp（會各自觸發一次密碼提示）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_AUDIO_ROOT="/work/u1501463/tool_use_benchmark_audio"

REMOTE_HOST="u1501463@nano5.nchc.org.tw"
REMOTE_CODE_DIR="$REPO_ROOT"
REMOTE_AUDIO_DIR="/work/u1501463/$(basename "$LOCAL_AUDIO_ROOT")"

usage() {
    echo "Usage: $0 [-H user@host] [-c remote_code_dir] [-a remote_audio_dir]" >&2
    exit 1
}

while getopts "H:c:a:h" opt; do
    case "$opt" in
        H) REMOTE_HOST="$OPTARG" ;;
        c) REMOTE_CODE_DIR="$OPTARG" ;;
        a) REMOTE_AUDIO_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "遠端主機:        $REMOTE_HOST"
echo "遠端程式碼目錄:   $REMOTE_CODE_DIR"
echo "遠端 audio 目錄:  $REMOTE_AUDIO_DIR"

# -- 建一條共用的 ssh 連線，之後的 ssh/scp 都重用它，全程只需認證一次 --
CONTROL_PATH="/tmp/.deploy_ssh_ctrl_$$_%r@%h:%p"
SSH_OPTS=(-o "ControlMaster=auto" -o "ControlPath=$CONTROL_PATH" -o "ControlPersist=10m")
cleanup() {
    ssh -O exit "${SSH_OPTS[@]}" "$REMOTE_HOST" >/dev/null 2>&1 || true
    rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo "== 建立共用連線 (只會問一次密碼) =="
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" true

echo "== 1/3 建立遠端目錄 =="
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
    "mkdir -p '$REMOTE_CODE_DIR/interface' '$REMOTE_CODE_DIR/testing_tool_use_benchmark' '$REMOTE_AUDIO_DIR'"

echo "== 2/3 複製程式碼 (json 跟 code 放同一個目錄) =="

# 先在本地把要送到 testing_tool_use_benchmark/ 底下的檔案 (含改寫過
# audio 路徑的 benchmark.json) 集中放到一個暫存目錄，這樣一次 scp
# 就能把該目錄要用到的所有檔案送完，不用每個檔案各跑一次。
STAGE_DIR="$(mktemp -d)"
python3 - "$SCRIPT_DIR/benchmark.json" "$LOCAL_AUDIO_ROOT" "$REMOTE_AUDIO_DIR" "$STAGE_DIR/benchmark.json" <<'PYEOF'
import json
import sys

src_path, old_root, new_root, out_path = sys.argv[1:5]
with open(src_path) as f:
    data = json.load(f)
for item in data:
    for k in ("source_audio", "target_audio"):
        if k in item and item[k].startswith(old_root):
            item[k] = new_root + item[k][len(old_root):]
with open(out_path, "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PYEOF
cp "$SCRIPT_DIR/run_eval.py" "$SCRIPT_DIR/run_eval_batch.py" "$SCRIPT_DIR/audio_metrics.py" "$SCRIPT_DIR/run.sbatch" "$STAGE_DIR/"

scp "${SSH_OPTS[@]}" "$REPO_ROOT/official_system_prompt.py" \
    "$REMOTE_HOST:$REMOTE_CODE_DIR/"
scp "${SSH_OPTS[@]}" "$REPO_ROOT"/interface/*.py \
    "$REMOTE_HOST:$REMOTE_CODE_DIR/interface/"
scp "${SSH_OPTS[@]}" "$STAGE_DIR"/* \
    "$REMOTE_HOST:$REMOTE_CODE_DIR/testing_tool_use_benchmark/"

echo "== 3/3 複製 audio 檔案 (一次 scp 送出所有 bench_id 目錄) =="
mapfile -t BENCH_DIRS < <(python3 -c "
import json
d = json.load(open('$SCRIPT_DIR/benchmark.json'))
print('\n'.join('$LOCAL_AUDIO_ROOT/' + i for i in sorted({item['id'] for item in d})))
")
scp -r "${SSH_OPTS[@]}" "${BENCH_DIRS[@]}" "$REMOTE_HOST:$REMOTE_AUDIO_DIR/"

echo "完成。遠端結構："
echo "  $REMOTE_CODE_DIR/official_system_prompt.py"
echo "  $REMOTE_CODE_DIR/interface/*.py"
echo "  $REMOTE_CODE_DIR/testing_tool_use_benchmark/{run_eval.py,run_eval_batch.py,audio_metrics.py,benchmark.json,run.sbatch}"
echo "  $REMOTE_AUDIO_DIR/<bench_id>/*.wav"
