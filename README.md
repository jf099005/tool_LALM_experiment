# tool_LALM_experiment

研究 audio-LLM（LALM）在「音訊被工具處理過」之後的表現與不確定性（UQ）：
同一道音訊選擇題，分別用**原始音訊（original）**與**套用工具後的音訊（tool_applied）**作答，
比較正確率與 UQ 指標（predictive entropy、length-normalized entropy、discrete semantic entropy、P(True)）。

本文件說明**本機（RTX 4070 12 GB、32 GB RAM）**上的完整操作方式。cluster（nano5）上的 Qwen3-Omni
實驗另見 [`QWEN3_UQ_BENCHMARK.md`](QWEN3_UQ_BENCHMARK.md)。

---

## 目錄

1. [環境與路徑](#1-環境與路徑)
2. [從零建立環境](#2-從零建立環境)
3. [資料準備](#3-資料準備)
4. [建立 tool-applied dataset](#4-建立-tool-applied-dataset)
5. [跑 UQ](#5-跑-uq)
6. [500k 分段流程](#6-500k-分段流程)
7. [新 ML 工具的小規模實驗](#7-新-ml-工具的小規模實驗)
8. [依 UQ 值篩選、分析與匯出](#8-依-uq-值篩選分析與匯出)（一次跑完整個流程：[8.4](#84-整個流程一次執行run_full_pipelinesh)）
9. [Qwen3-Omni](#9-qwen3-omni)
10. [程式與資料位置一覽](#10-程式與資料位置一覽)
11. [常見問題](#11-常見問題)

---

## 1. 環境與路徑

所有大型檔案（conda 環境、快取、資料集、model、結果）都放在**資料碟** `/dev/sda4`（NTFS），
不放系統碟（系統碟幾乎已滿）。資料碟由桌面自動掛載，掛載名稱會隨開機順序變動
（`新增磁碟區`、`新增磁碟區3` …），所以一律透過 `local_env.sh` 自動尋找：

```bash
cd /mnt/20F408ADF408876E/tool_LALM_experiment
source local_env.sh          # 每個新的 shell 都要先執行
echo $DATA_ROOT              # 例：/media/jf099005-u/新增磁碟區/LALM_UQ
```

若出現 `data disk not mounted`，先掛載資料碟：

```bash
udisksctl mount -b /dev/sda4
```

`local_env.sh` 設定的主要變數：

| 變數 | 內容 |
|---|---|
| `DATA_ROOT` | 資料碟上的工作根目錄 `.../LALM_UQ` |
| `PY` | 主環境 `lalm_uq` 的 python（vLLM 0.10.1.1） |
| `UQ_MODEL` | `$DATA_ROOT/models/Qwen2.5-Omni-3B` |
| `UQ_RESULT_DIR` | UQ 結果根目錄 `$DATA_ROOT/exp/eval_result` |
| `HF_HOME` / `PIP_CACHE_DIR` / `CONDA_PKGS_DIRS` / `TMPDIR` / `VLLM_CACHE_ROOT` … | 全部指向資料碟 |
| `HF_TOKEN_PATH` | `~/.cache/huggingface/token`（gated model 需要，例如 SAM-Audio） |

> 執行 Python 一律用 `$PY`（或 `$PY -m ...`），不要用 pip 安裝出來的指令捷徑（它們的 shebang
> 可能指向舊的掛載路徑）。

---

## 2. 從零建立環境

### 2.1 主環境 `lalm_uq`（UQ、資料處理）

```bash
source local_env.sh
~/anaconda3/bin/conda create -y -p "$DATA_ROOT/conda_envs/lalm_uq" python=3.11
$PY -m pip install "vllm==0.10.1.1" "transformers==4.55.4" \
    librosa soundfile scipy pyarrow pandas tqdm matplotlib "huggingface_hub[cli]"
```

- transformers 必須 < 5（5.x 與 vLLM 0.10.1 不相容）。

### 2.2 UQ 用的 model

```bash
$PY -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen2.5-Omni-3B', local_dir='$DATA_ROOT/models/Qwen2.5-Omni-3B')"
```

Qwen2.5-Omni-7B（bf16 約 18 GB）放不進 12 GB 顯示卡，本機一律用 3B。

### 2.3 ML 工具的環境

每個 ML 工具跑在自己的 conda 環境，由 `tools/heavy_tool_server.py` 以常駐 server 呼叫，
不會 import 進主環境：

```bash
tools/setup_local_heavy_envs.sh                       # 全部建立
tools/setup_local_heavy_envs.sh sam_audio extra_tools # 只建立指定的
```

| 環境 | 工具（tool 名稱） | 備註 |
|---|---|---|
| `deepfilternet` | DeepFilterNet（`human_voice_enhance`） | torch 2.0.1 |
| `audiosr` | AudioSR（`super_resolution`） | torch 2.0.1；500k 起預設停用 |
| `sam_audio` | SAM-Audio large（`extract_target`、`remove_target`） | **需要 HF 登入**：`~/anaconda3/bin/huggingface-cli login`（帳號需取得 `facebook/sam-audio-large` 存取權） |
| `extra_tools` | Demucs（`music_source_separation`）、ClearerVoice MossFormer2（`speech_enhance`）、Silero VAD（`vad_keep_speech`） | 不需授權 |
| `extra_tools2` | VoiceFixer（`speech_restore_voicefixer`）、WPE（`dereverb_wpe`） | 不需授權；VoiceFixer 權重會導向資料碟 |

輕量工具（`clipping`、`denoise`、`pitch_shift`、`time_stretch`、`trim_silence`、
`amplitude_normalize`、`loudness_normalize`、`remove_dc_offset`、`spectral_normalize`、
`pre_emphasis`）直接在主環境中執行。

---

## 3. 資料準備

### 3.1 AudioMCQ（Clotho + AudioCaps）—— 主要題庫

| 資料 | 來源 | 存放位置 |
|---|---|---|
| 題目 jsonl | HF `inclusionAI/AudioMCQ`（`AudioMCQ_Clotho.jsonl`、`AudioMCQ_AudioCaps.jsonl`） | `$DATA_ROOT/datasets/AudioMCQ/datasets/` |
| Clotho v2.1 音檔 | HF `gijs/clotho`（parquet） | `$DATA_ROOT/datasets/Clotho/` |
| AudioCaps v2.0 音檔 | HF `Zhaowc/AudioCaps`（tar.gz 分片） | `$DATA_ROOT/datasets/AudioCaps/` |

下載完成後，把音檔解成 AudioMCQ 使用的檔名：

```bash
cd tool_use_training/stage3_mcq_rl
D=$DATA_ROOT/datasets
$PY prepare_audiomcq_audio.py --audiomcq-dir "$D/AudioMCQ/datasets" \
    --clotho-parquet-dir "$D/Clotho/parquet/data" --clotho-out "$D/Clotho/audio" \
    --audiocaps-tar-dir "$D/AudioCaps/raw" --audiocaps-out "$D/AudioCaps/audio"
```

建立題目池（UQ 格式的 `uq_subset.json`）：

```bash
$PY build_mcq_dataset.py --format uq --output-dir "$D/audiomcq_all_94k" \
    --compa-r-n 0 --dcase-n 0 --audioset-n 0 --clotho-n -1 --audiocaps-n -1 \
    --audiomcq-contribution all --audiomcq-dir "$D/AudioMCQ/datasets" \
    --clotho-audio-dir "$D/Clotho/audio" --audiocaps-audio-dir "$D/AudioCaps/audio"
```

- `--audiomcq-contribution strong|weak|all`：AudioMCQ 的 audio-contribution 標記
  （strong = 必須聽音訊才能作答）。目前的題目池（94,310 題）用 `all`；先前的 50k 實驗只用 `strong`。
- `--total-n N` 可限制題數（例如 `--total-n 50000`）。

### 3.2 DCASE 2025 AudioQA（較早的實驗）

`tool_use_training/gen_2nd_2_data/run_local.sh` 會從 HF `gijs/dcase2025-audio-qa` 的 parquet
解出音檔並建立 `$DCASE_QA_JSON`。

---

## 4. 建立 tool-applied dataset

核心程式：`tool_use_training/gen_2nd_2_data/build_dataset.py`。對每一題套用工具，輸出：

```
<output-dir>/original_subset.json        原始音訊的題目清單
<output-dir>/tool_applied_subset.json    套用工具後的題目清單（同 id、同順序）
<output-dir>/manifest.json               每題用了哪個工具、參數、輸出路徑
<output-dir>/tool_applied/<id>/tool_applied.wav
```

### 4.1 工具設定檔（`tool_use_training/gen_2nd_2_data/`）

| 檔案 | 內容 |
|---|---|
| `tool_config.json` | cluster 原始設定（只有輕量工具） |
| `tool_config_local_ml.json` | 輕量工具 + 4 個 ML 工具，`heavy_tool_probability: 0.4`（ML 工具共佔 40%） |
| `tool_config_local_ml_nosr.json` | 同上，但停用 `super_resolution`（500k 使用） |
| `tool_config_extra_ml.json` | 只有 5 個新 ML 工具（第 7 節的實驗使用） |

每個工具的參數範圍寫在設定檔的 `params` 裡（`choice` / `uniform` / `randint` / `fixed`）。

### 4.2 執行

```bash
$PY tool_use_training/gen_2nd_2_data/build_dataset.py \
    --tool-config tool_use_training/gen_2nd_2_data/tool_config_local_ml.json \
    --input-json "$DATA_ROOT/datasets/audiomcq_all_94k/uq_subset.json" \
    --num-samples 1000 --min-tools 1 --max-tools 1 --seed 0 \
    --workers 12 --output-dir "$DATA_ROOT/datasets/my_set"
```

常用選項：

| 選項 | 作用 |
|---|---|
| `--workers N` | 多行程（建議 12）。ML 工具會**分階段**執行：先跑輕量工具，再依序每個 ML 環境各起一個常駐 server，避免多個 model 同時佔用 GPU |
| `--no-copy-original` | original 直接指向原始資料集的音檔，不複製（省一半空間） |
| `--only-ids ids.json` | 只重建指定 id（同一個指令加上這個選項與新的 `--output-dir`，參數會與原本相同） |
| `--max-attempts-per-sample N` | 失敗重試次數 |
| `--resume / --no-resume` | 預設可中斷續跑 |

題目若帶有 `force_tools: ["工具名"]` 欄位，就固定用該工具（參數仍隨機抽），
`plan_tool_variants.py` 就是用這個方式讓同一題的不同版本使用不同工具。

---

## 5. 跑 UQ

主程式：`qwen25_with_tool_chain_evaluation.py`（Qwen2.5-Omni，vLLM）。對每題輸出：
greedy 答案與正確與否、10 次 T=1.0 抽樣、各種 entropy、P(True)。

### 5.1 單一題目清單

```bash
$PY qwen25_with_tool_chain_evaluation.py \
    --subset_path my_set/tool_applied_subset.json --output_path results/my_uq.json \
    --model "$UQ_MODEL" --compute_uncertainty --uq_num_samples 10 --uq_sample_temperature 1.0 \
    --quantization fp8 --batch_size 32 --max_num_seqs 96 --max_num_batched_tokens 4096 \
    --gpu_memory_utilization 0.82 --max_model_len 4096 --max_audios_per_prompt 1
```

- 輸出檔名會自動加上 `_no_tool_chain_results.json`。
- 執行中的進度寫在 `<輸出檔>.partial.jsonl`，**隨時可中斷**；重跑同一指令會跳過已完成的題目（`--no-cont` 從頭開始）。
- `--paired_subset_path B.json --paired_output_path B_out.json`：同時跑兩份清單（例如 original 與 tool_applied），兩邊交錯處理、進度一致。

本機的建議設定（RTX 4070 12 GB）：

| 設定 | 速度 | 說明 |
|---|---|---|
| bf16：`--batch_size 16 --max_num_seqs 32 --gpu_memory_utilization 0.90` | 約 2.0 題/秒 | 50k 實驗使用 |
| **FP8：`--quantization fp8 --batch_size 32 --max_num_seqs 96 --max_num_batched_tokens 4096 --gpu_memory_utilization 0.82`** | 約 3.2 題/秒 | 500k 與新工具實驗使用。與 bf16 相比選到同一選項 97.5%，P(True) 平均略高約 0.013 → **兩種設定的結果不要混用** |

- `--gpu_memory_utilization` 不要設太高：桌面程式佔用 0.6–1 GB 顯存時，0.86 以上會在 sampler 暖機時 OOM。
- `--kv_cache_dtype fp8` 會讓 vLLM 0.10.1 退回 V0 引擎而 OOM，不要使用。

### 5.2 成對執行（original + tool_applied）

```bash
SUBSET_DIR=$DATA_ROOT/datasets/my_set RUN_TAG=my_set ./run_UQ_for_stage2_local.sh
```

可用環境變數覆寫：`MODEL`、`BATCH_SIZE`、`GPU_MEM_UTIL`、`MAX_MODEL_LEN`、`PAIRED=0`（改為依序跑）、`FRESH=1`（從頭開始）。

### 5.3 長時間執行：用 systemd 服務（建議）

```bash
./run_uq_service.sh start            # 啟動（OOM / 崩潰後 60 秒自動重啟並續跑）
./run_uq_service.sh status           # 狀態、重啟次數、進度
./run_uq_service.sh log              # 即時 log
./run_uq_service.sh stop             # 停止（不會自動重啟）
STEPS="tools uq" ./run_uq_service.sh start   # 連同建立 dataset 一起跑（stage3_mcq_rl/run_local_uq.sh）
```

以 systemd 服務執行的好處：記憶體不足時只會終止這個服務並自動重啟，不會把終端機或其他程式一起關掉。

---

## 6. 500k 分段流程

`run_500k_pipeline.sh`：用題目池的 94,310 題，每題重複套用 5–6 種**不同**工具
（排除該題在 50k 實驗已用過的工具，停用 `super_resolution`），共 500,000 筆，分成 10 個 slice：

1. **規劃**（`plan_tool_variants.py`）→ `$DATA_ROOT/datasets/audiomcq500k/plan/slice_XX/input.json`
2. **original 的 UQ**：每題只算一次，所有版本共用
3. 每個 slice：**套用工具** → **tool_applied 的 UQ** → **刪除**（`prune_slice.py`：P(True) 低於 original 的音檔刪除，紀錄保留）

```bash
./run_500k_pipeline.sh service   # 啟動（systemd 服務 lalm-500k，自動重啟續跑）
./run_500k_pipeline.sh status    # 每個 slice 的進度、保留/刪除數、錯誤數、剩餘空間
./run_500k_pipeline.sh stop
```

持續查看 log：

```bash
watch -n 600 ./run_500k_pipeline.sh status
tail -f "$DATA_ROOT/logs/lalm-500k.log" | tr '\r' '\n' | grep --line-buffered -E "^==|/[0-9]+ \[|Error|Traceback"
journalctl --user -u lalm-500k --since today      # 是否被 OOM 終止等
```

輸出：

| 內容 | 位置 |
|---|---|
| 每個 slice 的 manifest、題目清單、保留的音檔 | `$DATA_ROOT/datasets/audiomcq500k/slice_XX/` |
| 每筆的工具、參數、兩邊 P(True)、保留/刪除 | `slice_XX/prune_log.jsonl`、`slice_XX/prune_summary.json` |
| original 的 UQ | `$UQ_RESULT_DIR/audiomcq500k/Qwen2.5-Omni-3B-fp8_pool_original_uq_results_no_tool_chain_results.json` |
| 各 slice tool_applied 的 UQ | `$UQ_RESULT_DIR/audiomcq500k/Qwen2.5-Omni-3B-fp8_slice_XX_tool_applied_uq_results_no_tool_chain_results.json` |

重建已刪除的音檔：從 `prune_log.jsonl` 取出 `action == "deleted"` 的 id 存成 json 清單，
用該 slice 原本的 `build_dataset.py` 指令（見 `run_500k_pipeline.sh`）加上
`--only-ids ids.json --output-dir <新目錄>`。輕量工具可完全重現；SAM-Audio 本身有隨機性，結果不會完全相同。

每個 slice 約 11.3 小時（套用工具約 7 小時，UQ 約 4.3 小時）。

---

## 7. 新 ML 工具的小規模實驗

`run_extra_ml_experiment.sh`：從題目池隨機抽 `N_Q` 題，每題套用 `tool_config_extra_ml.json`
裡的每個工具，跑 UQ 後依工具分析（original 的 UQ 沿用 500k 的 FP8 結果）。

```bash
./run_extra_ml_experiment.sh service                         # 預設：SET=extra_ml_2000，N_Q=2000，SEED=8
SET=extra_ml_500 N_Q=500 SEED=3 ./run_extra_ml_experiment.sh service
journalctl --user -u lalm-extra-ml-extra-ml-2000             # 服務名稱為 lalm-extra-ml-<SET>
```

結果：`$UQ_RESULT_DIR/<SET>/summary.json`（同時印在 log 最後）。

新增工具的方法：在 `tools/extra_ml_tools.py` 實作一個 `Tool` 子類別 →
在 `tools/_tool_table.py` 與 `tool_use_training/gen_2nd_2_data/tool_appliers.py`
（`APPLIERS`、`_HEAVY_TOOLS_REQUIRING_OUTPUT_PATH`）註冊 → 在設定檔加入工具與其 `env`。

---

## 8. 依 UQ 值篩選、分析與匯出

### 8.1 `uq_threshold_confusion.py`：依門檻篩選並畫 confusion 圖

適用於 original 與 tool_applied **以相同 id 成對**的結果檔（例如 50k 實驗）。

```bash
$PY uq_threshold_confusion.py --threshold 0.8                                    # 預設：P(True) 套用工具後上升，且上升後 >= 0.8
$PY uq_threshold_confusion.py --on delta --metric p_true --threshold 0.1         # ΔP(True) = tool − original >= 0.1
$PY uq_threshold_confusion.py --on delta --metric discrete_semantic_entropy --threshold 1.0
$PY uq_threshold_confusion.py --on delta --metric p_true --threshold 0.1 --below # 改為 <=
```

| 選項 | 說明 |
|---|---|
| `--metric` | `p_true`（預設）、`predictive_entropy`、`length_normalized_entropy`、`discrete_semantic_entropy`、`semantic_entropy` |
| `--on` | `increase`（預設：上升且 tool 值 ≥ 門檻）、`delta`、`abs_delta`、`original`、`tool_applied`、`both`、`either` |
| `--original / --tool-applied` | 指定結果檔（預設為 50k 實驗）；可直接讀執行中的 `.partial.jsonl` |
| `--save-items x.json` | 輸出篩出的題目（工具、兩邊指標、對錯、回答） |
| `--export-dir DIR` | 把篩出的題目匯出成 dataset（兩份 subset、manifest、`filter.json`，預設連音檔一起複製） |

圖檔預設存在 `$UQ_RESULT_DIR/uq_confusion/`。

### 8.2 `analyze_tool_uq.py`：依工具統計（500k、新工具實驗用）

500k 與新工具實驗的 original 結果是共用的（以 `base_id` 配對），用這支分析：

```bash
$PY analyze_tool_uq.py \
    --tool-uq "$UQ_RESULT_DIR/extra_ml_2000/Qwen2.5-Omni-3B-fp8_tool_applied_uq_results_no_tool_chain_results.json" \
    --manifest "$DATA_ROOT/datasets/extra_ml_2000/build/manifest.json" \
    --original-uq "$UQ_RESULT_DIR/audiomcq500k/Qwen2.5-Omni-3B-fp8_pool_original_uq_results_no_tool_chain_results.json" \
    --baseline-slices "$DATA_ROOT/datasets/audiomcq500k" \
    --baseline-uq-glob "$UQ_RESULT_DIR/audiomcq500k/Qwen2.5-Omni-3B-fp8_slice_*_tool_applied_uq_results_no_tool_chain_results.json" \
    --out summary.json
```

每個工具輸出：筆數、正確率變化、T→F / F→T、平均 ΔP(True)、P(True) 上升比例、ΔP > 0.1 / 0.2 的筆數、Δ discrete semantic entropy。

500k 的逐筆 ΔP(True) 也可以直接從 `slice_*/prune_log.jsonl` 讀取（`p_true_original`、`p_true_tool_applied`）。

### 8.3 匯出並上傳到 cluster

```bash
./export_and_push.sh                          # ΔP(True) >= 0.1
./export_and_push.sh p_true 0.2               # <指標> <門檻>
ON=increase ./export_and_push.sh p_true 0.8   # 其他篩選規則
```

會先匯出到 `$DATA_ROOT/exports/<指標>_<規則>_ge<門檻>/`，再用 rsync 上傳到
`/work/u1501463/LALM_UQ/exports/`（需輸入 cluster 密碼）。cluster 路徑設定在 `cluster_paths.sh`；
單獨上傳可用 `./push_exports_to_nano5.sh <set 名稱>`（`DRY_RUN=1` 只列出不上傳）。

**500k 版本**（original 為共用的 pool 結果，以 manifest 的 `base_id` 配對；10 個 slice 合併成一組）：

```bash
./export_and_push_500k.sh                 # ΔP(True) >= 0.2 → exports/audiomcq500k_p_true_delta_ge0.2，再上傳
./export_and_push_500k.sh p_true 0.1      # <指標> <門檻>；已匯出過就直接上傳（REEXPORT=1 重新匯出）
$PY export_500k_uq_filtered.py --threshold 0.2   # 只匯出
```

匯出格式與 50k 相同（同一題的多個工具版本各自成為一筆，id 為 `<base_id>__v<k>`），
cluster 上用 `SET=audiomcq500k_p_true_delta_ge0.2 ./run_qwen3_uq_benchmark.sh` 執行。

### 8.4 整個流程一次執行：`run_full_pipeline.sh`

把「建立 dataset + Qwen2.5 UQ → 依門檻匯出 → 上傳 cluster」串成一支：

| `PIPELINE` | build 步驟 | export 步驟 | 匯出的 set 名稱 |
|---|---|---|---|
| `500k`（預設） | `run_500k_pipeline.sh run`（第 6 節） | `export_500k_uq_filtered.py` | `audiomcq500k_<指標>_<規則>_ge<門檻>` |
| `50k` | `stage3_mcq_rl/run_local_uq.sh`（第 3–5 節） | `uq_threshold_confusion.py --export-dir` | `<指標>_<規則>_ge<門檻>` |

```bash
./run_full_pipeline.sh service      # build + export 以 systemd 服務執行（lalm-full-<PIPELINE>，崩潰/OOM 自動重啟續跑）
./run_full_pipeline.sh status       # 各步驟是否完成、服務狀態、build 進度
./run_full_pipeline.sh push         # 服務完成後，自己執行上傳（需輸入 cluster 密碼；DRY_RUN=1 只列出）
./run_full_pipeline.sh run          # 或：在終端機前景跑完 build → export → push
./run_full_pipeline.sh stop

PIPELINE=50k THRESHOLD=0.1 ./run_full_pipeline.sh run       # 換 pipeline / 門檻
METRIC=discrete_semantic_entropy THRESHOLD=1.0 ./run_full_pipeline.sh service
STEPS="export push" ./run_full_pipeline.sh run              # 只跑部分步驟（build / export / push）
```

- 每一步都可以中斷後重跑：build 依各自的進度標記跳過已完成部分；export 在 `filter.json` 比所有 UQ 結果檔新時跳過
  （`REEXPORT=1` 強制重新匯出）；rsync 只傳缺少或變更的檔案。
- 上傳需要密碼，所以 `service` 模式不含 push；沒有終端機時 push 會跳過並印出要自己執行的指令。
- 若 `lalm-500k` / `lalm-uq` 服務正在執行，build 步驟會拒絕啟動（避免兩邊同時寫同一份資料）。
- 上傳後在 cluster：`SET=<set 名稱> ./run_qwen3_uq_benchmark.sh`。

---

## 9. Qwen3-Omni

`qwen3_uq_benchmark.py`：對匯出的題目組，分別用 original（不用 tool）與 tool_applied（用 tool）跑
Qwen3-Omni 的 UQ，輸出與 Qwen2.5 相同格式的結果檔與 `<model>_summary.json`
（正確率提升、confusion、各 UQ 指標的平均變化、各工具統計）。

- **cluster（正式數字）**：照 [`QWEN3_UQ_BENCHMARK.md`](QWEN3_UQ_BENCHMARK.md)，用 `run_qwen3_uq_benchmark.sh`（sbatch 交給 cluster 規範處理）。
- **本機（4-bit，僅供參考）**：vLLM 無法在 12 GB + 32 GB RAM 上載入；改用 llama.cpp（Q4_K_M + MoE 專家層放 CPU），約 5.7 秒/次：

```bash
B=$DATA_ROOT/llama.cpp/build/bin; M=$DATA_ROOT/models/Qwen3-Omni-30B-A3B-Instruct-GGUF
LD_LIBRARY_PATH=$B $B/llama-server -m $M/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf \
    --mmproj $M/mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf \
    -ngl 999 --n-cpu-moe 34 -c 16384 --parallel 4 -fa on --host 127.0.0.1 --port 8080 &
$PY qwen3_uq_benchmark.py --backend llamacpp --server_url http://127.0.0.1:8080 --concurrency 4 \
    --subset_dir "$DATA_ROOT/exports/p_true_delta_ge0.1" --output_dir "$DATA_ROOT/exp/qwen3_llamacpp" \
    --model "$M/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M"
```

4-bit 結果與 cluster 的 bf16 結果不可混用。

---

## 10. 程式與資料位置一覽

**程式**（repo 根目錄 `/mnt/20F408ADF408876E/tool_LALM_experiment/`）

| 檔案 | 用途 |
|---|---|
| `local_env.sh` | 路徑設定（每次先 `source`） |
| `qwen25_with_tool_chain_evaluation.py` | UQ 主程式（Qwen2.5-Omni） |
| `uncertainty_quantification_tools/` | UQ 指標計算 |
| `run_UQ_for_stage2_local.sh`、`run_uq_service.sh` | 成對 UQ、systemd 服務包裝 |
| `run_500k_pipeline.sh` | 500k 分段流程 |
| `run_extra_ml_experiment.sh` | 新 ML 工具實驗 |
| `uq_threshold_confusion.py`、`analyze_tool_uq.py` | 篩選 / 分析 |
| `run_full_pipeline.sh` | 整個流程一次執行（build → export → push，第 8.4 節） |
| `export_and_push.sh`、`push_exports_to_nano5.sh`、`cluster_paths.sh` | 匯出與上傳（50k） |
| `export_500k_uq_filtered.py`、`export_and_push_500k.sh` | 匯出與上傳（500k） |
| `qwen3_uq_benchmark.py`、`run_qwen3_uq_benchmark.sh`、`QWEN3_UQ_BENCHMARK.md` | Qwen3-Omni |
| `tools/` | 所有音訊工具（`extra_ml_tools.py` 為新 ML 工具；`heavy_tool_pool.py` / `heavy_tool_server.py` 為常駐 server） |
| `tools/setup_local_heavy_envs.sh` | ML 工具環境安裝 |
| `tool_use_training/gen_2nd_2_data/` | `build_dataset.py`、`tool_appliers.py`、工具設定檔、`plan_tool_variants.py`、`prune_slice.py` |
| `tool_use_training/stage3_mcq_rl/` | AudioMCQ 題目建構（`build_mcq_dataset.py`、`prepare_audiomcq_audio.py`、`run_local_uq.sh`） |

**資料**（`$DATA_ROOT` 底下）

| 目錄 | 內容 |
|---|---|
| `conda_envs/` | 所有 conda 環境 |
| `models/` | Qwen2.5-Omni-3B、Qwen3-Omni（AWQ / GGUF）、extra_tools 權重 |
| `hf_cache/` | HuggingFace 快取（SAM-Audio、AudioSR 等） |
| `datasets/` | 原始資料、題目池、各實驗的 dataset |
| `exp/eval_result/` | 所有 UQ 結果與分析 |
| `exports/` | 匯出的題目組 |
| `logs/` | 各服務的 log |

---

## 11. 常見問題

| 狀況 | 處理 |
|---|---|
| `data disk not mounted` | `udisksctl mount -b /dev/sda4`，再用對應的 `... service` 指令續跑 |
| vLLM 啟動時 OOM（sampler warm-up） | 調低 `--gpu_memory_utilization`（0.82 以下）或 `--max_num_seqs` |
| 服務被 systemd-oomd 終止 | 看 `journalctl --user -u <服務名>`；服務會自動重啟續跑。長時間任務期間避免開大型程式（瀏覽器分頁過多等） |
| 修改了正在執行的 `.sh` | **不要這樣做**：bash 是邊讀邊執行，修改執行中的腳本會讓它在下一步出錯。要改請先 `stop`，改完再 `service` |
| `.partial.jsonl` 是什麼 | 執行中的進度，完成後才合併成正式的 json；中斷後重跑會自動接續 |
| SAM-Audio 輸出每次不同 | 正常：它從隨機噪聲開始生成 |
| 需要 gated model（SAM-Audio） | `~/anaconda3/bin/huggingface-cli login`，token 存在 `~/.cache/huggingface/token` |
