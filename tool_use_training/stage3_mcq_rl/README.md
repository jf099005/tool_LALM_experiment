# Stage-3：用 verl-omni 直接以 MCQ benchmark 訓練「會用工具答題」的 LALM

目標：讓 Qwen3-Omni（stage-1 學會呼叫音訊工具之後）在 MMAU / MMAR 這類多選題上，**自己決定要不要、
怎麼呼叫音訊工具**（分離人聲、去噪、裁切片段、變速…），聽完工具輸出後再作答，用 RL（GSPO/GRPO）
直接優化答題正確率。

## 流程

```
merge_stage1.sbatch  → stage-1 LoRA 併回 base（verl 從這個 checkpoint 起始，再訓一個新的 LoRA）
build_data.sbatch    → data/{train,val,eval_mmau_test_mini,eval_mmar}.parquet
eval.sbatch          → (選用) 先量 stage-1 init 的 baseline
train.sbatch         → GSPO RL（val_before_train=true，第 0 步就會先跑一次 val）
CKPT=... eval.sbatch → 對 RL checkpoint 跑完整 MMAU test-mini + MMAR
```

全部在 `tool_use_training/stage3_mcq_rl/` 底下 `sbatch <script>` 提交。

## 一個 episode 長什麼樣

```
system   : stage-1 一模一樣的 tool catalogue（已驗證與 stage1_full_toolset/train.jsonl 逐字相同）
user     : 任務說明 + "Input audio: <audio_0><audio>" + 題目 + (A)…(D)
assistant: <tool_call>{"name": "extract_target", "arguments": {...}}</tool_call>   ← policy
tool     : Output of step 1: <audio_1><audio>                                       ← 真的執行，輸出音檔餵回
assistant: <answer>B</answer>                                                       ← policy
```

- `audio_tool_agent_loop.py`：自訂 verl agent loop。verl 內建的 `ToolAgentLoop` 只支援工具回傳**圖片**，
  不支援音訊，所以自己寫。解析用 `interface.protocol.parse_turn`、執行用 `interface.executor.run_tool_call`、
  audio id 由 harness 指派——全部沿用 stage-1 / `interface/` 的協定（見 AGENT_TOOL_DESIGN.md），
  不另刻一份。最多 4 個 assistant turn（≤3 次工具呼叫 + 作答），tool turn 的 token 不進 loss。
- `reward.py`：`1.0*答對 + 0.1*格式正確 + TOOL_BONUS*(答對且有用工具) − 0.05*工具錯誤數`。
  `TOOL_BONUS` 預設 0：只用正確率，工具「有幫助才會被強化」。若 policy 一開始就不呼叫工具導致學不到，
  再用 `STAGE3_TOOL_BONUS=0.1` 之類的值拉一把。log 裡有 `used_tool`、`acc_and_used_tool`、
  `num_tool_errors` 等指標可以觀察工具使用率。
- `mcq_protocol.py`：prompt / 選項 / `<answer>` 格式的唯一定義（資料、rollout、reward 共用）。
- `tests/smoke_agent_loop.py`：CPU-only smoke test（假的 LLM server + 真的跑 clipping 工具），
  驗證 rollout 的 token 序列與 HF chat template 對同一段對話的渲染**逐 token 相同**。

## 資料（`build_mcq_dataset.py`）

| data_source | 來源 | 預設筆數 | 備註 |
|---|---|---|---|
| `compa_r` | AudioMCQ 的 CompA-R 子集，音檔對到本機 `/work/u1501463/CompA_R` mp3 | 8000 | 197k 題全部有本機音檔；預設只取 `audio-contribution=strong`（需要聽音檔才答得出來的題） |
| `dcase2025` | DCASE 2025 task-5 AQA train（`generate/dcase/dcase_7481.json`，路徑重新對到 `/work/u1501463/DCASE_2025_task5`） | 全部 7481（扣掉 >30s） | |
| `audioset_labels` | audioset_20k 只有音檔+標籤，**沒有 QA**，所以自己合成「下列哪個聲音有出現」 | 2000 | 干擾項只取 Animal / Natural sounds / Sounds of things 分支、且與答案不同分支（AudioSet 標籤不完整，人聲/音樂/環境音常常有出現卻沒標） |
| `mmau_test_mini` | MMAU test-mini（已下載到 `/work/u1501463/MMAU`） | 1000（val 取 200 分層抽樣） | **只做 val/eval，不訓練** |
| `mmar` | MMAR | 999 | **只做 eval**；1 題答案不在選項內被剔除 |

- 選項順序每題重新洗牌（固定 seed），避免答案位置偏差。
- eval 集保留完整分母：MMAU 27 題選項重複 → 去重；MMAR 4 題答案文字與選項不完全相同 → 對到最接近的選項（`extra_info.repaired` 有標記）。
- 沒有使用：AudioMCQ 其他子集（AudioCaps/Clotho/MusicCaps/SpeechCraft/TACOS 音檔不在本機）、GAMA（OpenAQA 為開放式問答、非 MCQ）、MMAU-Pro（本機只有音檔、沒有題目檔）。
  網路上沒找到叫 "openMCU" 的資料集；對 audioset_20k 用上面的標籤合成方式補 MCQ。

## 資源與注意事項

- **GPU**：1 node × 4 H100。actor（FSDP2 + param/optimizer offload）與 vLLM-Omni rollout（TP=2 × 2 replica）共置；
  重量級工具（SAM-Audio-large / AudioSR / DeepFilterNet）的常駐 server 釘在 GPU 3（`HEAVY_TOOL_GPU`），
  所以 `GPU_MEM_UTIL=0.5`。`launch_with_tool_pool.py` 負責起 server 並把 `HEAVY_TOOL_POOL_DIR`
  傳給所有 Ray worker；沒有它 agent loop 會直接報錯（不讓每次呼叫都重新載模型）。
- **吞吐量瓶頸**：重量級工具每個 env 一個 server、序列處理。若 rollout 很慢，先看 log 的 `timing/tool_calls`。
- **環境**（2026-10-12 修過，見 memory）：verl-omni env 補裝了 `librosa`、`qwen-omni-utils`、`qwen-vl-utils`；
  `kernels` 降到 0.12.3（原本的 0.16 讓 `transformers` import 直接失敗）；vllm 換成 `0.27.0+cu129` wheel
  （PyPI 預設的 0.27.0 是 CUDA-13 build，GPU 節點 driver 550 跑不動；2026-10-15 換掉）。
  Slurm 會同時 export `ROCR_VISIBLE_DEVICES`，verl worker 遇到會直接報錯，`run_grpo.sh` 會先 unset。
  sam_audio 需要的 `cuda/12.4` module 只給工具 server 用，不會汙染 verl-omni 的程序。
- 訓練輸出：`/work/u1501463/stage3_mcq_rl/{ckpt,val_generations,tensorboard}/<EXP_NAME>`。

## 本機 UQ 用法（RTX 4070 PC，非 cluster）

`build_mcq_dataset.py` 另外支援 AudioMCQ 的 **Clotho / AudioCaps** 子集（`--clotho-n` / `--audiocaps-n`，預設 0 = 關閉，cluster 行為不變），
以及 `--format uq`：輸出 `uq_subset.json`（dcase_subset.json 格式，選項為 `A. ...`），可直接給
`gen_2nd_2_data/build_dataset.py`（套用隨機工具）與 `qwen25_with_tool_chain_evaluation.py`（UQ）使用。
路徑可用 `STAGE3_WORK_DIR` 或各 `--*-dir` 參數覆寫。

```
./run_local_uq.sh                     # 音檔準備 → 50k strong MCQ → 套工具 → UQ（每步都可中斷續跑）
TOTAL_N=10000 CONTRIBUTION=all ./run_local_uq.sh
STEPS="uq" ./run_local_uq.sh          # 只跑其中幾步：audio mcq tools uq
```

音檔來源：Clotho v2.1 = HF `gijs/clotho`；AudioCaps v2.0 = HF `Zhaowc/AudioCaps`（tar.gz 分片，
`prepare_audiomcq_audio.py` 只抽出 AudioMCQ 用到的檔案）。
