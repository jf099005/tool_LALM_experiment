# PC 待辦：匯出 10k 題給 cluster 跑原始精度（bf16）Qwen3-Omni UQ

給在本機（RTX 4070 PC）上執行的 agent。照順序做，每一步完成後檢查「完成條件」；
任何一步失敗且原因不明顯時，**停下來回報**，不要猜。

## 背景

- 目前 Qwen3-Omni 的 UQ 有兩份結果：PC 上的量化版（llama.cpp IQ2_M，`run_qwen3_llamacpp_local.sh`
  → `qwen3_uq_benchmark.py --backend llamacpp`）和 cluster 上的 bf16（同一支程式，`--backend vllm`）。
- 之前上傳的 `audiomcq50k_lowp_ptrue_lt0.8_dp_ge0.2` 是依**量化版**的 P(True) 篩出來的；cluster 上
  bf16 的結果顯示這個篩選有明顯偏差（量化版看到的「工具有幫助」在 bf16 上大多不成立）。
- 所以下一步：從 50k 資料集**隨機抽 10k 題、不經任何篩選**，在 cluster 上用 bf16 跑，
  並（若可行）附上量化版在同一批題目上的結果，直接比較兩者。
- cluster 端已準備好：`qwen3_bf16_uq.sbatch`（1 張 H100，bf16）。cluster 上沒有 AudioMCQ 的音檔，
  所以音檔必須由 PC 匯出並上傳。

## 0. 準備

```bash
cd /mnt/20F408ADF408876E/tool_LALM_experiment      # repo
git fetch origin && git checkout local_PC_test && git pull
source local_env.sh                                  # 每個新 shell 都要；設定 $DATA_ROOT、$PY
```

- 若 `data disk not mounted`：`udisksctl mount -b /dev/sda4`，再 `source local_env.sh`。
- **不要**在 `lalm-500k`、`lalm-uq`、`qwen3-run` 等 systemd 服務執行中修改它們用到的 `.sh`。
  先看 `systemctl --user list-units 'lalm-*' 'qwen3-*'`，有服務在跑就回報，不要停掉它。

完成條件：`git log -1` 包含 `PC_TODO_bf16_10k.md` 這次的 commit；`echo $DATA_ROOT` 有值。

## 1. 確認來源資料集

```bash
D=$DATA_ROOT/datasets/audiomcq_strong_50k_tool_applied_ml40
ls "$D"/original_subset.json "$D"/tool_applied_subset.json "$D"/manifest.json
df -h "$DATA_ROOT"                                   # 需要約 10–20 GB 空間（20k 個 wav）
```

- 若目錄名稱不同，在 `$DATA_ROOT/datasets/` 找「50k、strong、tool_applied、ml40」的那個
  （它是 `exports/p_true_delta_ge0.1/filter.json` 的 `source_dataset_dir`），並在回報中寫明。

完成條件：三個檔案都在；兩份 subset 各 50000 題、id 順序相同（腳本會自己檢查）。

## 2. 匯出隨機 10k（不篩選）

```bash
$PY export_random_subset.py --subset-dir "$D" --n 10000 --seed 0 \
    --export-dir "$DATA_ROOT/exports/audiomcq50k_random10k_s0"
```

- 只用 CPU；可中斷，重跑同一指令會跳過已複製的音檔。**seed 不要改**（cluster 端預設的 set 名稱
  `audiomcq50k_random10k_s0` 對應 seed 0）。
- 腳本是在 cluster 上寫的、還沒實際執行過。若出錯，修正腳本本身（保持參數與輸出格式不變），
  修正後 commit 並 push 到 `local_PC_test`，在回報中說明改了什麼。

完成條件：`$DATA_ROOT/exports/audiomcq50k_random10k_s0/` 有 `original_subset.json`、
`tool_applied_subset.json`（各 10000 題）、`manifest.json`、`filter.json`、`original/`、`tool_applied/`。

## 3.（建議）量化版在同一批 10k 上的結果

目的：讓 cluster 能直接比較 bf16 與量化版（同題、同工具）。

1. 先找有沒有現成結果：量化版先前的結果在 `$DATA_ROOT/exp/qwen3_llamacpp/` 底下（例如 `lowp_rounds`）。
   只有在某個目錄是 `qwen3_uq_benchmark.py` 的輸出格式（`*_original_uq_results_no_tool_chain_results.json`
   與 `*_tool_applied_...json`），**而且**涵蓋這 10k 題的兩邊（original 與 round-1 的 tool_applied，
   id 與 50k 資料集相同）時才能直接用。部分涵蓋也可以，腳本只會放入兩邊都有的題目。
2. 沒有現成結果，就在匯出的 set 上直接跑量化版（GPU，約 0.5 秒/邊 → 20k 次約 3 小時）：
   ```bash
   SUBSET_DIR=$DATA_ROOT/exports/audiomcq50k_random10k_s0 \
   OUTPUT_DIR=$DATA_ROOT/exp/qwen3_llamacpp/audiomcq50k_random10k_s0_IQ2_M \
       ./run_qwen3_llamacpp_local.sh service
   ./run_qwen3_llamacpp_local.sh status      # 單次查看；不要用 watch / 迴圈輪詢
   ```
   - 預設 `QUANT=IQ2_M`、`SAMPLES=0`（只算 greedy + P(True)），與先前的量化版設定相同，不要改。
   - 跑之前確認 GPU 沒有被其他服務佔用（見第 0 步）。
3. 有了結果後，**用同樣的參數**再跑一次第 2 步，加上 `--ref-dir`（音檔已複製過，會直接跳過）：
   ```bash
   $PY export_random_subset.py --subset-dir "$D" --n 10000 --seed 0 \
       --export-dir "$DATA_ROOT/exports/audiomcq50k_random10k_s0" \
       --ref-dir "$DATA_ROOT/exp/qwen3_llamacpp/<結果目錄>"
   ```

完成條件：export 目錄多了 `reference/`（兩個結果檔）；`filter.json` 有 `items`、`model`、
`n_reference_items`。這一步做不到時跳過即可（cluster 照樣能跑 bf16），在回報中說明原因。

## 4. 上傳到 cluster（只能由使用者本人執行）

連線 cluster 需要密碼與驗證碼（OTP），連 `DRY_RUN=1` 也會連線、要求登入。所以 agent：

- **不要執行** `push_exports_to_nano5.sh`（包括 `DRY_RUN=1`）、`rsync`、`scp`、`ssh` 或任何會連到
  cluster（`nano5.nchc.org.tw`）的指令；不要嘗試自動輸入密碼、設定金鑰或改用其他登入方式。
- 只在本機做上傳前檢查，然後停下，把下面的上傳指令交給使用者：
  ```bash
  E=$DATA_ROOT/exports/audiomcq50k_random10k_s0
  du -sh "$E"                                        # 匯出大小
  find "$E/original" "$E/tool_applied" -type f | wc -l   # 應為 20000
  ```
- 使用者自己執行（會要求密碼與驗證碼）：
  ```bash
  ./push_exports_to_nano5.sh "$DATA_ROOT/exports/audiomcq50k_random10k_s0"
  ```

完成條件：使用者確認 rsync 完成（agent 不需要等，回報後即可結束）。

## 5. 回報給使用者

- 來源資料集路徑、匯出的題數、export 目錄大小。
- 第 3 步採用哪個方式（現成結果 / 新跑 / 跳過），reference 涵蓋幾題。
- 若修改過腳本：改了什麼、commit id。
- 接下來 cluster 上要做的事（只列在回報裡給使用者；agent 不要連線 cluster、不要執行）：
  ```bash
  cd /home/u1501463/tool_use_LALM_pc && git pull
  SMOKE=1 sbatch --time=00:20:00 qwen3_bf16_uq.sbatch    # 先 32 題，確認 1 張 GPU 放得下
  sbatch qwen3_bf16_uq.sbatch                             # 全部 10k
  ```
