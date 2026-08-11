# TODO -- 待同步/待處理事項（2026-08-06)

這份清單記錄「LLM 不再自己指定 output audio id（改由 agent/harness 指派)」+
「stage-1 資料生成加入可選 CoT」這次改動之後，**還沒**跟著同步的地方。

已改完、可以當作範例參照的檔案：`tools/abstract_tool.py`、`interface/protocol.py`、
`interface/agent.py`、`interface/executor.py`、
`tool_use_training/gen_1st_stage_data/{build_dataset.py,cot_templates.py}`，
設計細節見 [AGENT_TOOL_DESIGN.md](AGENT_TOOL_DESIGN.md) §2.1/§3/§6/§9。

## 1. `output_audio_id` 改由 harness 指派 -- 尚未同步的消費端

- [ ] **`testing_tool_use_benchmark/run_eval.py`**
  - `DEFAULT_PROTOCOL_SYSTEM_PROMPT`（約 L70-80)還是舊文字，要求 zero-shot 模型
    自己宣告 `output_audio_id`（"Every audio-producing tool also takes an
    `output_audio_id` argument: a fresh id you choose..."）。這跟現在
    `tools_registry.describe_available_tools()` 產生的 schema（已經不再列出
    `output_audio_id`)互相矛盾——zero-shot 模型會同時看到「schema 沒有這個參數」
    跟「system prompt 說你要給一個」。改成跟 `interface/protocol.py::build_system_prompt`
    的新措辭一致（agent 執行後才指派、下一輪才展示給你看)。
  - 逐輪執行迴圈裡 `output_audio_id = parameters.get("output_audio_id") or
    f"audio_{step_idx + 1}"`（約 L212)還是「model 給了就用 model 的」邏輯。目前
    functionally 沒壞（新 schema 下 model 不會給,所以永遠落到 fallback,數字規則
    剛好跟 `interface/agent.py` 目前的指派規則一致),但邏輯上該比照
    `interface/agent.py::run()` 改成**永遠**由這裡指派、不讀 `parameters`，
    避免將來 `interface/agent.py` 的指派規則改掉之後這裡沒跟著改，兩邊產生分歧。
  - 同段附近（約 L222-233)的註解也提到「`agent.py`: primary output under
    `output_audio_id`」,措辭要跟著更新。

- [ ] **`tool_use_training/tool_use_RL_{full,partial}/build_rl_dataset.py`、
  `reward.py`**：檢查過，這兩個檔案的 GRPO 單發格式本來就只要求
  `{"tool_calls": [{"tool_name":..., "parameters":...}]}`，從來沒有要求
  `output_audio_id`，`reward.py::_score_single_call` 也只比對 `tool_name`/
  `parameters`，不受 `answer.tool_calls[i]` 新增的 `output_audio_id`/`cot`
  兩個 top-level key 影響。**目前不需要改**，但如果之後要把 stage-2 也改成
  逐輪 `<tool_call>` 格式（而不是單發 JSON),要記得重新確認這件事還成不成立。

- [ ] **`testing_tool_use_benchmark/build_benchmark.py`**：直接呼叫
  `build_dataset.build_one_sample`，會自動拿到新的 `cot`/不變的
  `output_audio_id`（bookkeeping 用)欄位，**不需要改**就能跑。如果之後想讓
  benchmark 也能選擇性帶 CoT（例如想測「有 CoT 提示時模型表現如何」),需要仿照
  `build_dataset.py` 的 `--use-cot` 加一個等價開關（目前 `build_benchmark.py`
  沒有 `to_swift_sample` 那條路徑,只有 raw dataset JSON)。

- [ ] **既有的、用舊協定生成的資料檔要重新生成才能拿去訓練/評估**（不重新生成的話,
  訓練資料裡的 assistant turn 仍然會教模型產生 `output_audio_id`,跟現在的
  schema/system prompt 矛盾)：
  - `tool_use_training/gen_1st_stage_data/tool_usage_qa.json`
  - `tool_use_training/gen_1st_stage_data/exp/tool_usage_qa_audioset.json`、
    `exp/train.jsonl`
  - `testing_tool_use_benchmark/benchmark.json`
  - （根目錄的 `merged.json`/`merge.py` 是另一條 Dcase/UQ prediction 合併的
    pipeline，跟 stage-1 tool-use 資料生成無關，這裡不用管)
  - 任何用這些舊檔案訓練出來的 checkpoint,拿去跟新協定的 `interface/agent.py`
    或新版 `run_eval.py` 對接時，都要注意訓練/推論協定不一致的問題。

## 2. CoT（`cot_templates.py`) -- 尚未涵蓋的部分

- [ ] 目前只涵蓋 `tools_registry.py` 現在真的 `@register` 起來的 6 個工具
  （`clipping`、`denoise`、`pitch_shift`、`pre_emphasis`、`time_stretch`、
  `trim_silence`)。`tools_registry.py` 裡被註解掉、之後如果重新啟用的工具
  （`amplitude_normalize`、`loudness_normalize`、`remove_dc_offset`、
  `spectral_normalize`、`human_voice_enhance`、`super_resolution`、
  `extract_target`、`remove_target` 等)目前只會落到 `cot_templates.py` 的
  generic fallback 句型（"applying {tool_name} with these parameters..."),
  沒有專屬模板——重新啟用時記得補幾句。
- [ ] `--use-cot` 目前預設 `False`（保留舊行為)。如果之後決定要預設開啟、或是
  想要「部分開啟」（例如按機率決定每個 sample 要不要有 CoT,而不是整批
  一起開/關),要在 `build_dataset.py`/`cot_templates.py` 加對應邏輯。
  現在的粒度是**整個 --swift-output-file 開/關**，不是逐 sample。
- [ ] `testing_tool_use_benchmark/run_eval.py`／RL 的 `build_rl_dataset.py`
  完全沒有意識到 `cot` 這個新欄位——如果之後想讓 eval 或 RL 資料也能帶 CoT
  提示，需要仿照 `to_swift_sample(..., use_cot=...)` 各自加一份。

## 3. 一般性提醒

- 修改上述任何檔案時，維持 [AGENT_TOOL_DESIGN.md](AGENT_TOOL_DESIGN.md) 記錄的分層
  規則（`tools/` 定義工具、`tool_call_formats.py` 管 wire format、`interface/`
  管即時執行、`tools_registry.py` 是「目前實際採用的工具集」唯一定義),不要在
  單一消費端另外發明一份 allowlist 或協定文字。
- 改完任何一個檔案的職責分工，記得回來更新 `AGENT_TOOL_DESIGN.md`（它自己在
  §8 裡也是這樣要求的),並視情況更新/清空這份 TODO。
