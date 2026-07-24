# Agent / Tool 設計說明與注意事項

> 這份文件記錄 `tool_use_LALM` 專案中「工具集定義」「system prompt / tool-calling wire
> format」「real-time agent 執行」三層架構的設計邏輯與彼此關係，以及本次重構過程中
> 建立的設計原則。**開新 task 前請先讀這份文件**，避免重新發明或重複實作已存在的機制。

## 1. 整體分層

```
tools/                      <- 工具「是什麼」：實作 + 完整清單 + 專案採用的子集
tool_call_formats.py         <- tool-calling 的 wire format（<tool_call> 等）怎麼寫/怎麼解析
official_system_prompt.py    <- system prompt 的「persona」部分（讀 model 自己的 chat_template）
interface/                   <- agent「怎麼跑」：組 system prompt、跑多輪對話、真正執行 tool call
tool_use_training/, testing_tool_use_benchmark/
                              <- 上述三層的消費者：產生訓練資料 / RL 資料 / 跑 eval
```

依賴方向永遠是單向的：`interface/` 和 `tool_use_training/`、`testing_tool_use_benchmark/`
都讀 `tools/` 與 `tool_call_formats.py`，反過來不成立。

---

## 2. `tools/` — 工具集的唯一定義

- **`tools/abstract_tool.py`**：`Tool` abstract base class。每個工具實作
  `name()` / `description()` / `parameter_schema()`（執行期用的 JSON schema，
  裡面是 `audio_path`，因為工具本身只認得真實檔案路徑）、`execute()`。
  - `produces_audio()`：預設 `True`；只有 `asr` 覆寫成 `False`（輸出是文字，不是新音檔）。
  - `requires_output_path()`：這個工具是否需要呼叫端額外傳一個 `output_path`
    （denoise、normalize 家族、pitch/time、voice enhance、super-resolution 是；
    clipping、source separation、extract/remove target、asr 不是，輸出路徑自動推導）。
  - **`to_function_schema()`**：把 `parameter_schema()` 轉成「model 看到的」function-calling
    schema——關鍵轉換：拿掉 `audio_path`、加上 `audio_id`（optional，語意見下方）跟
    `output_audio_id`（僅當 `produces_audio()` 為真時，required）。**也會把
    `description()` 裡殘留的 `audio_path` 字樣自動替換成 `audio_id`**，因為
    `description()` 這個字串同時被路徑導向的消費者（`prompts/`、`audio_edit/editor.py`）
    使用，不能直接改寫，只能在轉出 schema 的當下做替換。

- **`tools/_tool_table.py`**：`{tool_name: (module_path, class_name)}` 的唯一對照表。
  每個 tool module 自己用 `try/except ImportError` 包住重量級依賴（torch、
  DeepFilterNet、AudioSR、sam_audio），所以這張表在任何 env 下都能被完整 import——
  它是「整個專案曾經實作過的工具」的完整清單，**不代表哪些工具現在真的在用**。

- **`tools/__init__.py`**：從上表 eager import 出 `TOOL_NAME_TO_CLASS` / `TOOL_CLASSES`
  （完整清單），加上 `generate_tool_descriptions()`（給 `prompts/`/`audio_edit/editor.py`
  那條 DCASE/MCQ 路徑用的散文式目錄）與 `tool_function_schemas()`（結構化 schema 版本）。

- **`tools/tools_registry.py`**（原名 `synthetic_registry.py`，已改名）：
  **整個專案「目前實際採用」的工具集的唯一定義**。`available_tool_names()` /
  `available_tool_schemas()` / `describe_available_tools(tool_call_format=...)`
  是這件事的權威答案——`interface/`、`tool_use_training/gen_1st_stage_data/build_dataset.py`、
  `testing_tool_use_benchmark/build_benchmark.py` 都從這裡讀,不各自維護一份清單。
  這樣才不會出現「agent 會 advertise/呼叫一個模型從沒訓練過的工具」的漂移
  （這正是本次重構修掉的實際 bug：改之前 `interface` 的預設工具集有 14 個，
  `build_dataset.py` 訓練資料只涵蓋 6 個)。
  - 除了「有哪些工具在用」這個共用清單以外，這個檔案也保留了它原本的角色：
    每個 `@register(...)` 裝飾的 `_apply_*` function 是**合成訓練資料專用**的
    隨機參數生成邏輯（`REGISTRY[name].apply(audio_path, output_path, rng, duration)`），
    這部分只有 `build_dataset.py`/`build_benchmark.py` 會用到,**不是** live agent
    執行的路徑。

### 2.1 `audio_path` vs `audio_id` —— 兩層位址空間，不要混用

- `parameter_schema()`（執行期 schema）：永遠是 `audio_path`，工具實作只認得真實檔案路徑。
- `to_function_schema()`（model 看到的 schema）：永遠是 `audio_id`（+ `output_audio_id`）。
  `audio_id` **是 optional**——沒給時預設是「上一輪自己產生的音檔」（或第一輪唯一的
  input audio），這是這個專案自訂的 chaining 慣例，不是任何模型的官方慣例。
- 兩者的轉換點只有一個：`interface/executor.py::run_tool_call()`——把 `audio_id`/
  `output_audio_id` 從 `parameters` 裡拿掉、換上真正解析出的 `audio_path`,再呼叫
  `cls.execute()`。**新工具或新程式碼不要自己重新實作這個轉換**，一律呼叫
  `run_tool_call()`。

---

## 3. `tool_call_formats.py` — tool-calling 的 wire format 是可插拔的

- `ToolCallFormat` abstract class,四個方法：`render_tools_preamble(schemas)`、
  `render_tool_call(name, arguments)`、`render_done()`、`parse_turn(text)`。
- **`QwenToolCallFormat`**（`name="qwen"`,預設值）：Qwen2.5/Qwen3 官方 Hermes 風格,
  `<tools>...</tools>` + `<tool_call>{"name":...,"arguments":...}</tool_call>`。
  已對照本機的 `chat_template.json`（Qwen2.5-Omni / Qwen3-Omni 兩份快取）與 ms-swift
  內建的 `swift.agent_template.hermes` 逐字核對過,格式完全一致。
- **`LegacyJSONToolCallFormat`**（`name="legacy"`）：專案原本手刻的扁平 JSON
  （`{"tool_name":..., "parameters":..., "output_audio_id":...}` + `{"done": true}`）,
  保留下來是為了「以後要換一個不是 Qwen 慣例的模型」時,不用改任何呼叫端,只要加一個
  新的 `ToolCallFormat` 子類別、傳 `--tool-call-format <新名字>` 即可。
- **`output_audio_id` 一律裝進 `arguments`/`parameters` 裡**（不是獨立欄位),因為
  function-calling schema 本來就只有一個 `arguments` 物件可以放東西。
  `LegacyJSONToolCallFormat` 會在 render 時把它拆回獨立欄位、parse 時再塞回去,
  對上層調用者維持統一的 `{"tool_name":..., "parameters": {...含 output_audio_id...}}`
  介面。
- `render_tool_result_message()`（在 `interface/protocol.py`,不在這個檔案裡)
  **刻意不**把 tool 執行結果包進 `<tool_response>` 標籤——見下一節的 ms-swift 細節。

### 3.1 重要細節：`<tool_response>` 包裝在哪裡做,不能重複包

- ms-swift 的預設行為（`template_backend='swift'`,這是 default,不是 `'jinja'`)
  對 Qwen 系列模型的 `agent_template` 預設是 `'hermes'`（見
  `swift/template/templates/qwen.py::QwenTemplateMeta.agent_template = 'hermes'`,
  Qwen2.5-Omni/Qwen3-Omni 的 template 都沒有覆寫這個值)。這代表：**訓練資料裡
  `role: "tool"` 的 message content,ms-swift 在 tokenize 時會自動幫你包成
  `<tool_response>...</tool_response>`**——不管你有沒有給 `tools` 欄位。
  - 所以 `build_dataset.py`/`interface/protocol.py::render_tool_result_message()`
    的 tool turn 內容一律是**未包裝的純文字**（例如 `"Output of step 1: <audio_2><audio>"`)。
    如果自己又手動包一次 `<tool_response>`,ms-swift 會再包一層,變成雙重包裝——**不要這樣做**。
  - `interface/engine.py::SwiftEngine`（inference 走 ms-swift 自己的 Template
    機制)跟 training 走的是同一套邏輯,所以行為一致,不用擔心 train/infer 不一致。
  - **唯一的例外**是 `interface/engine.py::VLLMEngine`——它是手刻的 ChatML renderer,
    底下完全沒有 ms-swift 那層自動包裝,所以它的 `_render_prompt()`
    自己把連續的 `tool` turn 合併進一個 `<|im_start|>user` block、每則包
    `<tool_response>`,模仿 Qwen 官方 chat_template 真正的行為（這是唯一需要手動包裝
    的地方,因為它是唯一沒有任何 templating 層的路徑,直接對著 raw/未微調的官方
    checkpoint 講話)。

- Qwen2.5-Omni 的**官方** `chat_template.json`（HF 上發布的那份)其實完全沒有
  tool-calling 支援（沒有 `{% if tools %}` 分支)；Qwen3-Omni 的才有完整的
  `<tools>`/`<tool_call>`/`<tool_response>` 邏輯。ms-swift 能對兩者都套用
  Hermes 慣例,是因為它用自己的 `template_backend='swift'` 機制(不是直接呼叫
  model 自帶的 `apply_chat_template`),额外幫 Qwen2.5-Omni 補上了官方模板沒有的
  tool-calling 能力。這也是為什麼 `official_system_prompt.py` 讀 `chat_template.json`
  只讀得到「persona 那句話」（`"You are a helpful assistant."`),讀不到 tool-calling
  格式——tool-calling 格式是 `tool_call_formats.py` 自己刻的,不依賴 model 自帶模板。

---

## 4. `official_system_prompt.py` — persona 部分

- `load_official_system_prompt(model_dir, model_type)`：從 `chat_template.json` /
  `tokenizer_config.json` 裡,用 regex 抓 model 自己預設會塞的那句 system 文字
  （例如 Qwen2.5-Omni 是 `"You are a helpful assistant."`)。抓不到就 fallback 到
  `KNOWN_DEFAULT_SYSTEM_PROMPTS[model_type]`,再抓不到就用通用 fallback。
  - **注意**：Qwen3(-Omni) 的 chat_template 沒有硬編碼任何預設 persona
    （沒給 system message 時,模板什麼都不會自動補),所以這個 regex 對 Qwen3
    抓不到東西,會落到 fallback——這個 fallback 值**不代表** Qwen3 的官方行為,
    只是巧合抓到同一句話。幫 Qwen3 系列訓練時,記得檢查
    `KNOWN_DEFAULT_SYSTEM_PROMPTS` 有沒有對應的 key,沒有的話這個 fallback
    是不準的。
- `compose_system_prompt(tools_block, base_system_prompt=None, ...)`：單純字串接合
  `f"{base}\n\n{tools_block}"`。**這個函式對 tool-calling wire format 一無所知**——
  `tools_block` 必須是呼叫端（`tools_registry.describe_available_tools()`)已經用
  某個 `ToolCallFormat` 排版好的完整字串,不要在這裡加任何額外的 label
  （之前踩過一次雷：這裡曾經多接了一句 `"Available tools:\n"`,跟
  `describe_available_tools()` 自己的 heading 重複/衝突,已修掉)。

---

## 5. `interface/` — agent 怎麼跑

- **`interface/protocol.py`**：組「即時 agent」用的 system prompt
  （`build_system_prompt()`——protocol 說明文字 + `tool_call_formats` 的 tools preamble),
  以及把單輪對話渲染/解析的四個函式都包成薄 wrapper,轉呼叫 `tool_call_formats.py`：
  `render_tool_call()` / `render_done()` / `parse_turn()` / `render_tool_result_message()`
  （最後一個不經過 format,見上面 §3.1)。
  `audio_to_audio_tool_names()` 現在是 `tools_registry.available_tool_names()`
  再過濾 `produces_audio()`——**不要**改回直接讀 `tools.TOOL_NAME_TO_CLASS`（那是
  完整清單,不是專案採用的清單)。
- **`interface/executor.py`**：**唯一**「給定 `tool_name` + `parameters`（含
  `audio_id`/`output_audio_id`)+ 真實 `input_audio_path`,實際執行一個工具」的實作
  ——`run_tool_call()`。會先檢查 `tool_name` 在不在
  `tools_registry.available_tool_names()`（不只是「這個 class 存不存在」),再把
  `audio_id`/`output_audio_id`/`output_path` 從 `parameters` 拿掉、換上真正的路徑,
  呼叫 `validate_parameters()` + `execute()`。**任何要「執行一個 model 預測出來的
  tool call」的新程式碼都應該呼叫這個函式,不要重新刻一份。**
  已確認的呼叫端：`interface/agent.py`、`testing_tool_use_benchmark/run_eval.py`、
  `tool_use_training/tool_use_RL_full/reward.py`（`ToolAudioClosenessReward`)。
- **`interface/agent.py`**：`ToolCallingAgent.run()`——多輪迴圈本體。
  `engine.generate_turn(messages, audios)` → `protocol.parse_turn()` →
  （是 tool call 就)`executor.run_tool_call()` →
  `protocol.render_tool_result_message()` 塞進下一輪,直到 `{"done": True}`
  或 `max_steps`。
- **`interface/engine.py`**：`SwiftEngine`（ms-swift `TransformersEngine`,吃
  LoRA/官方 checkpoint 皆可)與 `VLLMEngine`（純 vLLM,只能跑未微調的官方 checkpoint,
  手刻 ChatML,見 §3.1)。兩者對外都是同一個介面：
  `generate_turn(messages, audios) -> str`。

---

## 6. 訓練資料 / RL 資料 / Eval 的消費關係

| 檔案 | 角色 | 用到的共用元件 |
|---|---|---|
| `tool_use_training/gen_1st_stage_data/build_dataset.py` | 產生 stage-1 SFT 多輪對話資料（合成 A→B 音檔鏈) | `tools_registry`（工具清單 + 隨機參數合成)、`official_system_prompt.compose_system_prompt`、`interface.protocol.render_tool_call/render_done/render_tool_result_message` |
| `testing_tool_use_benchmark/build_benchmark.py` | 產生 held-out eval benchmark（跟 build_dataset.py 同一套合成邏輯,不同 seed) | 直接呼叫 `build_dataset.build_one_sample` + `tools_registry` |
| `testing_tool_use_benchmark/run_eval.py` | 跑一個 checkpoint,對 benchmark 逐輪生成、真的執行、算 metric | `official_system_prompt`、`interface.protocol.parse_turn/render_tool_result_message`、`interface.executor.run_tool_call`、`interface.engine.{SwiftEngine,VLLMEngine}` |
| `tool_use_training/tool_use_RL_{full,partial}/build_rl_dataset.py` | 產生 stage-2 GRPO 單輪資料——**不是**逐輪 `<tool_call>` 格式,而是要求模型一次輸出整條 `{"tool_calls":[...]}` | `official_system_prompt.compose_system_prompt`（system prompt 共用),但**不用** `tool_call_formats.py`——這是刻意的,見下方 §7 |
| `tool_use_training/tool_use_RL_{full,partial}/reward.py` | GRPO reward——比對預測 vs 標準答案的 `tool_calls` 序列;`ToolAudioClosenessReward` 還會真的執行、比對音檔相似度 | 自己的 `parse_tool_call_payload()`（解析單發 JSON,含 code fence/推理文字容錯),真正執行時呼叫 `interface.executor.run_tool_call` |

---

## 7. 看起來像重複、但刻意分開的部分（不要合併!)

這次檢查特別確認過:以下這些「長得很像」的程式碼,是針對不同任務的合理分工,
**不是**架構上該統一進 `interface/` 的重複實作：

1. **`tools_registry.py` 的 `@register` appliers** vs
   **`tool_use_training/gen_2nd_2_data/tool_appliers.py` 的 `APPLIERS`**：
   兩者都是「合成訓練資料時,發明隨機參數 + 呼叫 `.execute()`」,但服務不同的資料生成
   策略（前者是 `gen_1st_stage_data` 的鏈式合成,寫死在 code 裡抽樣；後者是
   `gen_2nd_2_data` 的 disturb 策略,參數範圍來自 `tool_config.json`)。
   `tool_appliers.py` 的 docstring 明講是「故意」複製而非 import,為了讓每個
   生成階段的目錄能獨立演化。這是**資料合成**的關注點,跟 `interface/` 的
   **即時執行 model 預測結果**是不同層次的事,不要混在一起。
2. **`reward.py::parse_tool_call_payload`/`extract_tool_calls`** vs
   `tool_call_formats.py` 的 `parse_turn`：解析的是不同的 wire format
   （單發整包 `{"tool_calls":[...]}`,還可能夾雜推理文字、code fence,需要
   balanced-brace 掃描;vs 逐輪的 `<tool_call>` 標籤)。**執行**的部分已經統一
   （都呼叫 `interface.executor.run_tool_call`),只有「這個特定格式怎麼從文字裡挖出來」
   是分開的,因為格式本身就不同。
3. **`apply_tools.py`（root)/ `tools/tool_execute.py` / `tools/tool_batch_execute.py`**：
   DCASE 資料集的離線批次工具套用,直接用真實 `audio_path`（沒有 `audio_id` 這層
   位址),而且對重量級工具是 subprocess 到專屬 conda env 執行。這是完全不同的
   執行 transport,跟 `interface/executor.py` 的「單次、in-process、
   audio_id 定址」模型是兩回事。
4. **`prompts/__init__.py` / `audio_edit/editor.py`**：DCASE/MCQ QA pipeline,
   直接用 `audio_path` 呼叫工具,不走 `audio_id`/tool_call_formats 這套協定。
   本次重構刻意沒有動這條路徑（使用者也沒有要求),如果之後要統一,需要先決定
   要不要把它也改成 `audio_id` 定址。

---

## 8. 給下一個 task 的注意事項清單

- **改/加一個工具「要不要被 agent 使用」，一律去改 `tools/tools_registry.py`**
  （`@register(...)`)。不要在 `interface/` 裡另外加 allowlist/denylist——
  那樣又會製造出第二份工具清單。
- **工具的 `description()` 字串裡,不要假設它會被 model 看到就直接改成講
  `audio_id`**——那個字串也被 `prompts/`/`audio_edit/editor.py`（真實 audio_path
  呼叫)共用。要修正 model 看到的版本,去改 `to_function_schema()` 裡的替換邏輯。
- **`role: "tool"` 的訊息內容不要手動包 `<tool_response>`**（training data / 任何
  經過 ms-swift `template_backend='swift'`——也就是預設值——的路徑),那層包裝
  ms-swift 會自動做。只有 `VLLMEngine`（零樣本、無微調、無 ms-swift 的路徑)
  需要自己包。
- **`output_audio_id` 一律放進 `parameters`/`arguments` 裡**,不要當獨立欄位——
  這是為了讓 `tool_call_formats.py` 的 function-calling schema 保持標準形狀。
- **新增 tool-calling 慣例（例如未來換一個不是 Qwen 的 model)：在
  `tool_call_formats.py` 加一個新的 `ToolCallFormat` 子類別**,不要去改
  `interface/protocol.py`/`build_dataset.py` 呼叫端的邏輯——那些地方已經設計成
  透過 `tool_call_format` 這個字串參數切換,新增格式不該動到既有呼叫端。
- **`tools_block`（存在 raw dataset JSON 裡的那個 tool 目錄字串)是不透明文字**——
  `official_system_prompt.compose_system_prompt` 純接合,不解析內容。改動它的格式
  只需要改 `tool_call_formats.py`/`tools_registry.describe_available_tools`,
  下游（`build_rl_dataset.py`、`run_eval.py`)完全不用動。
- **同一次資料生成裡,`--tool-call-format` 要對 `tools_block` 生成跟逐輪答案渲染
  用同一個值**——`build_dataset.py` 已經把這個參數從 CLI 一路傳到
  `build_one_sample()`（決定 `tools_block`)跟 `to_swift_sample()`（決定逐輪答案),
  不要繞過這條路徑各自呼叫。
- **`interface/executor.py::run_tool_call` 是執行 model 預測 tool call 的唯一入口**——
  新的 eval script、新的 reward function、新的 agent 變體,都應該 import 這個函式,
  不要重新刻一份 validate + execute 邏輯。
- **Qwen2.5-Omni 跟 Qwen3(-Omni) 的官方 chat_template 行為不一樣**（前者無 tool-calling
  支援、有預設 persona；後者有完整 Hermes 支援、無預設 persona)——幫不同版本的
  Qwen 訓練/推論時,先重新確認這兩點,不要假設兩者一致。
- **這份文件本身也要維護**：改動上述任何一個檔案的職責分工時,回來更新這份文件,
  避免文件跟程式碼漂移。
