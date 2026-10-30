"""CPU-only smoke test for audio_tool_agent_loop.py (no GPU, no vLLM, no Ray).

Replays one scripted episode against a fake LLM server -- turn 1 calls `clipping` on audio_0
(executed for real through interface.executor), turn 2 answers -- and checks:
  1. #audio placeholder runs in the token stream == #audios handed to rollout/actor;
  2. tool-turn tokens are masked out (response_mask == 0), model tokens are in (== 1);
  3. the full token sequence equals the HF chat template + processor rendering of the same
     conversation (= the ChatML that stage-1 SFT trained on), i.e. no train/rollout drift;
  4. reward.compute_score scores the episode as correct.

    python tests/smoke_agent_loop.py [--parquet .../train.parquet] [--row 0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("HEAVY_TOOL_POOL_DIR", "/nonexistent-smoke-test")  # no heavy tool is called here

import pandas as pd  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from verl.utils.dataset.rl_dataset import RLHFDataset  # noqa: E402
from verl_omni.utils.dataset.omni_rl_datasets import QwenOmniRLHFDataset  # noqa: E402

import reward  # noqa: E402
from audio_tool_agent_loop import AudioToolMCQAgentLoop  # noqa: E402
from verl.utils.tokenizer.continuous_token_wiring import ContinuousTokenModelFamily, create_continuous_token_builder  # noqa: E402

MODEL = "/work/u1501463/Qwen3-Omni-30B-A3B-Instruct"


class FakeServer:
    def __init__(self, tokenizer, turns):
        self.tokenizer, self.turns, self.calls = tokenizer, list(turns), []

    async def generate(self, request_id, prompt_ids, sampling_params, audio_data=None, **kw):
        self.calls.append({"n_prompt": len(prompt_ids), "n_audio": len(audio_data or [])})
        text = self.turns.pop(0)
        ids = self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
        return SimpleNamespace(token_ids=ids, log_probs=None, num_preempted=0, extra_fields={}, routed_experts=None)


def build_loop(processor, server) -> AudioToolMCQAgentLoop:
    loop = AudioToolMCQAgentLoop.__new__(AudioToolMCQAgentLoop)
    loop.tokenizer, loop.processor, loop.server_manager = processor.tokenizer, processor, server
    loop.dataset_cls, loop.data_config = QwenOmniRLHFDataset, {}
    loop.apply_chat_template_kwargs, loop.mm_processor_kwargs = {}, {"sampling_rate": 16000}
    loop.rollout_config = SimpleNamespace(prompt_length=6144, response_length=4096,
                                          multi_turn=SimpleNamespace(max_assistant_turns=4))
    loop.loop = asyncio.get_event_loop()
    loop.continuous_token_builder = create_continuous_token_builder(
        loop.tokenizer, model_family=ContinuousTokenModelFamily.QWEN_VL,
        chat_template_kwargs={}, mm_processor_kwargs=loop.mm_processor_kwargs, processor=processor)
    loop.max_assistant_turns, loop.response_length = 4, 4096
    from interface import protocol
    loop.tool_names = set(protocol.audio_to_audio_tool_names())
    return loop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/work/u1501463/stage3_mcq_rl/data_smoke/train.parquet")
    ap.add_argument("--row", type=int, default=0)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    row = pd.read_parquet(args.parquet).iloc[args.row].to_dict()
    row["prompt"] = [dict(m) for m in row["prompt"]]
    row["audios"] = list(row["audios"])
    raw_prompt = RLHFDataset._build_messages(SimpleNamespace(image_key="images", video_key="videos",
                                                             audio_key="audios", processor=processor), row, "prompt")
    gt = row["reward_model"]["ground_truth"]

    call = '<tool_call>\n{"name": "clipping", "arguments": {"audio_begin": "00:00:00.000", "audio_end": "00:00:03.000", "audio_id": "audio_0"}}\n</tool_call>'
    turns = ["I should zoom into the first three seconds.\n" + call, f"<answer>{gt}</answer>"]
    server = FakeServer(tok, turns)
    loop = build_loop(processor, server)
    os.environ["STAGE3_KEEP_TOOL_OUTPUTS"] = "0"
    out = loop.loop.run_until_complete(loop.run({"temperature": 1.0}, raw_prompt=raw_prompt))

    ids = out.prompt_ids + out.response_ids
    pad = tok.convert_tokens_to_ids("<|audio_pad|>")
    runs = sum(1 for i, t in enumerate(ids) if t == pad and (i == 0 or ids[i - 1] != pad))
    n_audio = len(out.multi_modal_data["audios"])
    print("server calls:", server.calls)
    print("extra_fields:", {k: v for k, v in out.extra_fields.items() if k != "final_text"})
    print(f"audio placeholder runs={runs}  audios={n_audio}")
    assert runs == n_audio == 2, "placeholder/audio mismatch"

    mask = out.response_mask
    masked = tok.decode([t for t, m in zip(out.response_ids, mask) if m == 0])
    trained = tok.decode([t for t, m in zip(out.response_ids, mask) if m == 1])
    print("--- masked (env) tokens:\n" + masked.replace("<|audio_pad|>", "")[:400])
    print("--- trained (policy) tokens:\n" + trained)
    assert "<tool_response>" in masked and "<tool_response>" not in trained and "<tool_call>" in trained

    # Reference: render the same conversation in one shot with the HF template + processor.
    audio_iter = iter(out.multi_modal_data["audios"])
    ref_msgs = []
    for m in raw_prompt:
        content = m["content"]
        if isinstance(content, list):
            content = [{"type": "audio", "audio": next(audio_iter)} if b.get("type") == "audio" else b for b in content]
        ref_msgs.append({"role": m["role"], "content": content})
    ref_msgs += [
        {"role": "assistant", "content": turns[0]},
        {"role": "tool", "content": [{"type": "text", "text": "Output of step 1: <audio_1>"},
                                     {"type": "audio", "audio": next(audio_iter)}]},
        {"role": "assistant", "content": turns[1]},
    ]
    text = processor.apply_chat_template(ref_msgs, tokenize=False, add_generation_prompt=False)
    audios = [b["audio"] for m in ref_msgs if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "audio"]
    ref_ids = processor(text=[text], audio=audios, sampling_rate=16000, return_tensors="pt")["input_ids"][0].tolist()
    ref_ids = ref_ids[:-1] if ref_ids[-1] == tok.encode("\n", add_special_tokens=False)[0] else ref_ids  # final "\n"
    same = ids == ref_ids
    print(f"token-identical to HF template rendering: {same}  (loop={len(ids)} ref={len(ref_ids)})")
    if not same:
        k = next(i for i, (a, b) in enumerate(zip(ids, ref_ids)) if a != b)
        print("first diff at", k, repr(tok.decode(ids[k - 10:k + 10])), "vs", repr(tok.decode(ref_ids[k - 10:k + 10])))
    assert same

    score = reward.compute_score(row["data_source"], tok.decode(out.response_ids), gt, extra_info=dict(out.extra_fields))
    print("reward:", json.dumps(score))
    assert score["acc"] == 1.0 and score["num_tool_success"] == 1.0
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
