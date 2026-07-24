"""Model backends for the tool-calling interface.

Both backends expose the same `generate_turn(messages, audios) -> str`
interface `agent.ToolCallingAgent` drives. `testing_tool_use_benchmark/run_eval.py`
imports these same classes directly for its benchmark rather than keeping its
own copy, so the two never drift apart.

- `SwiftEngine` -- ms-swift's `TransformersEngine`. Handles both a fine-tuned
  LoRA checkpoint (pass `adapter_dir`) *and* a raw official model id/path
  with no adapter (any model ms-swift recognizes via `model_type`, e.g.
  "Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen2-Audio-7B-Instruct", ...). Must run
  under the `ms-swift` conda env.
- `VLLMEngine` -- plain vLLM, no ms-swift/adapter dependency. Fastest path to
  run an out-of-the-box official checkpoint zero-shot. Must run under an env
  with vLLM installed.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from .protocol import AUDIO_TOKEN


class SwiftEngine:
    def __init__(
        self,
        model: str,
        adapter_dir: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        model_type: Optional[str] = None,
        torch_dtype=None,
    ):
        # ms-swift's real package lives at site-packages/swift; some envs also have
        # an unrelated `swift` package shadowing it via ~/.local/lib -- drop that.
        sys.path = [p for p in sys.path if ".local" not in p]
        from swift import InferRequest, RequestConfig  # noqa: E402
        from swift.infer_engine import TransformersEngine  # noqa: E402

        self._InferRequest = InferRequest
        adapters = [adapter_dir] if adapter_dir else None
        self.engine = TransformersEngine(model, adapters=adapters, model_type=model_type, torch_dtype=torch_dtype)
        self.request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)

    def generate_turn(self, messages: List[Dict[str, Any]], audios: List[str]) -> str:
        request = self._InferRequest(messages=messages, audios=audios)
        response = self.engine.infer([request], self.request_config)[0]
        return response.choices[0].message.content


class VLLMEngine:
    """Raw vLLM backend for running an unmodified official checkpoint.

    No LoRA/adapter support by design -- use `SwiftEngine` for a fine-tuned
    checkpoint. Renders the conversation as ChatML
    (`<|im_start|>{role}\\n{content}<|im_end|>\\n`), substituting each
    `AUDIO_TOKEN` occurrence for vLLM's `<|audio_bos|><|AUDIO|><|audio_eos|>`
    sentinel, in the same order audios were appended to the conversation.

    This is the one path in the project with no templating layer of its own
    underneath it (unlike `SwiftEngine`, whose ms-swift `Template` machinery
    already wraps a `role: "tool"` message's content in
    `<tool_response>...</tool_response>` at encode time -- see
    `tool_call_formats.py`'s module docstring), so `_render_prompt` does that
    wrapping itself, merging consecutive tool turns into one `user` block --
    matching Qwen's own official chat template convention (verified against
    the Qwen3-Omni chat_template.json snapshot) for a raw, un-fine-tuned
    checkpoint that expects exactly that convention. Hardcoded to Qwen's
    ChatML/ `<tool_response>` syntax throughout, same as the rest of this
    class -- a future non-Qwen zero-shot vLLM backend would need its own
    renderer, not a `tool_call_format` switch here.
    """

    _AUDIO_SENTINEL = "<|audio_bos|><|AUDIO|><|audio_eos|>"

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_model_len: int = 20000,
        max_num_seqs: int = 8,
        max_audios_per_prompt: int = 12,
        download_dir: Optional[str] = None,
    ):
        import librosa  # noqa: E402
        from vllm import LLM, SamplingParams  # noqa: E402

        self._librosa = librosa
        self._sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"audio": max_audios_per_prompt},
            trust_remote_code=True,
            download_dir=download_dir,
        )
        self._audio_cache: Dict[str, Any] = {}

    def _load_audio(self, path: str, sr: int = 16000):
        if path not in self._audio_cache:
            audio, _ = self._librosa.load(path, sr=sr, mono=True)
            self._audio_cache[path] = audio.astype("float32")
        return self._audio_cache[path], sr

    def _render_prompt(self, messages: List[Dict[str, Any]]) -> str:
        parts = []
        i, n = 0, len(messages)
        while i < n:
            msg = messages[i]
            if msg["role"] == "tool":
                # Qwen's official chat template merges one or more consecutive
                # `tool` turns into a single `user` block, each wrapped in
                # <tool_response></tool_response> -- it never emits a literal
                # `<|im_start|>tool` turn.
                parts.append("<|im_start|>user\n")
                while i < n and messages[i]["role"] == "tool":
                    content = messages[i]["content"].replace(AUDIO_TOKEN, self._AUDIO_SENTINEL)
                    parts.append(f"<tool_response>\n{content}\n</tool_response>\n")
                    i += 1
                parts.append("<|im_end|>\n")
                continue
            content = msg["content"].replace(AUDIO_TOKEN, self._AUDIO_SENTINEL)
            parts.append(f"<|im_start|>{msg['role']}\n{content}<|im_end|>\n")
            i += 1
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def generate_turn(self, messages: List[Dict[str, Any]], audios: List[str]) -> str:
        prompt = self._render_prompt(messages)
        multi_modal_data = {"audio": [self._load_audio(p) for p in audios]}
        outputs = self.llm.generate(
            [{"prompt": prompt, "multi_modal_data": multi_modal_data}],
            sampling_params=self._sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()
