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


def _find_undecodable_tokens(tokenizer, generate_ids: List[int], decode_kwargs: Dict[str, Any]) -> List[Dict[str, int]]:
    """Leave-one-out search for which token id(s) in `generate_ids` the tokenizer can't
    turn into valid UTF-8. `tokenizer.decode` never raises on a bad byte sequence -- it
    silently substitutes U+FFFD -- so the only way to localize the culprit is to check
    whether dropping a given position lowers the replacement-character count in the
    redecoded text. Only called after a `�` has already been seen in the real output,
    so the O(n) extra decode calls (n = generated tokens, typically well under 1000) are
    a one-off debugging cost, not a hot path.
    """
    full_text = tokenizer.decode(generate_ids, **decode_kwargs)
    baseline = full_text.count("�")
    if baseline == 0:
        return []
    culprits = []
    for i, token_id in enumerate(generate_ids):
        trial_ids = generate_ids[:i] + generate_ids[i + 1:]
        trial_text = tokenizer.decode(trial_ids, **decode_kwargs)
        if trial_text.count("�") < baseline:
            culprits.append({"position": i, "token_id": token_id})
    return culprits


def _suppress_padding_vocab_ids(engine: Any) -> None:
    """Qwen embedding matrices are padded to a round number for tensor-core
    efficiency -- e.g. Qwen2.5-Omni-7B's `thinker.lm_head` has 152064 rows but
    the tokenizer only assigns strings to ids 0..151664 (`len(tokenizer) ==
    151665`). The remaining ~400 rows were never trained during pretraining
    and have no tokenizer mapping; if generation ever puts one of them in the
    top slot (`�` shows up in the output, `last_decode_warnings` names a
    `token_id` in that gap), `tokenizer.decode` can't render it and the
    surrounding text comes out garbled. Suppressing those ids' logits at every
    decoding step keeps the model inside the ids the tokenizer actually knows.
    Best-effort: silently no-ops for model architectures where the expected
    `thinker` submodule / vocab layout isn't present.
    """
    try:
        vocab_len = len(engine.tokenizer)
        lm_head = engine.model.thinker.get_output_embeddings()
        dead_ids = list(range(vocab_len, lm_head.weight.shape[0]))
    except AttributeError:
        return
    if not dead_ids:
        return
    if engine.model.generation_config is not None:
        engine.model.generation_config.suppress_tokens = dead_ids
    if engine.model.thinker.generation_config is not None:
        engine.model.thinker.generation_config.suppress_tokens = dead_ids


class SwiftEngine:
    def __init__(
        self,
        model: str,
        adapter_dir: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        model_type: Optional[str] = None,
        torch_dtype=None,
        device_map: Optional[Any] = None,
    ):
        # ms-swift's real package lives at site-packages/swift; some envs also have
        # an unrelated `swift` package shadowing it via ~/.local/lib -- drop that.
        sys.path = [p for p in sys.path if ".local" not in p]
        from swift import InferRequest, RequestConfig  # noqa: E402
        from swift.infer_engine import TransformersEngine  # noqa: E402

        self._InferRequest = InferRequest
        adapters = [adapter_dir] if adapter_dir else None
        # fp32 roughly doubles the footprint of the default half-precision load, so
        # a single 7B-class model no longer fits on one GPU -- `device_map="auto"`
        # (the run_eval.py default) shards the model across all visible GPUs via
        # accelerate instead of requiring the caller to fit it on one device.
        self.engine = TransformersEngine(
            model, adapters=adapters, model_type=model_type, torch_dtype=torch_dtype, device_map=device_map,
        )
        _suppress_padding_vocab_ids(self.engine)
        # `return_details=True` is what makes ms-swift populate `choice.token_ids` with
        # the raw generated ids alongside the decoded text -- needed below to localize
        # any `�` the decoder produced back to the specific token id that caused it.
        self.request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, return_details=True)
        # Diagnostics for the most recent `generate_turn` call: a list of
        # {"position": int, "token_id": int} for any token the tokenizer couldn't
        # decode to valid UTF-8 (empty when the output had no `�`). Callers that
        # want this (e.g. run_eval.py) read it right after each `generate_turn` call;
        # it's kept out of the return value so the `-> str` contract this class shares
        # with `VLLMEngine` and `agent.ToolCallingAgent` doesn't change.
        self.last_decode_warnings: List[Dict[str, int]] = []

    def generate_turn(self, messages: List[Dict[str, Any]], audios: List[str]) -> str:
        request = self._InferRequest(messages=messages, audios=audios)
        response = self.engine.infer([request], self.request_config)[0]
        choice = response.choices[0]
        content = choice.message.content
        self.last_decode_warnings = []
        if "�" in content and choice.token_ids:
            decode_kwargs = {"skip_special_tokens": False, "spaces_between_special_tokens": False}
            self.last_decode_warnings = _find_undecodable_tokens(self.engine.tokenizer, choice.token_ids, decode_kwargs)
        return content


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
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
    ):
        import librosa  # noqa: E402
        from vllm import LLM, SamplingParams  # noqa: E402

        self._librosa = librosa
        self._sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
        # fp32 roughly doubles the footprint of vLLM's default half-precision load, so
        # a single 7B-class model no longer fits on one GPU -- `tensor_parallel_size`
        # (defaults to 2 in run_eval.py) shards the model across GPUs instead of
        # requiring the caller to fit it on one device.
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"audio": max_audios_per_prompt},
            trust_remote_code=True,
            download_dir=download_dir,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._audio_cache: Dict[str, Any] = {}
        # See `SwiftEngine.last_decode_warnings` for what this holds and why.
        self.last_decode_warnings: List[Dict[str, int]] = []

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
        completion = outputs[0].outputs[0]
        text = completion.text.strip()
        self.last_decode_warnings = []
        if "�" in text and completion.token_ids:
            decode_kwargs = {"skip_special_tokens": False, "spaces_between_special_tokens": False}
            self.last_decode_warnings = _find_undecodable_tokens(
                self.llm.get_tokenizer(), list(completion.token_ids), decode_kwargs
            )
        return text
