from tqdm import tqdm
import json
import os
from typing import List, Optional
from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.pop("TRITON_PTXAS_PATH", None)
import librosa
import numpy as np
from vllm import LLM, SamplingParams

from concurrent.futures import ThreadPoolExecutor

from uncertainty_quantification_tools import compute_uncertainty_batch

def parse_arguments():
    parser = ArgumentParser(description="Evaluate Qwen2.5-Omni-7B on DCASE subset with optional tool chain results.")
    parser.add_argument("--subset_path", type=str, default="MMAU_pro_subset.json", help="Path to the JSON file containing the evaluation subset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-7B", help="vLLM model identifier.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--output_path", type=str, default='predictions.json', help="Path to save detailed predictions as JSON. If not set, predictions will not be saved.")
    parser.add_argument("--load_tool_chain_results", '-tool', action='store_true', help="Whether to load and include tool chain results in the prompt.")
    parser.add_argument('--cont', '--resume', dest='cont', action=BooleanOptionalAction, default=True, help="Resume from an existing output file (skips ids already done). On by default; pass --no-cont to start over.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=float(os.environ.get("VLLM_GPU_MEM_UTIL", 0.9)), help="Fraction of GPU memory vLLM may use.")
    parser.add_argument('--max_model_len', type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", 20000)), help="Max context length; lower it to save KV-cache memory on small GPUs.")
    parser.add_argument('--max_num_seqs', type=int, default=16, help="Max concurrent sequences (must be >= --uq_num_samples to sample them in one batch).")
    parser.add_argument('--max_num_batched_tokens', type=int, default=None, help="Cap on tokens per scheduler step (vLLM default 8192). Lower it if engine start-up OOMs during memory profiling.")
    parser.add_argument('--paired_subset_path', type=str, default=None, help="Optional second subset evaluated in lockstep with --subset_path (e.g. tool-applied vs original audio of the same items): items are interleaved into the same batches, each subset writes its own results.")
    parser.add_argument('--paired_output_path', type=str, default=None, help="Output path for --paired_subset_path (same suffix rules as --output_path).")
    parser.add_argument('--mm_processor_cache_gb', type=float, default=0, help="vLLM multimodal preprocessor cache size (GiB, kept in both the frontend and engine processes). Default 0: every audio clip is seen once here, so the cache only costs RAM (vLLM's default 4 GiB per process ran a 31 GB machine out of memory).")
    parser.add_argument('--batch_size', type=int, default=1, help="Items per llm.generate call (greedy, UQ samples and P(True) are each batched across items). Progress is saved after every batch.")
    parser.add_argument('--max_audios_per_prompt', type=int, default=4, help="Max audio clips per prompt (1 is enough without --load_tool_chain_results).")
    parser.add_argument('--enforce_eager', action='store_true', help="Disable CUDA graphs (saves some GPU memory, slightly slower).")
    parser.add_argument('--download_dir', type=str, default=os.environ.get("HF_HUB_CACHE") or None, help="Where vLLM downloads weights when --model is a hub id.")
    parser.add_argument('--system_prompt', type=str, default=None, help="Override the system prompt. Default: Qwen-Omni prompt for Omni models, generic assistant prompt otherwise.")
    parser.add_argument('--overwrite_original_audio', action = 'store_true', help="Whether to overwrite the original audio with the tool chain result audio. If set, the original audio will not be included in the prompt even if tool chain results are loaded.")
    parser.add_argument('--specify_tool', default=None, help="The name of the tool to load results from, e.g., 'extract_target' or 'remove_target'. This argument is required if --load_tool_chain_results is set.")
    parser.add_argument('--tool_results_path', default='/home/u1501463/tool_use_LALM/apply_tool_results', help="Path to the folder containing tool chain results.")
    parser.add_argument('--compute_uncertainty', '-uq', action='store_true', help="Whether to compute uncertainty quantification for each prediction. Master switch; individual metrics below are only computed if this is set.")
    parser.add_argument('--uq_num_samples', type=int, default=10, help="Number of stochastic samples to draw per item. Required (>0) for predictive/length-normalized/discrete-semantic entropy; set to 0 to skip all sample-based metrics (e.g. to only compute P(True)).")
    parser.add_argument('--uq_sample_temperature', type=float, default=1.0, help="Sampling temperature used when drawing UQ samples.")
    parser.add_argument('--uq_predictive_entropy', action=BooleanOptionalAction, default=True, help="Whether to report predictive entropy (H_pred). Requires --uq_num_samples > 0.")
    parser.add_argument('--uq_length_normalized_entropy', action=BooleanOptionalAction, default=True, help="Whether to report length-normalized entropy (H_norm_tok). Requires --uq_num_samples > 0.")
    parser.add_argument('--uq_discrete_semantic_entropy', action=BooleanOptionalAction, default=True, help="Whether to report discrete semantic entropy (H_disc) and the answer distribution. Requires --uq_num_samples > 0.")
    parser.add_argument('--uq_semantic_entropy', action=BooleanOptionalAction, default=False, help="Whether to additionally compute NLI-based semantic entropy (H_sem). Loads a cross-encoder model on first use; leave off unless needed.")
    parser.add_argument('--uq_p_true', action=BooleanOptionalAction, default=True, help="Whether to compute the P(True) self-verification metric (one extra generation per item).")
    return parser.parse_args()


def load_audio(path: str, sr: int = 16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr
    

def load_subset(subset_path: str) -> List[dict]:
    with open(subset_path, "r", encoding="utf-8") as f:
        subset = json.load(f)
    if not isinstance(subset, list):
        raise ValueError("Subset file must contain a JSON list.")
    return subset

import prompts


def save_results_atomic(results: List[dict], output_path: str) -> None:
    """Write to a temp file then rename, so an interrupt (Ctrl+C, OOM kill,
    power loss) never leaves a half-written, unparsable results file."""
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


def partial_path(output_path: str) -> str:
    return f"{output_path}.partial.jsonl"


def append_partial(entries: List[dict], output_path: str) -> None:
    """Progress log: one JSON line per finished item, appended after every batch.
    Cheap at any dataset size (unlike rewriting the full JSON each time); merged
    into `output_path` when the run completes."""
    with open(partial_path(output_path), "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_previous_results(output_path: str) -> List[dict]:
    """Finished results from `output_path` (a completed or older run) plus the
    progress log of an interrupted run. A line cut off by a crash is ignored."""
    results: List[dict] = []
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)
            results = data if isinstance(data, list) else []
        except json.JSONDecodeError:
            backup = f"{output_path}.corrupt"
            os.replace(output_path, backup)
            print(f"Warning: {output_path} was unreadable; moved to {backup}.")
    seen = {r.get("id") for r in results}
    partial = partial_path(output_path)
    if os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("id") not in seen:
                    seen.add(entry.get("id"))
                    results.append(entry)
    return results

def build_toolchain_prompt(system_prompt: str, question: str, choices: List[str], tool_results_path, specify_tool=None) -> str:
    choice_text = "\n".join(choices) if choices else ""
    audio_token = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    tool_output, tool_audios_path = prompts.tool_chain_result_read.read_tool_chain(
        tool_chain_folder_path=tool_results_path,
        audio_token=audio_token,
        indexing = False,
        specify_tool=specify_tool
    )

    _, user_prompt = prompts.QA_prompt(
        question=question,
        options=choice_text,
        audio_token=audio_token,
        tool_results=tool_output,
        tools_list=None,
        tools_description=None,
        final_round=True
    )
    
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        # "Focus on the audio clips and answer the question."
        f"{user_prompt}"
        # "Please think step by step, then output the best option letter and the full text of the selected option on the final line."
        "Please answer with the best choice letter and the full choice text.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ), tool_audios_path

def build_standard_prompt(question: str, choices: List[str], system_prompt: str) -> str:
    choice_text = "\n".join(choices) if choices else ""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        "Focus on the audio clip and answer the question."
        "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        f"Question: {question}\n"
        f"Choose from the following options:\n{choice_text}\n"
        # "Please think step by step, then output the best option letter and the full text of the selected option on the final line."
        "Please answer with the best choice letter and the full choice text.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def evaluate(
    llm,
    subset_path: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    output_path: str,
    load_tool_chain_results = False,
    prv_results = [],
    overwrite_original_audio = False,
    specify_tool = None,
    tool_results_root = '/home/u1501463/tool_use_LALM/apply_tool_results',
    compute_uq = False,
    uq_num_samples = 10,
    uq_sample_temperature = 1.0,
    uq_compute_predictive_entropy = True,
    uq_compute_length_normalized_entropy = True,
    uq_compute_discrete_semantic_entropy = True,
    uq_compute_semantic = False,
    uq_compute_p_true = True,
    batch_size = 1,
    paired_subset_path = None,
    paired_output_path = None,
    paired_prv_results = (),
) -> None:
    """Evaluate one subset, or two in lockstep (`paired_*`): e.g. the original and the
    tool-applied version of the same items. Paired subsets are interleaved item by item
    into the same batches (same model in memory, same throughput), each writing its own
    output file, so at any moment both have covered the same items and can be compared."""
    os.environ.pop("TRITON_PTXAS_PATH", None)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    streams = [{"subset": load_subset(subset_path), "output_path": output_path, "prv": prv_results}]
    if paired_subset_path:
        streams.append({"subset": load_subset(paired_subset_path), "output_path": paired_output_path,
                        "prv": paired_prv_results})
    for st in streams:
        st["results"] = list(st["prv"])
        st["done_ids"] = {item['id'] for item in st["prv"]}
        st["correct"] = sum(1 for item in st["prv"] if item.get('correct'))
        if st["done_ids"]:
            print(f"Resuming: {len(st['done_ids'])} item(s) already done in {st['output_path']}")
    if not load_tool_chain_results and overwrite_original_audio:
        raise ValueError("Cannot overwrite original audio if not loading tool chain results, because there will be no tool chain results to overwrite with. Please set --load_tool_chain_results if you want to overwrite original audio with tool chain results.")

    def prepare(idx, item):
        """Prompt + loaded audio for one subset item (runs in a worker thread)."""
        question = item.get("question", "")
        choices = item.get("choice", [])
        problem_id = item.get("id", f"sample_{idx}")
        audio_path = item.get("audio_path") or item.get("audio_url")
        if not audio_path:
            raise ValueError(f"No audio path found for subset item index {idx}")
        audio_data, sr = load_audio(audio_path)
        tool_output_path = os.path.join(tool_results_root, problem_id)
        if load_tool_chain_results and not overwrite_original_audio:
            prompt, tool_audios_path = build_toolchain_prompt(
                question=question, choices=choices, system_prompt=system_prompt,
                tool_results_path=tool_output_path, specify_tool=specify_tool)
        else:
            prompt = build_standard_prompt(question=question, choices=choices, system_prompt=system_prompt)
            tool_audios_path = []
            if load_tool_chain_results and overwrite_original_audio:
                # Load the tool-chain result audio but don't include the original audio in the prompt.
                _, tool_audios_path = build_toolchain_prompt(
                    question=question, choices=choices, system_prompt=system_prompt,
                    tool_results_path=tool_output_path, specify_tool=specify_tool)
                assert len(tool_audios_path) == 1, f"Expected exactly one audio file, but got {len(tool_audios_path)} for item index {idx}: {tool_audios_path}"
        mm_data = {"audio": [(audio_data, sr)] + [load_audio(p) for p in tool_audios_path]}
        if overwrite_original_audio:
            mm_data["audio"] = [mm_data["audio"][1]]
        return {
            "idx": idx, "id": problem_id, "question": question, "choices": choices,
            "answer": item.get("answer", "N/A"), "prompt": prompt, "multi_modal_data": mm_data,
        }

    needs_samples = (
        uq_compute_predictive_entropy
        or uq_compute_length_normalized_entropy
        or uq_compute_discrete_semantic_entropy
        or uq_compute_semantic
    )

    def run_batch(batch):
        outputs = llm.generate(
            [{"prompt": b["prompt"], "multi_modal_data": b["multi_modal_data"]} for b in batch],
            sampling_params=sampling_params, use_tqdm=False)
        entries = []
        for b, out in zip(batch, outputs):
            pred = out.outputs[0].text.strip()
            b["greedy_prediction"] = pred
            answer = b["answer"]
            try:
                is_correct = answer != "N/A" and answer.strip() in pred
            except Exception:
                is_correct = False
            entries.append({
                "index": b["idx"],
                'prompt': b["prompt"],
                'load tool chain results': load_tool_chain_results,
                "id": b["id"],
                "question": b["question"],
                "choices": b["choices"],
                "answer": answer,
                "prediction": pred,
                "correct": is_correct,
            })
        if compute_uq:
            uq_results = compute_uncertainty_batch(
                llm, batch, system_prompt=system_prompt,
                num_samples=uq_num_samples if needs_samples else 0,
                sample_temperature=uq_sample_temperature, max_tokens=max_tokens,
                compute_semantic=uq_compute_semantic, compute_p_true=uq_compute_p_true)
            for entry, uq_result in zip(entries, uq_results):
                if not uq_compute_predictive_entropy:
                    uq_result.pop("predictive_entropy", None)
                if not uq_compute_length_normalized_entropy:
                    uq_result.pop("length_normalized_entropy", None)
                if not uq_compute_discrete_semantic_entropy:
                    uq_result.pop("discrete_semantic_entropy", None)
                    uq_result.pop("answer_distribution", None)
                entry["uncertainty"] = uq_result
        return entries

    # (stream index, item index, item), interleaved: item 1 of every stream, then item 2, ...
    pending = []
    for idx in range(1, max(len(st["subset"]) for st in streams) + 1):
        for si, st in enumerate(streams):
            if idx <= len(st["subset"]) and st["subset"][idx - 1].get('id') not in st["done_ids"]:
                pending.append((si, idx, st["subset"][idx - 1]))
    chunks = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    total = sum(len(st["subset"]) for st in streams)
    pbar = tqdm(total=total, initial=total - len(pending))

    def prepare_tagged(task):
        si, idx, item = task
        prepared = prepare(idx, item)
        prepared["stream"] = si
        return prepared

    # Audio decoding/resampling for the next batch overlaps with generation of the current one.
    with ThreadPoolExecutor(max_workers=4) as pool:
        future = pool.map(prepare_tagged, chunks[0]) if chunks else None
        for ci in range(len(chunks)):
            batch = list(future)
            future = pool.map(prepare_tagged, chunks[ci + 1]) if ci + 1 < len(chunks) else None
            entries = run_batch(batch)
            for si, st in enumerate(streams):
                mine = [e for b, e in zip(batch, entries) if b["stream"] == si]
                if not mine:
                    continue
                st["results"].extend(mine)
                st["correct"] += sum(e["correct"] for e in mine)
                if st["output_path"]:
                    append_partial(mine, st["output_path"])
            pbar.update(len(entries))
    pbar.close()

    for st in streams:
        if st["output_path"]:
            save_results_atomic(st["results"], st["output_path"])
            partial = partial_path(st["output_path"])
            if os.path.exists(partial):
                os.remove(partial)
        n = len(st["results"])
        print("=== Evaluation Summary ===")
        print(f"Total samples: {n}")
        print(f"Correct predictions: {st['correct']}")
        print(f"Accuracy: {(st['correct'] / n if n else 0.0):.2%}")
        if st["output_path"]:
            print(f"Saved detailed predictions to: {st['output_path']}")


def mm_memory_kwargs(model: str, max_audios: int) -> dict:
    """Multimodal limits that keep vLLM's start-up memory profiling small.

    This task is audio-only, but for Qwen2.5-Omni vLLM 0.10.1 profiles a full-size
    dummy image/video through the vision tower (OOM on a 12 GB card). Setting
    image/video to 0 is not an option: vLLM's encoder-cache budget for this model
    only counts image/video tokens (audio is missing), so with both at 0 the cache is
    never created and audio requests hang forever. Instead keep one tiny image
    (max_pixels) and no video.
    """
    if "omni" not in model.lower():
        return {"limit_mm_per_prompt": {"audio": max_audios}}
    return {
        "limit_mm_per_prompt": {"audio": max_audios, "image": 1, "video": 0},
        "mm_processor_kwargs": {"max_pixels": 64 * 28 * 28},
    }


def main():
    args = parse_arguments()
    SUBSET_PATH       = args.subset_path
    MODEL             = args.model
    TEMPERATURE       = args.temperature
    MAX_TOKENS        = args.max_tokens
    OUTPUT_PATH = args.output_path
    LOAD_TOOL_CHAIN_RESULTS = args.load_tool_chain_results
    TOOL_RESULTS_PATH = args.tool_results_path
    if args.system_prompt is not None:
        SYSTEM_PROMPT = args.system_prompt
    elif "omni" in MODEL.lower():
        SYSTEM_PROMPT = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        )
    else:
        # e.g. Qwen2-Audio-7B-Instruct, which uses the same <|audio_bos|><|AUDIO|><|audio_eos|> tokens.
        SYSTEM_PROMPT = "You are a helpful assistant."

    llm = LLM(
        model=MODEL,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        **mm_memory_kwargs(MODEL, args.max_audios_per_prompt),
        trust_remote_code=True,
        download_dir=args.download_dir,
    )

    output_file_name = Path(OUTPUT_PATH).stem
    if LOAD_TOOL_CHAIN_RESULTS:
        output_file_name += "_with_tool_chain_results"
    else:
        output_file_name += "_no_tool_chain_results"
    suffix = "_with_tool_chain_results" if LOAD_TOOL_CHAIN_RESULTS else "_no_tool_chain_results"
    OUTPUT_PATH = str(Path(OUTPUT_PATH).parent / f"{output_file_name}.json")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    prv_result = load_previous_results(OUTPUT_PATH) if args.cont else []
    PAIRED_OUTPUT_PATH = None
    paired_prv = []
    if args.paired_subset_path:
        if not args.paired_output_path:
            raise ValueError("--paired_subset_path needs --paired_output_path")
        p = Path(args.paired_output_path)
        PAIRED_OUTPUT_PATH = str(p.parent / f"{p.stem}{suffix}.json")
        Path(PAIRED_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        paired_prv = load_previous_results(PAIRED_OUTPUT_PATH) if args.cont else []

    evaluate(
        llm,
        subset_path=os.path.abspath(SUBSET_PATH),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        system_prompt=SYSTEM_PROMPT,
        output_path=OUTPUT_PATH,
        load_tool_chain_results=LOAD_TOOL_CHAIN_RESULTS,
        prv_results=prv_result,
        overwrite_original_audio=args.overwrite_original_audio,
        specify_tool=args.specify_tool,
        tool_results_root=TOOL_RESULTS_PATH,
        compute_uq=args.compute_uncertainty,
        uq_num_samples=args.uq_num_samples,
        uq_sample_temperature=args.uq_sample_temperature,
        uq_compute_predictive_entropy=args.uq_predictive_entropy,
        uq_compute_length_normalized_entropy=args.uq_length_normalized_entropy,
        uq_compute_discrete_semantic_entropy=args.uq_discrete_semantic_entropy,
        uq_compute_semantic=args.uq_semantic_entropy,
        uq_compute_p_true=args.uq_p_true,
        batch_size=args.batch_size,
        paired_subset_path=os.path.abspath(args.paired_subset_path) if args.paired_subset_path else None,
        paired_output_path=PAIRED_OUTPUT_PATH,
        paired_prv_results=paired_prv,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Results are saved after every item, so nothing is lost; rerun the same command to resume.
        print("\nInterrupted. Progress is saved; rerun the same command to resume.")
        raise SystemExit(130)