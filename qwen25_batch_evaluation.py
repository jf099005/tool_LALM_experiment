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

from uncertainty_quantification_tools import compute_uncertainty
from uncertainty_quantification_tools_batch import compute_p_true_batch

def parse_arguments():
    parser = ArgumentParser(description="Evaluate Qwen2.5-Omni-7B on DCASE subset with optional tool chain results (batched generation).")
    parser.add_argument("--subset_path", type=str, default="MMAU_pro_subset.json", help="Path to the JSON file containing the evaluation subset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-7B", help="vLLM model identifier.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--output_path", type=str, default='predictions.json', help="Path to save detailed predictions as JSON. If not set, predictions will not be saved.")
    parser.add_argument("--load_tool_chain_results", '-tool', action='store_true', help="Whether to load and include tool chain results in the prompt.")
    parser.add_argument('--cont', action = 'store_true')
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
    parser.add_argument('--batch_size', type=int, default=8, help="Number of items submitted to llm.generate() together. Raise this (together with --max_num_seqs) when GPU memory allows, to increase vLLM's continuous-batching throughput.")
    parser.add_argument('--max_num_seqs', type=int, default=16, help="vLLM engine's max concurrent sequences. Should be >= --batch_size to let a whole batch run concurrently.")
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
        f"{user_prompt}"
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
        "Please answer with the best choice letter and the full choice text.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_request(
    item: dict,
    idx: int,
    load_tool_chain_results: bool,
    overwrite_original_audio: bool,
    specify_tool: Optional[str],
    tool_results_root: str,
    system_prompt: str,
) -> dict:
    """Build the vLLM request dict plus bookkeeping metadata for one subset item."""
    question = item.get("question", "")
    choices = item.get("choice", [])
    answer = item.get("answer", "N/A")
    problem_id = item.get("id", f"sample_{idx}")
    audio_path = item.get("audio_path") or item.get("audio_url")

    if not audio_path:
        raise ValueError(f"No audio path found for subset item index {idx}")

    audio_data_raw, sr = load_audio(audio_path)

    tool_output_path = os.path.join(tool_results_root, problem_id)
    if not load_tool_chain_results and overwrite_original_audio:
        raise ValueError("Cannot overwrite original audio if not loading tool chain results, because there will be no tool chain results to overwrite with. Please set --load_tool_chain_results if you want to overwrite original audio with tool chain results.")

    tool_audios_path = []
    if load_tool_chain_results and not overwrite_original_audio:
        prompt, tool_audios_path = build_toolchain_prompt(
            question=question,
            choices=choices,
            system_prompt=system_prompt,
            tool_results_path=tool_output_path,
            specify_tool=specify_tool
        )
    else:
        prompt = build_standard_prompt(
            question=question,
            choices=choices,
            system_prompt=system_prompt
        )
        if load_tool_chain_results and overwrite_original_audio:
            # If we are loading tool chain results and overwriting original audio, we still want to get the tool chain result audio path to load the audio, but we won't include the original audio in the prompt.
            _, tool_audios_path = build_toolchain_prompt(
                question=question,
                choices=choices,
                system_prompt=system_prompt,
                tool_results_path=tool_output_path,
                specify_tool=specify_tool
            )
            assert len(tool_audios_path) == 1, f"Expected exactly one audio file, but got {len(tool_audios_path)} for item index {idx}: {tool_audios_path}"

    audio_data = {"audio": [(audio_data_raw, sr)] + [load_audio(p) for p in tool_audios_path]}
    if overwrite_original_audio:
        audio_data["audio"] = audio_data["audio"][1]

    request = {"prompt": prompt, "multi_modal_data": audio_data}
    meta = {
        "idx": idx,
        "problem_id": problem_id,
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": prompt,
        "audio_data": audio_data,
    }
    return request, meta


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
    batch_size = 8,
) -> None:
    subset = load_subset(subset_path)
    os.environ.pop("TRITON_PTXAS_PATH", None)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    results = prv_results
    prv_ids = {item['id'] for item in prv_results}
    correct = 0

    pending = [(idx, item) for idx, item in enumerate(subset, start=1) if item.get('id') not in prv_ids]
    num_batches = (len(pending) + batch_size - 1) // batch_size

    for batch_start in tqdm(range(0, len(pending), batch_size), total=num_batches, desc="batches"):
        batch = pending[batch_start: batch_start + batch_size]

        batch_requests = []
        batch_meta = []
        for idx, item in batch:
            request, meta = build_request(
                item=item,
                idx=idx,
                load_tool_chain_results=load_tool_chain_results,
                overwrite_original_audio=overwrite_original_audio,
                specify_tool=specify_tool,
                tool_results_root=tool_results_root,
                system_prompt=system_prompt,
            )
            batch_requests.append(request)
            batch_meta.append(meta)

        # A single llm.generate() call over the whole batch lets vLLM's
        # continuous batching run these requests concurrently, instead of
        # the one-request-at-a-time pattern of the original script.
        outputs = llm.generate(
            batch_requests,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        batch_entries = []
        for meta, output in zip(batch_meta, outputs):
            pred = output.outputs[0].text.strip()
            answer = meta["answer"]
            try:
                is_correct = answer != "N/A" and answer.strip() in pred
            except Exception:
                is_correct = False

            result_entry = {
                "index": meta["idx"],
                'prompt': meta["prompt"],
                'load tool chain results': load_tool_chain_results,
                "id": meta["problem_id"],
                "question": meta["question"],
                "choices": meta["choices"],
                "answer": answer,
                "prediction": pred,
                "correct": is_correct,
            }
            batch_entries.append(result_entry)

        if compute_uq:
            needs_samples = (
                uq_compute_predictive_entropy
                or uq_compute_length_normalized_entropy
                or uq_compute_discrete_semantic_entropy
                or uq_compute_semantic
            )
            if not needs_samples and uq_compute_p_true:
                # No stochastic sampling requested, only P(True): the
                # verification prompt for each item depends solely on that
                # item's own greedy prediction (already computed above), not
                # on any other item, so the whole batch's verification
                # prompts can be submitted in a single llm.generate() call.
                p_true_items = [
                    {
                        "prompt": meta["prompt"],
                        "multi_modal_data": meta["audio_data"],
                        "choices": meta["choices"],
                        "question": meta["question"],
                        "greedy_prediction": entry["prediction"],
                    }
                    for meta, entry in zip(batch_meta, batch_entries)
                ]
                p_true_results = compute_p_true_batch(
                    llm,
                    items=p_true_items,
                    system_prompt=system_prompt,
                    max_tokens=5,
                )
                for entry, uq_result in zip(batch_entries, p_true_results):
                    entry["uncertainty"] = uq_result
            else:
                # Stochastic-sampling metrics stay per-item: compute_uncertainty
                # issues its own llm.generate() call per item (already batched
                # internally across its K samples via SamplingParams(n=K, ...)).
                for meta, entry in zip(batch_meta, batch_entries):
                    uq_result = compute_uncertainty(
                        llm,
                        prompt=meta["prompt"],
                        multi_modal_data=meta["audio_data"],
                        choices=meta["choices"],
                        greedy_prediction=entry["prediction"],
                        system_prompt=system_prompt,
                        question=meta["question"],
                        num_samples=uq_num_samples if needs_samples else 0,
                        sample_temperature=uq_sample_temperature,
                        max_tokens=max_tokens,
                        compute_semantic=uq_compute_semantic,
                        compute_p_true=uq_compute_p_true,
                    )
                    if not uq_compute_predictive_entropy:
                        uq_result.pop("predictive_entropy", None)
                    if not uq_compute_length_normalized_entropy:
                        uq_result.pop("length_normalized_entropy", None)
                    if not uq_compute_discrete_semantic_entropy:
                        uq_result.pop("discrete_semantic_entropy", None)
                        uq_result.pop("answer_distribution", None)
                    entry["uncertainty"] = uq_result

        for entry in batch_entries:
            results.append(entry)
            if entry["correct"]:
                correct += 1

        # Checkpoint once per batch (rather than once per item) so --cont
        # resume still works, without adding a disk write per single item.
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = correct / len(results) if results else 0.0
    print("=== Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    if output_path:
        print(f"Saved detailed predictions to: {output_path}")


def main():
    args = parse_arguments()
    SUBSET_PATH       = args.subset_path
    MODEL             = args.model
    TEMPERATURE       = args.temperature
    MAX_TOKENS        = args.max_tokens
    OUTPUT_PATH = args.output_path
    LOAD_TOOL_CHAIN_RESULTS = args.load_tool_chain_results
    TOOL_RESULTS_PATH = args.tool_results_path
    SYSTEM_PROMPT     = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    CACHE_DIR = "/work/u1501463/hf_cache"

    llm = LLM(
        model=MODEL,
        max_model_len=20000,
        max_num_seqs=max(args.max_num_seqs, args.batch_size),
        limit_mm_per_prompt={"audio": 4},
        trust_remote_code=True,
        download_dir = CACHE_DIR
    )

    output_file_name = Path(OUTPUT_PATH).stem
    if LOAD_TOOL_CHAIN_RESULTS:
        output_file_name += "_with_tool_chain_results"
    else:
        output_file_name += "_no_tool_chain_results"
    OUTPUT_PATH = str(Path(OUTPUT_PATH).parent / f"{output_file_name}.json")
    prv_result = []
    if args.cont:
        with open(OUTPUT_PATH) as f:
            prv_result = json.load(f)

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
    )


if __name__ == "__main__":
    main()
