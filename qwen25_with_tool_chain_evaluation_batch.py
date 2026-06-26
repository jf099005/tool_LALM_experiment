from tqdm import tqdm
import json
import os
from typing import List, Optional
from pathlib import Path
from argparse import ArgumentParser

os.environ.pop("TRITON_PTXAS_PATH", None)
import librosa
import numpy as np
from vllm import LLM, SamplingParams

def parse_arguments():
    parser = ArgumentParser(description="Evaluate Qwen2.5-Omni-7B on DCASE subset with optional tool chain results.")
    parser.add_argument("--subset_path", type=str, default="dcase_subset.json", help="Path to the JSON file containing the evaluation subset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-7B", help="vLLM model identifier.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--output_path", type=str, default='predictions.json', help="Path to save detailed predictions as JSON. If not set, predictions will not be saved.")
    parser.add_argument("--load_tool_chain_results", '-tool', action='store_true', help="Whether to load and include tool chain results in the prompt.")

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

def build_toolchain_prompt(system_prompt: str, question: str, choices: List[str], tool_results_path) -> str:
    choice_text = "\n".join(choices) if choices else ""
    audio_token = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    tool_output, tool_audios_path = prompts.tool_chain_result_read.read_tool_chain(
        tool_chain_folder_path=tool_results_path,
        audio_token=audio_token,
        indexing = False,
        tools_mask=['extract_target']
        # tools_mask=['remove_target']
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
) -> None:
    subset = load_subset(subset_path)
    os.environ.pop("TRITON_PTXAS_PATH", None)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    # --- Build all inputs before inference ---
    print("Building prompts and loading audio...")
    batch_inputs = []
    metadata = []

    for idx, item in tqdm(enumerate(subset, start=1), total=len(subset)):
        question = item.get("question", "")
        choices = item.get("choice", [])
        answer = item.get("answer", "N/A")
        problem_id = item.get("id", f"sample_{idx}")
        audio_path = item.get("audio_path") or item.get("audio_url")

        if not audio_path:
            raise ValueError(f"No audio path found for subset item index {idx}")

        audio_data, sr = load_audio(audio_path)

        tool_output_path = os.path.join(
            '/home/u1501463/tool_use_LALM/apply_tool_results_accelerated',
            problem_id
        )

        if load_tool_chain_results:
            prompt, audios_path = build_toolchain_prompt(
                question=question,
                choices=choices,
                system_prompt=system_prompt,
                tool_results_path=tool_output_path
            )
        else:
            prompt = build_standard_prompt(
                question=question,
                choices=choices,
                system_prompt=system_prompt
            )
            audios_path = []

        batch_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"audio": [
                (audio_data, sr)
            ] + [load_audio(p) for p in audios_path]},
        })

        metadata.append({
            "index": idx,
            "prompt": prompt,
            "load tool chain results": load_tool_chain_results,
            "id": problem_id,
            "question": question,
            "choices": choices,
            "answer": answer,
        })

    # --- Run batch inference ---
    print(f"Running batch inference on {len(batch_inputs)} samples...")
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    # --- Collect results ---
    results = []
    correct = 0

    for meta, output in zip(metadata, outputs):
        pred = output.outputs[0].text.strip()
        answer = meta["answer"]
        is_correct = answer != "N/A" and answer.strip() in pred

        results.append({
            **meta,
            "prediction": pred,
            "correct": is_correct,
        })
        if is_correct:
            correct += 1

    accuracy = correct / len(results) if results else 0.0
    print("=== Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed predictions to: {output_path}")


def main():
    args = parse_arguments()
    SUBSET_PATH       = args.subset_path
    MODEL             = args.model
    TEMPERATURE       = args.temperature
    MAX_TOKENS        = args.max_tokens
    OUTPUT_PATH = args.output_path
    LOAD_TOOL_CHAIN_RESULTS = args.load_tool_chain_results
    SYSTEM_PROMPT     = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    CACHE_DIR = "/work/u1501463/hf_cache"

    # os.environ["HF_HOME"] = CACHE_DIR
    # os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
    # os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

    llm = LLM(
        model=MODEL,
        max_model_len=20000,
        max_num_seqs=1,
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

    evaluate(
        llm,
        subset_path=os.path.abspath(SUBSET_PATH),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        system_prompt=SYSTEM_PROMPT,
        output_path=OUTPUT_PATH,
        load_tool_chain_results=LOAD_TOOL_CHAIN_RESULTS
    )


if __name__ == "__main__":
    main()
