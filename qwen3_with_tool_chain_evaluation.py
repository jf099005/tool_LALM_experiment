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
    parser = ArgumentParser(description="Evaluate Qwen3-Omni on DCASE subset with optional tool chain results.")
    parser.add_argument("--subset_path", type=str, default="MMAU_pro_subset.json", help="Path to the JSON file containing the evaluation subset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="vLLM model identifier.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate.")
    parser.add_argument("--output_path", type=str, default='predictions.json', help="Path to save detailed predictions as JSON. If not set, predictions will not be saved.")
    parser.add_argument("--load_tool_chain_results", '-tool', action='store_true', help="Whether to load and include tool chain results in the prompt.")
    parser.add_argument('--cont', action = 'store_true')
    parser.add_argument('--overwrite_original_audio', action = 'store_true', help="Whether to overwrite the original audio with the tool chain result audio. If set, the original audio will not be included in the prompt even if tool chain results are loaded.")
    parser.add_argument('--specify_tool', default=None, help="The name of the tool to load results from, e.g., 'extract_target' or 'remove_target'. This argument is required if --load_tool_chain_results is set.")
    parser.add_argument('--tool_results_path', default='/home/u1501463/tool_use_LALM/apply_tool_results', help="Path to the folder containing tool chain results.")
    parser.add_argument('--tensor_parallel_size', type=int, default=2, help="Number of GPUs to use for tensor parallelism (Qwen3-Omni-30B-A3B in fp16 needs ~60GB, so 2x V100-32GB are required).")
    parser.add_argument('--dtype', default='float16', help="Model dtype. V100 GPUs (compute capability 7.0) do not support bfloat16, so float16 is used by default.")
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
    audio_token = "<|audio_start|><|audio_pad|><|audio_end|>"
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
        "<|audio_start|><|audio_pad|><|audio_end|>\n"
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
    tool_results_root = '/home/u1501463/tool_use_LALM/apply_tool_results'
) -> None:
    subset = load_subset(subset_path)
    os.environ.pop("TRITON_PTXAS_PATH", None)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    results = prv_results
    prv_ids = [item['id'] for item in prv_results]
    correct = 0

    for idx, item in tqdm(enumerate(subset, start=1), total=len(subset) ):
        item_id = item.get('id')
        if item_id in prv_ids:
            continue

        question = item.get("question", "")
        choices = item.get("choice", [])
        answer = item.get("answer", "N/A")
        problem_id = item.get("id", f"sample_{idx}")
        audio_path = item.get("audio_path") or item.get("audio_url")

        if not audio_path:
            raise ValueError(f"No audio path found for subset item index {idx}")

        audio_data, sr = load_audio(audio_path)

        tool_output_path = os.path.join(
            tool_results_root,
            problem_id
        )
        if not load_tool_chain_results and overwrite_original_audio:
            raise ValueError("Cannot overwrite original audio if not loading tool chain results, because there will be no tool chain results to overwrite with. Please set --load_tool_chain_results if you want to overwrite original audio with tool chain results.")
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
            tool_audios_path = []
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

        audio_data = {"audio": [
                        (audio_data, sr)
                        ] + [(load_audio(p)) for p in tool_audios_path]}

        if overwrite_original_audio:
            audio_data["audio"] = audio_data["audio"][1]


        print('prompt:', prompt)
        outputs = llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": audio_data,
                }
            ],
            sampling_params=sampling_params,
            use_tqdm=False
        )

        pred = outputs[0].outputs[0].text.strip()
        try:
            is_correct = answer != "N/A" and answer.strip() in pred
        except:
            is_correct = False

        results.append(
            {
                "index": idx,
                'prompt': prompt,
                'load tool chain results': load_tool_chain_results,
                "id": problem_id,
                "question": question,
                "choices": choices,
                "answer": answer,
                "prediction": pred,
                "correct": is_correct,
            }
        )
        if is_correct:
            correct += 1

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

    # os.environ["HF_HOME"] = CACHE_DIR
    # os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
    # os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

    llm = LLM(
        model=MODEL,
        max_model_len=20000,
        max_num_seqs=32,
        limit_mm_per_prompt={"audio": 4},
        trust_remote_code=True,
        download_dir = CACHE_DIR,
        dtype=args.dtype,
        # enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
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
        tool_results_root=TOOL_RESULTS_PATH
    )


if __name__ == "__main__":
    main()