import argparse
import json
import os
from typing import List, Optional

import librosa
import numpy as np
from vllm import LLM, SamplingParams


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def load_audio(path: str, sr: int = 16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def build_prompt(question: str, choices: List[str], system_prompt: str) -> str:
    choice_text = "\n".join(choices) if choices else ""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        f"Question: {question}\n"
        f"Choices:\n{choice_text}\n"
        "Please answer with the best choice letter and the full choice text.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_subset(subset_path: str) -> List[dict]:
    with open(subset_path, "r", encoding="utf-8") as f:
        subset = json.load(f)
    if not isinstance(subset, list):
        raise ValueError("Subset file must contain a JSON list.")
    return subset


def evaluate(
    subset_path: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    output_predictions: Optional[str],
) -> None:
    subset = load_subset(subset_path)
    os.environ.pop("TRITON_PTXAS_PATH", None)

    llm = LLM(
        model=model,
        max_model_len=20000,
        max_num_seqs=1,
        limit_mm_per_prompt={"audio": 4},
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    results = []
    correct = 0

    for idx, item in enumerate(subset, start=1):
        question = item.get("question", "")
        choices = item.get("choice", [])
        answer = item.get("answer", "N/A")
        audio_path = item.get("audio_path") or item.get("audio_url")

        if not audio_path:
            raise ValueError(f"No audio path found for subset item index {idx}")

        audio_data, sr = load_audio(audio_path)
        prompt = build_prompt(question, choices, system_prompt)

        outputs = llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"audio": [(audio_data, sr)]},
                }
            ],
            sampling_params=sampling_params,
        )

        pred = outputs[0].outputs[0].text.strip()
        is_correct = answer != "N/A" and answer.strip() in pred

        results.append(
            {
                "index": idx,
                "question": question,
                "choices": choices,
                "answer": answer,
                "prediction": pred,
                "correct": is_correct,
            }
        )
        if is_correct:
            correct += 1

        print(f"=== Sample {idx} ===")
        print(f"Question: {question}")
        print("Choices:")
        for choice in choices:
            print(choice)
        print(f"Ground truth: {answer}")
        print("Prediction:")
        print(pred)
        print(f"Correct: {is_correct}\n")

    accuracy = correct / len(results) if results else 0.0
    print("=== Evaluation Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    if output_predictions:
        with open(output_predictions, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed predictions to: {output_predictions}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen 2.5 on a sampled DCASE AudioQA subset."
    )
    parser.add_argument(
        "--subset",
        default="dcase_subset.json",
        help="Path to the sampled subset JSON file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Omni-7B",
        help="LLM model identifier for evaluation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--output-predictions",
        default=None,
        help="Optional path to save detailed predictions as JSON.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to provide to the model.",
    )
    args = parser.parse_args()

    evaluate(
        subset_path=os.path.abspath(args.subset),
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        output_predictions=args.output_predictions,
    )


if __name__ == "__main__":
    main()
