import argparse
import glob
import json
import os
import random
from typing import List


def resolve_audio_path(item: dict, validation_json_dir: str, root: str) -> str:
    audio_url = item.get("audio_url", "")
    audio_path = os.path.normpath(os.path.join(validation_json_dir, audio_url))
    if os.path.exists(audio_path):
        return audio_path
    fallback_path = os.path.normpath(os.path.join(root, audio_url))
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Audio file not found for URL: {audio_url}")

def load_audio_test(audio_path: str):
    import librosa
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        return True
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return False

def sample_subset(
    root: str,
    validation_dir: str,
    output_path: str,
    n: int,
    seed: int,
) -> None:
    validation_json_dir = os.path.join(root, validation_dir)
    question_files = sorted(glob.glob(os.path.join(validation_json_dir, "*.json")))
    if not question_files:
        raise FileNotFoundError(
            f"No JSON files found in validation directory: {validation_json_dir}"
        )

    random.seed(seed)
    selected_files = random.sample(question_files, min(n, len(question_files)))

    subset: List[dict] = []
    for json_path in selected_files:
        with open(json_path, "r", encoding="utf-8") as f:
            item = json.load(f)
        item["_json_path"] = os.path.abspath(json_path)
        try:
            item["audio_path"] = resolve_audio_path(item, validation_json_dir, root)
            if load_audio_test(item["audio_path"]):
                subset.append(item)

        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping item from file: {json_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print(f"Saved subset of {len(subset)} problems to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a subset of DCASE AudioQA validation problems and save them locally."
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5,
        help="Number of validation problems to sample.",
    )
    parser.add_argument(
        "--root",
        default="/work/u1501463/2025_DCASE_AudioQA_Official",
        help="Root path for the DCASE AudioQA dataset.",
    )
    parser.add_argument(
        "--validation-subdir",
        default="dcase_2025_question_path/train",
        help="Relative directory containing validation JSON files.",
    )
    parser.add_argument(
        "--output",
        default="dcase_subset.json",
        help="Output file path for the sampled subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    sample_subset(
        root=args.root,
        validation_dir=args.validation_subdir,
        output_path=os.path.abspath(args.output),
        n=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
