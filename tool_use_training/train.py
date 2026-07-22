"""Python equivalent of train.sh: calls ms-swift's `sft_main` directly instead
of shelling out to the `swift sft` CLI.

Must run under the `ms-swift` conda env (same requirement as
interface/engine.py's SwiftEngine).
"""

from __future__ import annotations
from argparse import Namespace, ArgumentParser
import sys


def report_gpu_status() -> None:
    import torch

    if not torch.cuda.is_available():
        print("[GPU] CUDA not available -- training will run on CPU.")
        return

    print(f"[GPU] CUDA available, {torch.cuda.device_count()} device(s) visible.")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        used_gb = (total_bytes - free_bytes) / 1024**3
        total_gb = total_bytes / 1024**3
        print(f"[GPU {i}] {name}: {used_gb:.2f} GiB used / {total_gb:.2f} GiB total")

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a model using ms-swift's SFT.")
    parser.add_argument(
        "--model",
        type=str,
        default="/work/u1501463/hf_cache/",
        help="Model name or path (default: Qwen/Qwen2.5-Omni-7B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["./train.jsonl"],
        help="Path(s) to dataset file(s) (default: ./train.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the trained model (default: output)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20000,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="Torch data type for training (default: float16)",
    )
    return parser.parse_args()

def main() -> None:
    report_gpu_status()

    # ms-swift's real package lives at site-packages/swift; some envs also have
    # an unrelated `swift` package shadowing it via ~/.local/lib -- drop that.
    sys.path = [p for p in sys.path if ".local" not in p]
    from swift import SftArguments, sft_main  # noqa: E402

    args = parse_args()
    args = SftArguments(
        model=args.model,
        dataset=args.dataset,
        tuner_type="lora",
        output_dir=args.output_dir,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
    )
    sft_main(args)


if __name__ == "__main__":
    main()
