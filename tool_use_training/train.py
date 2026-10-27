"""Python equivalent of train.sh: calls ms-swift's `sft_main` directly instead
of shelling out to the `swift sft` CLI.

Must run under the `ms-swift` conda env (same requirement as
interface/engine.py's SwiftEngine).
"""

from __future__ import annotations
from argparse import Namespace, ArgumentParser
import os
import sys


def report_gpu_status() -> None:
    import torch

    if not torch.cuda.is_available():
        print("[GPU] CUDA not available -- training will run on CPU.")
        return

    print(f"[GPU] CUDA available, {torch.cuda.device_count()} device(s) visible.")
    # 每個 torchrun rank 只查自己的 LOCAL_RANK 那張卡，避免每個 process 都去碰所有 GPU
    # （在別的 rank 佔用的卡上建立 context 會出事）；查詢失敗也只印警告，不中斷訓練。
    i = int(os.environ.get("LOCAL_RANK", 0))
    try:
        name = torch.cuda.get_device_name(i)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        used_gb = (total_bytes - free_bytes) / 1024**3
        total_gb = total_bytes / 1024**3
        print(f"[GPU {i}] {name}: {used_gb:.2f} GiB used / {total_gb:.2f} GiB total")
    except Exception as e:
        print(f"[GPU {i}] WARNING: failed to query device ({e!r}); node={os.uname().nodename}")

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
        default="bfloat16",
        help="Torch data type for training (default: float16)",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=3,
        help="Batch size per GPU before gradient accumulation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,  # 目前 log 顯示 DeepSpeed config 用的是 8，先保留現況當預設值
        help="Number of steps to accumulate gradients before an optimizer step",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save a checkpoint every N optimizer steps",
    )

    return parser.parse_args()

def main() -> None:
    report_gpu_status()

    # ms-swift's real package lives at site-packages/swift; some envs also have
    # an unrelated `swift` package shadowing it via ~/.local/lib -- drop that.
    sys.path = [p for p in sys.path if ".local" not in p]
    from swift import SftArguments, sft_main  # noqa: E402

    args = parse_args()
    ckpt = args.resume_from_checkpoint
    adapters = [ckpt] if ckpt and ckpt.lower() != "none" else []
    sft_args = SftArguments(
        model=args.model,
        dataset=args.dataset,
        tuner_type="lora",
        output_dir=args.output_dir,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        # 沒給 checkpoint（或給 "none"）就從頭訓練，不要傳 [None] 進 adapters。
        adapters=adapters,
        #resume_from_checkpoint=args.resume_from_checkpoint,
        deepspeed="zero3",
        # 舊的 save_steps=500 大於整個 run 的步數（~315），導致從沒存過 checkpoint。
        save_steps=args.save_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    sft_main(sft_args)


if __name__ == "__main__":
    main()
