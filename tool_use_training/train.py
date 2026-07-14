"""Python equivalent of train.sh: calls ms-swift's `sft_main` directly instead
of shelling out to the `swift sft` CLI.

Must run under the `ms-swift` conda env (same requirement as
tool_use_benchmark/model_engine.py).
"""

from __future__ import annotations

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


def main() -> None:
    report_gpu_status()

    # ms-swift's real package lives at site-packages/swift; some envs also have
    # an unrelated `swift` package shadowing it via ~/.local/lib -- drop that.
    sys.path = [p for p in sys.path if ".local" not in p]
    from swift import SftArguments, sft_main  # noqa: E402

    args = SftArguments(
        model="Qwen/Qwen2.5-Omni-7B",
        dataset=["./train.jsonl"],
        tuner_type="lora",
        output_dir="output",
        max_length=4096,
        torch_dtype="float16",
    )
    sft_main(args)


if __name__ == "__main__":
    main()
