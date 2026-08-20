"""Python equivalent of train.sh: calls ms-swift's sft_main directly instead
of shelling out to the swift sft CLI.
Must run under the ms-swift conda env (same requirement as
tool_use_benchmark/model_engine.py).
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
        help="Model name or path (e.g. Qwen/Qwen3-Omni-30B-A3B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
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
        default=15000,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="Torch data type for training (default: float16)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="none",
        help="LoRA checkpoint dir to resume from, or 'none' (default) to train fresh from --model.",
    )
    return parser.parse_args()
    return parser.parse_args()
def main() -> None:
    report_gpu_status()
    # ms-swift's real package lives at site-packages/swift; some envs also have
    # an unrelated swift package shadowing it via ~/.local/lib -- drop that.
    sys.path = [p for p in sys.path if ".local" not in p]
    from swift import SftArguments, sft_main  # noqa: E402
    args = parse_args()
    sft_kwargs = dict(
        model=args.model,
        template="qwen3_omni",
        dataset=args.dataset,
        tuner_type="lora",
        lora_rank=32,
        lora_alpha=32,
        output_dir=args.output_dir,
        max_length=args.max_length,
        save_steps=100,
        per_device_train_batch_size=4,
        #modules_to_save=["lm_head"],
        #target_modules=["all-linear", "lm_head"],
        loss_scale="hermes",
        # 30B-A3B's frozen bf16 base weights are ~70GB -- doesn't fit
        # comfortably on one 80GB GPU alongside activations, so ZeRO-3
        # shards them across the ranks torchrun launches (see
        # submit_python.sh). Not needed (and unused) for a single-GPU run.
        # CAVEAT (unverified on this checkpoint): ms-swift users have
        # reported ZeRO-3 + LoRA breaking gradient flow on Qwen3-MoE
        # architectures (github.com/modelscope/ms-swift issues around
        # Qwen3-30B-A3B), with ZeRO-2 (or zero2_offload) suggested as the
        # workaround -- but ZeRO-2 doesn't shard parameters, so it likely
        # won't fit this model on 2x80GB. This is exactly what the small
        # smoke-test run should surface before committing a long job to it;
        # if it breaks, the fallback that avoids the ZeRO-3/LoRA/MoE
        # interaction entirely is QLoRA (quant_method="bnb", quant_bits=4),
        # which shrinks the frozen base to ~17.5GB and fits under plain
        # ZeRO-2/DDP.
        deepspeed="zero3",
    )
    # "none" (the --resume_from_checkpoint default) means train fresh from
    # --model with no prior adapter -- omit the key entirely rather than
    # passing "none" through to ms-swift, which would try to load it as a
    # literal path.
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "none":
        sft_kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint
        sft_kwargs["resume_only_model"] = True
    args = SftArguments(**sft_kwargs)
    sft_main(args)
if __name__ == "__main__":
    main()
