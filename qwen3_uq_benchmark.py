"""Qwen3-Omni UQ benchmark on an (original, tool-applied) item set: does the tool help?

Runs Qwen3-Omni on the same items twice -- without the tool (original audio) and with it
(tool-applied audio) -- in lockstep, computing the same UQ metrics as the Qwen2.5-Omni runs
(predictive / length-normalized / discrete-semantic entropy over K samples, P(True)), then
summarises accuracy, accuracy gain, the T->T/T->F/F->T/F->F confusion, and the per-side means
and mean change of every UQ metric, overall and per applied tool.

Input: a dataset dir in gen_2nd_2_data/build_dataset.py's layout, typically one exported by
`uq_threshold_confusion.py --export-dir` (original_subset.json, tool_applied_subset.json,
and optionally manifest.json for the per-tool breakdown and filter.json, whose recorded
`export_dir` lets the audio paths be re-rooted when the dir was copied to another machine).

Output (in --output_dir), same schema and naming as the Qwen2.5-Omni UQ results, so
uq_threshold_confusion.py can analyse them too:
    <tag>_original_uq_results_no_tool_chain_results.json
    <tag>_tool_applied_uq_results_no_tool_chain_results.json
    <tag>_summary.json
where <tag> = --tag (default: the model dir/name's basename). Progress is appended after
every batch (`.partial.jsonl`), so the run can be interrupted and resumed (--no-cont restarts).

Usage (see run_qwen3_uq_benchmark.sh):
    python qwen3_uq_benchmark.py --subset_dir exports/p_true_delta_ge0.1 --output_dir predictions/qwen3_uq
    python qwen3_uq_benchmark.py --subset_dir ... --output_dir ... --summary_only   # re-summarise, no model
"""
from __future__ import annotations

import collections
import dataclasses
import json
import math
import os
from argparse import ArgumentParser, BooleanOptionalAction
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

os.environ.pop("TRITON_PTXAS_PATH", None)

from tqdm import tqdm

# Shared with the Qwen2.5-Omni UQ run: audio loading, resumable result files.
from qwen25_with_tool_chain_evaluation import (
    append_partial,
    load_audio,
    load_previous_results,
    load_subset,
    partial_path,
    save_results_atomic,
)
# The project's Qwen3-Omni prompt (audio token <|audio_start|><|audio_pad|><|audio_end|>).
from qwen3_with_tool_chain_evaluation import build_standard_prompt
from uncertainty_quantification_tools import QWEN3_OMNI_AUDIO_TOKEN, compute_uncertainty_batch

SIDES = ("original", "tool_applied")
METRICS = ["p_true", "predictive_entropy", "length_normalized_entropy", "discrete_semantic_entropy", "semantic_entropy"]
RESULT_SUFFIX = "_uq_results_no_tool_chain_results.json"
DEFAULT_SYSTEM_PROMPT = (  # same as qwen3_with_tool_chain_evaluation.py
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def parse_arguments():
    p = ArgumentParser(description="Qwen3-Omni UQ benchmark: original vs tool-applied audio of the same items.")
    p.add_argument("--subset_dir", type=Path, help="dir with original_subset.json + tool_applied_subset.json "
                                                   "(e.g. from uq_threshold_confusion.py --export-dir)")
    p.add_argument("--original_subset", type=Path, help="override: original subset json")
    p.add_argument("--tool_applied_subset", type=Path, help="override: tool-applied subset json")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="default: $REMOTE_QWEN3_OUTPUT_ROOT/<subset_dir name> (cluster_paths.sh)")
    p.add_argument("--tag", default=None, help="result-file prefix (default: model basename)")
    p.add_argument("--path_remap", nargs=2, action="append", metavar=("OLD_PREFIX", "NEW_PREFIX"), default=[],
                   help="rewrite audio-path prefixes (repeatable); filter.json's export_dir is remapped automatically")
    p.add_argument("--summary_only", action="store_true", help="only (re)compute the summary from existing results")
    p.add_argument("--cont", "--resume", dest="cont", action=BooleanOptionalAction, default=True,
                   help="resume from existing results (default on; --no-cont starts over)")

    # Model / vLLM (defaults follow qwen3_with_tool_chain_evaluation.py).
    p.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--dtype", default="float16", help="float16 for V100 (no bf16); bfloat16 on newer GPUs")
    p.add_argument("--max_model_len", type=int, default=20000)
    p.add_argument("--max_num_seqs", type=int, default=32)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--mm_processor_cache_gb", type=float, default=0,
                   help="vLLM multimodal preprocessor cache; 0 = off (each clip is seen once, the cache only costs RAM)")
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--download_dir", default=os.environ.get("REMOTE_HF_CACHE") or os.environ.get("HF_HUB_CACHE")
                   or "/work/u1501463/hf_cache", help="model download cache (default: cluster_paths.sh's REMOTE_HF_CACHE)")
    p.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16, help="items per llm.generate call; progress saved per batch")

    # UQ (same switches as qwen25_with_tool_chain_evaluation.py).
    p.add_argument("--uq_num_samples", type=int, default=10)
    p.add_argument("--uq_sample_temperature", type=float, default=1.0)
    p.add_argument("--uq_semantic_entropy", action=BooleanOptionalAction, default=False)
    p.add_argument("--uq_p_true", action=BooleanOptionalAction, default=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def subset_paths(args):
    o = args.original_subset or (args.subset_dir / "original_subset.json" if args.subset_dir else None)
    t = args.tool_applied_subset or (args.subset_dir / "tool_applied_subset.json" if args.subset_dir else None)
    if o is None or t is None:
        raise SystemExit("Pass --subset_dir, or both --original_subset and --tool_applied_subset.")
    return Path(o), Path(t)


def path_remaps(args) -> List[tuple]:
    remaps = [tuple(r) for r in args.path_remap]
    if args.subset_dir and (args.subset_dir / "filter.json").exists():
        exported_at = json.loads((args.subset_dir / "filter.json").read_text(encoding="utf-8")).get("export_dir")
        here = str(args.subset_dir.expanduser().resolve())
        if exported_at and exported_at != here:
            remaps.append((exported_at, here))
    return remaps


def remap(path: str, remaps) -> str:
    for old, new in remaps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_tools(args) -> Dict[str, str]:
    manifest = args.subset_dir / "manifest.json" if args.subset_dir else None
    if not manifest or not manifest.exists():
        return {}
    return {r["id"]: "+".join(s["tool"] for s in r["tool_chain"]) or "(none)"
            for r in json.loads(manifest.read_text(encoding="utf-8"))}


def load_reference(args) -> Dict[str, dict]:
    """Per-item values from the run the set was filtered on (filter.json), for comparison."""
    f = args.subset_dir / "filter.json" if args.subset_dir else None
    if not f or not f.exists():
        return {}
    return {it["id"]: it for it in json.loads(f.read_text(encoding="utf-8")).get("items", [])}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_llm(args):
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
        download_dir=args.download_dir,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )
    # Only on vLLM versions that have it (the cluster's may differ from the local one).
    if "mm_processor_cache_gb" in {f.name for f in dataclasses.fields(EngineArgs)}:
        kwargs["mm_processor_cache_gb"] = args.mm_processor_cache_gb
    return LLM(**kwargs)


def run(args, llm, subsets: Dict[str, List[dict]], out_paths: Dict[str, Path], remaps) -> None:
    from vllm import SamplingParams

    greedy = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    done = {}
    for side in SIDES:
        prev = load_previous_results(str(out_paths[side])) if args.cont else []
        done[side] = {r["id"] for r in prev}
        if prev:
            print(f"Resuming {side}: {len(prev)} item(s) done")

    # Lockstep: item i on both sides before item i+1, so the two always cover the same items.
    tasks = [(side, idx, subsets[side][idx - 1])
             for idx in range(1, max(len(s) for s in subsets.values()) + 1)
             for side in SIDES
             if idx <= len(subsets[side]) and subsets[side][idx - 1]["id"] not in done[side]]
    batches = [tasks[i:i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
    total = sum(len(s) for s in subsets.values())

    def prepare(task):
        side, idx, item = task
        audio_path = remap(item.get("audio_path") or item.get("audio_url"), remaps)
        audio, sr = load_audio(audio_path)
        return {
            "side": side, "idx": idx, "id": item["id"],
            "question": item.get("question", ""), "choices": item.get("choice", []),
            "answer": item.get("answer", "N/A"),
            "prompt": build_standard_prompt(question=item.get("question", ""), choices=item.get("choice", []),
                                            system_prompt=args.system_prompt),
            "multi_modal_data": {"audio": [(audio, sr)]},
        }

    needs_samples = args.uq_num_samples > 0
    with ThreadPoolExecutor(max_workers=4) as pool, tqdm(total=total, initial=total - len(tasks)) as bar:
        future = pool.map(prepare, batches[0]) if batches else None
        for bi in range(len(batches)):
            batch = list(future)
            future = pool.map(prepare, batches[bi + 1]) if bi + 1 < len(batches) else None
            outputs = llm.generate([{"prompt": b["prompt"], "multi_modal_data": b["multi_modal_data"]} for b in batch],
                                   sampling_params=greedy, use_tqdm=False)
            for b, out in zip(batch, outputs):
                b["greedy_prediction"] = out.outputs[0].text.strip()
            uq = compute_uncertainty_batch(
                llm, batch, system_prompt=args.system_prompt,
                num_samples=args.uq_num_samples if needs_samples else 0,
                sample_temperature=args.uq_sample_temperature, max_tokens=args.max_tokens,
                compute_semantic=args.uq_semantic_entropy, compute_p_true=args.uq_p_true,
                audio_token=QWEN3_OMNI_AUDIO_TOKEN,
            )
            for side in SIDES:
                entries = [{
                    "index": b["idx"],
                    "prompt": b["prompt"],
                    "load tool chain results": False,
                    "id": b["id"],
                    "question": b["question"],
                    "choices": b["choices"],
                    "answer": b["answer"],
                    "prediction": b["greedy_prediction"],
                    "correct": b["answer"] != "N/A" and b["answer"].strip() in b["greedy_prediction"],
                    "uncertainty": u,
                } for b, u in zip(batch, uq) if b["side"] == side]
                if entries:
                    append_partial(entries, str(out_paths[side]))
            bar.update(len(batch))

    # Merge each side's progress log into its final results file (subset order).
    for side in SIDES:
        results = {r["id"]: r for r in load_previous_results(str(out_paths[side]))}
        save_results_atomic([results[i["id"]] for i in subsets[side] if i["id"] in results], str(out_paths[side]))
        if os.path.exists(partial_path(str(out_paths[side]))):
            os.remove(partial_path(str(out_paths[side])))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


def summarize_group(pairs) -> dict:
    """pairs: list of (original_entry, tool_applied_entry)."""
    n = len(pairs)
    conf = collections.Counter(("T" if o["correct"] else "F") + "->" + ("T" if t["correct"] else "F") for o, t in pairs)
    acc_o = sum(o["correct"] for o, _ in pairs) / n if n else None
    acc_t = sum(t["correct"] for _, t in pairs) / n if n else None
    metrics = {}
    for m in METRICS:
        vo = [(o.get("uncertainty") or {}).get(m) for o, _ in pairs]
        vt = [(t.get("uncertainty") or {}).get(m) for _, t in pairs]
        if all(v is None for v in vo + vt):
            continue
        metrics[m] = {
            "mean_original": _mean(vo),
            "mean_tool_applied": _mean(vt),
            "mean_delta": _mean([b - a for a, b in zip(vo, vt) if a is not None and b is not None]),
        }
    return {
        "n": n,
        "accuracy_original": acc_o,
        "accuracy_tool_applied": acc_t,
        "accuracy_gain": (acc_t - acc_o) if n else None,
        "confusion": {k: conf[k] for k in ("T->T", "T->F", "F->T", "F->F")},
        "net_correct_gain": conf["F->T"] - conf["T->F"],
        "uq": metrics,
    }


def summarize(args, out_paths: Dict[str, Path], tag: str) -> dict:
    res = {side: {r["id"]: r for r in load_previous_results(str(out_paths[side]))} for side in SIDES}
    ids = [i for i in res["original"] if i in res["tool_applied"]]
    pairs = {i: (res["original"][i], res["tool_applied"][i]) for i in ids}
    tools = load_tools(args)
    ref = load_reference(args)

    summary = {"model": args.model, "tag": tag, "subset_dir": str(args.subset_dir) if args.subset_dir else None,
               "overall": summarize_group(list(pairs.values()))}
    if tools:
        by_tool = collections.defaultdict(list)
        for i, pair in pairs.items():
            by_tool[tools.get(i, "(unknown)")].append(pair)
        summary["per_tool"] = {t: summarize_group(p) for t, p in sorted(by_tool.items(), key=lambda kv: -len(kv[1]))}
    if ref:
        common = [i for i in ids if i in ref]
        if common:
            summary["reference_run"] = {  # the model the set was filtered on, same items
                "n": len(common),
                "accuracy_original": sum(ref[i]["correct_original"] for i in common) / len(common),
                "accuracy_tool_applied": sum(ref[i]["correct_tool_applied"] for i in common) / len(common),
            }

    out = args.output_dir / f"{tag}_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    o = summary["overall"]
    pct = lambda v: "n/a" if v is None else f"{v:.2%}"
    print(f"\n=== {tag}: original (no tool) vs tool-applied, {o['n']} paired item(s) ===")
    print(f"accuracy        : {pct(o['accuracy_original'])} -> {pct(o['accuracy_tool_applied'])}"
          f"   gain {o['accuracy_gain']:+.2%}" if o["n"] else "no paired results yet")
    if o["n"]:
        c = o["confusion"]
        print(f"confusion       : T->T {c['T->T']}  T->F {c['T->F']}  F->T {c['F->T']}  F->F {c['F->F']}"
              f"   net {o['net_correct_gain']:+d}")
        for m, v in o["uq"].items():
            fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
            print(f"{m:26}: {fmt(v['mean_original'])} -> {fmt(v['mean_tool_applied'])}   mean Δ {fmt(v['mean_delta'])}")
    if "reference_run" in summary:
        r = summary["reference_run"]
        print(f"reference run   : {pct(r['accuracy_original'])} -> {pct(r['accuracy_tool_applied'])} on the same {r['n']} items")
    if "per_tool" in summary:
        print(f"\n{'tool':24}{'n':>6}{'acc orig':>10}{'acc tool':>10}{'gain':>9}{'net':>6}")
        for t, g in summary["per_tool"].items():
            print(f"{t:24}{g['n']:>6}{pct(g['accuracy_original']):>10}{pct(g['accuracy_tool_applied']):>10}"
                  f"{g['accuracy_gain']:>+9.1%}{g['net_correct_gain']:>+6d}")
    print(f"\nsummary: {out}")
    return summary


def main():
    args = parse_arguments()
    if args.output_dir is None:
        root = os.environ.get("REMOTE_QWEN3_OUTPUT_ROOT", "/work/u1501463/LALM_UQ/qwen3_uq")
        if not args.subset_dir:
            raise SystemExit("Pass --output_dir (no --subset_dir to name the default one after).")
        args.output_dir = Path(root) / args.subset_dir.expanduser().resolve().name
    tag = args.tag or Path(args.model.rstrip("/")).name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {side: args.output_dir / f"{tag}_{side}{RESULT_SUFFIX}" for side in SIDES}

    if not args.summary_only:
        orig_path, tool_path = subset_paths(args)
        subsets = {"original": load_subset(str(orig_path)), "tool_applied": load_subset(str(tool_path))}
        if [i["id"] for i in subsets["original"]] != [i["id"] for i in subsets["tool_applied"]]:
            raise SystemExit("original and tool-applied subsets must list the same ids in the same order")
        llm = build_llm(args)
        run(args, llm, subsets, out_paths, path_remaps(args))
    summarize(args, out_paths, tag)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; rerun the same command to resume.")
        raise SystemExit(130)
