"""Build stage-2 tool-use training data via a second generation strategy: sample real
audio (optionally with attached QA), apply a randomly drawn tool + parameters straight
from `tools/` to it, and emit an (original, tool_applied) pair.

This is deliberately a *different* strategy from `gen_2nd_stage_data/build_by_disturb.py`:

- gen_2nd_stage_data first *disturbs* a clip with a synthetic `audio_edit/` augmentation
  (add_noise, change_pitch, ...), then looks up a hand-paired *recovery* tool call that
  best-effort undoes that specific disturbance (`disturb_recover.py`'s fixed disturb-op ->
  recovery-tool table). The point is teaching "the tool call that fixes this exact kind of
  corruption."
- This script (gen_2nd_2_stage_data) never disturbs anything. It samples audio straight
  from a source dataset and applies a tool drawn (name + parameters) directly at random
  from a config file (`tool_config.json`), with no notion of "undoing" anything -- there is
  no forward/reverse pairing, just "here is audio, here is what one of the available tools
  does to it." That json file is the single control point for (a) which tools are in play
  and their relative sampling weight, and (b) each tool's parameter ranges (e.g.
  time_stretch's rate choices, pitch_shift's n_steps choices, denoise's noise_factor
  range) -- see tool_config.json's own top-of-file comment for the spec format, and
  `tool_appliers.py` for how each tool actually gets called.

Pipeline
--------
0. `tool_config.json` (or whatever `--tool-config` points at) declares the tool set.
1. Sample n items from one or more sources: `--input-json` (dcase_subset.json-shaped QA
   lists -- this is how a "dcase 2025" corpus, or any other pre-existing QA json, feeds
   in), and/or `--sources audioset|vctk` (raw directory pools, no QA attached) and/or
   arbitrary `--audio-glob` patterns for "or other" datasets. Items without QA fields get
   sensible defaults (question="", answer="N/A", ...), same convention as
   `build_by_disturb.py`'s `to_dcase_entry`.
2. For each item, draw k in [min-tools, max-tools] distinct tool names (weighted by
   tool_config.json's `weight`) and their parameters, and apply them in sequence.
3. Mirroring build_by_disturb.py's output shape: write `original/` (a WAV copy of the
   untouched source) and `tool_applied/` (the post-tool-chain audio) under --output-dir,
   plus `original_subset.json` / `tool_applied_subset.json` (dcase_subset.json-shaped,
   pointing at those two audio sets) -- both directly usable as `--subset_path` for
   `qwen25_with_tool_chain_evaluation.py`, i.e. as the input `run_UQ_for_stage2.sh` expects
   (repoint that script's `disturbed_path`/`recovered_path` at these two files, or copy it
   and rename the variables).

Usage:
    python build_dataset.py \\
        --input-json /home/u1501463/tool_use_LALM/dcase_subset.json \\
        --num-samples 200 \\
        --output-dir /work/u1501463/gen_2nd_2_stage_tool_applied \\
        --seed 0

    # Or draw straight from raw audio pools instead of a QA json:
    python build_dataset.py --sources audioset vctk --num-samples 200 ...

Run under an env with librosa/soundfile/scipy (e.g. the same `Whisper`/`ms-swift` env
`gen_2nd_stage_data`'s scripts need) for the light tools this script calls in-process;
heavy ML tools (remove_target/extract_target/human_voice_enhance/super_resolution) are
disabled by default in tool_config.json and, if enabled, are dispatched to their own conda
env per-call (see tool_appliers.py) so they don't further constrain this script's own env.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import random
import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import tool_appliers as ta  # noqa: E402
from tools import heavy_tool_pool  # noqa: E402

DEFAULT_AUDIOSET_DIR = Path(os.environ.get("AUDIOSET_AUDIO_DIR", "/work/u1501463/audioset_20k/20k/train"))
DEFAULT_VCTK_DIR = Path(os.environ.get("VCTK_AUDIO_DIR", "/work/u1501463/VCTK/wav48_silence_trimmed"))


# ---------------------------------------------------------------------------
# Step 1: sample input items (QA-shaped and/or raw audio pools).
# ---------------------------------------------------------------------------

def load_qa_items(paths: List[Path]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise SystemExit(f"{path} must contain a JSON list of QA-pair dicts.")
        items.extend(data)
    return items


def collect_raw_audio_items(sources: List[str], audio_globs: List[str]) -> List[Dict[str, Any]]:
    files: List[Path] = []
    if "audioset" in sources:
        files.extend(sorted(DEFAULT_AUDIOSET_DIR.glob("*.wav")))
    if "vctk" in sources:
        files.extend(sorted(DEFAULT_VCTK_DIR.glob("*/*.flac")))
    for pattern in audio_globs:
        files.extend(sorted(Path(p) for p in glob_module.glob(pattern, recursive=True)))
    return [{"audio_path": str(f)} for f in files]


def get_audio_path(item: Dict[str, Any]) -> str:
    audio_path = item.get("audio_path") or item.get("audio_url")
    if not audio_path:
        raise ValueError(f"Item {item.get('id')!r} has no audio_path/audio_url.")
    return str(Path(audio_path).expanduser().resolve())


def to_dcase_entry(item: Dict[str, Any], audio_path: str) -> Dict[str, Any]:
    """Project an (optionally QA-less) item + a (possibly new) audio path into
    dcase_subset.json's shape -- same convention as build_by_disturb.py's helper of the
    same name, so items with no attached QA (e.g. raw audioset/vctk pool draws) still get
    a well-formed, evaluator-shaped entry."""
    return {
        "question": item.get("question", ""),
        "choice": item.get("choice", []),
        "answer": item.get("answer", "N/A"),
        "id": item["id"],
        "audio_url": audio_path,
        "question_type": item.get("question_type"),
        "_json_path": item.get("_json_path"),
        "audio_path": audio_path,
    }


def sample_items(
    items: List[Dict[str, Any]],
    num_samples: int,
    seed: int,
    dedupe_by_audio: bool,
) -> List[Dict[str, Any]]:
    if dedupe_by_audio:
        seen = set()
        deduped = []
        for item in items:
            key = item.get("audio_path") or item.get("audio_url")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        items = deduped

    rng = random.Random(seed)
    k = min(num_samples, len(items))
    return rng.sample(items, k)


# ---------------------------------------------------------------------------
# Step 2: pick + apply a random tool chain.
# ---------------------------------------------------------------------------

def weighted_sample_without_replacement(
    names: List[str], weights: List[float], k: int, rng: random.Random
) -> List[str]:
    pool_names = list(names)
    pool_weights = list(weights)
    chosen: List[str] = []
    for _ in range(k):
        total = sum(pool_weights)
        r = rng.uniform(0.0, total)
        upto = 0.0
        for i, w in enumerate(pool_weights):
            upto += w
            if upto >= r:
                chosen.append(pool_names.pop(i))
                pool_weights.pop(i)
                break
    return chosen


def draw_tool_names(
    tool_names: List[str],
    tool_weights_list: List[float],
    k: int,
    rng: random.Random,
    tool_config: Dict[str, Any],
) -> List[str]:
    """k distinct tools. Plain weighted draw by default. If tool_config sets
    `heavy_tool_probability` p, each pick first goes to the heavy (ML, own-env) group
    with probability p -- else the light group -- then is weighted within that group,
    so the ML share stays p however many ML tools are enabled/available."""
    p = tool_config.get("heavy_tool_probability")
    if p is None:
        return weighted_sample_without_replacement(tool_names, tool_weights_list, k, rng)
    tools_cfg = tool_config["tools"]
    pool = list(zip(tool_names, tool_weights_list))
    chosen: List[str] = []
    for _ in range(k):
        heavy = [(n, w) for n, w in pool if tools_cfg[n].get("env")]
        light = [(n, w) for n, w in pool if not tools_cfg[n].get("env")]
        group = heavy if heavy and (not light or rng.random() < p) else light
        name = weighted_sample_without_replacement([n for n, _ in group], [w for _, w in group], 1, rng)[0]
        chosen.append(name)
        pool = [(n, w) for n, w in pool if n != name]
    return chosen


def apply_tool_chain(
    source_wav: Path,
    work_dir: Path,
    tool_names: List[str],
    tool_weights_list: List[float],
    tool_config: Dict[str, Any],
    min_tools: int,
    max_tools: int,
    rng: random.Random,
) -> Dict[str, Any]:
    k = rng.randint(min_tools, max_tools)
    k = max(0, min(k, len(tool_names)))
    chosen = draw_tool_names(tool_names, tool_weights_list, k, rng, tool_config)

    current_path = source_wav
    chain: List[Dict[str, Any]] = []
    for step_index, name in enumerate(chosen, start=1):
        tool_cfg = tool_config["tools"][name]
        duration = ta.get_duration_seconds(current_path)
        out_path = work_dir / f"step{step_index}_{name}.wav"
        params, current_path = ta.APPLIERS[name](tool_cfg, current_path, out_path, rng, duration)
        chain.append({
            "tool": name,
            "parameters": {k2: v for k2, v in params.items() if k2 != "audio_path"},
            "output_path": str(current_path),
        })

    return {"chain": chain, "final_path": current_path}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_one_sample(
    item: Dict[str, Any],
    output_dir: Path,
    tool_names: List[str],
    tool_weights_list: List[float],
    tool_config: Dict[str, Any],
    min_tools: int,
    max_tools: int,
    rng: random.Random,
) -> Dict[str, Any]:
    sample_id = str(item["id"])
    source = Path(get_audio_path(item))

    original_dir = output_dir / "original" / sample_id
    tool_applied_dir = output_dir / "tool_applied" / sample_id
    original_dir.mkdir(parents=True, exist_ok=True)
    tool_applied_dir.mkdir(parents=True, exist_ok=True)

    working_wav = ta.ensure_wav(source, tool_applied_dir)
    original_wav = original_dir / "original.wav"
    if working_wav.resolve() != original_wav.resolve():
        shutil.copy(str(working_wav), str(original_wav))

    result = apply_tool_chain(
        working_wav, tool_applied_dir, tool_names, tool_weights_list, tool_config, min_tools, max_tools, rng
    )
    chain = result["chain"]
    final_path = result["final_path"]

    if chain:
        tool_applied_path = tool_applied_dir / "tool_applied.wav"
        if final_path.resolve() != tool_applied_path.resolve():
            # shutil.move, not Path.replace/os.rename: see tool_appliers._finalize's
            # docstring -- a chain step's output can land outside tool_applied_dir's
            # filesystem (e.g. clipping.py always writes next to its input).
            shutil.move(str(final_path), str(tool_applied_path))
            chain[-1]["output_path"] = str(tool_applied_path)
    else:
        # k == 0: nothing applied -- tool_applied aliases original directly rather than
        # making a pointless copy.
        tool_applied_path = original_wav

    return {
        "id": sample_id,
        "source_audio": str(source),
        "original_audio": str(original_wav),
        "tool_chain": chain,
        "tool_applied_audio": str(tool_applied_path),
    }


# ---------------------------------------------------------------------------
# Parallel build (--workers > 1)
# ---------------------------------------------------------------------------

_WORKER_CTX: Dict[str, Any] = {}


def _init_worker(ctx: Dict[str, Any]) -> None:
    _WORKER_CTX.update(ctx)


def _build_task(task):
    idx, item = task
    c = _WORKER_CTX
    rng = random.Random(f"{c['seed']}:{idx}:{item['id']}")
    errors = []
    for attempt in range(c["max_attempts"]):
        try:
            record = build_one_sample(
                item, c["output_dir"], c["tool_names"], c["tool_weights"], c["tool_config"],
                c["min_tools"], c["max_tools"], rng,
            )
            return idx, record, None
        except Exception:
            errors.append(f"Attempt {attempt + 1} failed for {item['id']}:\n{traceback.format_exc()}")
    return idx, None, "\n".join(errors)


def _peek_tools(task, ctx) -> List[str]:
    """The tool names `_build_task` will draw for this sample (same RNG, same first draws
    as apply_tool_chain), without doing any audio work."""
    return _peek(task, ctx)[0]


def _peek(task, ctx):
    """(tool names, first heavy tool's drawn params or None) for this sample. The params
    are only predictable when the chain's first tool is heavy (heavy appliers draw their
    params first; light ones consume the RNG in data-dependent ways)."""
    idx, item = task
    rng = random.Random(f"{ctx['seed']}:{idx}:{item['id']}")
    k = rng.randint(ctx["min_tools"], ctx["max_tools"])
    k = max(0, min(k, len(ctx["tool_names"])))
    names = draw_tool_names(ctx["tool_names"], ctx["tool_weights"], k, rng, ctx["tool_config"])
    params = None
    if names and ctx["tool_config"]["tools"][names[0]].get("env"):
        params = ta.draw_params(ctx["tool_config"]["tools"][names[0]], rng)
    return names, params


def _run_phase(tasks, ctx, args, sampled, done, progress_file) -> None:
    import multiprocessing as mp

    if not tasks:
        return
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(ctx,))
    try:
        with progress_file.open("a", encoding="utf-8") as out, tqdm(total=len(tasks)) as bar:
            for n, (idx, record, err) in enumerate(pool.imap_unordered(_build_task, tasks, chunksize=4), 1):
                if err:
                    print(err, file=sys.stderr)
                    print(f"Giving up on {sampled[idx - 1]['id']} after {args.max_attempts_per_sample} attempts.",
                          file=sys.stderr)
                done[idx] = record
                out.write(json.dumps({"idx": idx, "record": record}, ensure_ascii=False) + "\n")
                if n % args.checkpoint_every == 0:
                    out.flush()
                    os.fsync(out.fileno())
                bar.update(1)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def build_parallel(sampled, args, tool_names, tool_weights_list, tool_config, min_tools, max_tools):
    """Process-pool version of the main loop. Finished samples are appended to
    <output-dir>/.records_parallel.jsonl as they complete; a rerun skips them (and retries
    the ones that failed every attempt)."""

    progress_file = args.output_dir / ".records_parallel.jsonl"
    key_file = args.output_dir / ".records_parallel_key.json"
    run_key = {
        "seed": args.seed, "num_samples": args.num_samples, "repetition": args.repetition,
        "min_tools": min_tools, "max_tools": max_tools, "tools": tool_names,
        "sampled_ids": [str(item["id"]) for item in sampled],
    }
    done: Dict[int, Any] = {}
    if args.resume and progress_file.exists() and key_file.exists() \
            and json.loads(key_file.read_text(encoding="utf-8")) == run_key:
        with progress_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # line cut off by an interrupt
                    continue
                if row["record"] is not None:  # failed samples are retried on resume
                    done[row["idx"]] = row["record"]
        print(f"Resuming: {len(done)}/{len(sampled)} done", file=sys.stderr)
    else:
        if progress_file.exists():
            progress_file.unlink()
        key_file.write_text(json.dumps(run_key), encoding="utf-8")

    tasks = [(idx, item) for idx, item in enumerate(sampled, start=1) if idx not in done]
    ctx = {
        "seed": args.seed, "output_dir": args.output_dir, "tool_names": tool_names,
        "tool_weights": tool_weights_list, "tool_config": tool_config, "min_tools": min_tools,
        "max_tools": max_tools, "max_attempts": args.max_attempts_per_sample,
    }
    # One BLAS/numba thread per worker; spawned children inherit this environment.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"

    # Run in phases, one set of heavy-tool envs at a time: a server keeps its model on the
    # GPU once loaded, and e.g. SAM-Audio + AudioSR resident together don't fit in 12 GB.
    # Each sample's tools are known up front (its RNG is seeded per sample, and the chain is
    # drawn before any audio work), so phasing doesn't change what gets drawn.
    phases: Dict[tuple, list] = {}
    for task in tasks:
        envs = tuple(sorted({tool_config["tools"][n]["env"] for n in _peek_tools(task, ctx)
                             if tool_config["tools"][n].get("env")}))
        phases.setdefault(envs, []).append(task)
    for envs in sorted(phases, key=lambda e: (len(e), e)):
        label = ", ".join(Path(e).parents[1].name for e in envs) or "light tools"
        print(f"Phase [{label}]: {len(phases[envs])} sample(s)", file=sys.stderr)
        # Group same-parameter calls (e.g. AudioSR's basic vs speech checkpoint) so a server
        # swaps models as rarely as possible. Order doesn't affect what each sample draws.
        phase_tasks = sorted(phases[envs], key=lambda t: json.dumps(_peek(t, ctx)[1], sort_keys=True))
        with heavy_tool_pool.serve(envs):
            _run_phase(phase_tasks, ctx, args, sampled, done, progress_file)
    return [done[i] for i in sorted(done) if done[i] is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample real audio and apply a randomly drawn tools/ call to build (original, tool_applied) pairs."
    )
    parser.add_argument("--tool-config", type=Path, default=Path(__file__).resolve().parent / "tool_config.json")
    parser.add_argument("--input-json", type=Path, nargs="*", default=[],
                         help="dcase_subset.json-shaped QA list(s) to sample from (e.g. a DCASE 2025 subset).")
    parser.add_argument("--sources", nargs="*", choices=["audioset", "vctk"], default=[],
                         help="Raw audio pools (no QA attached) to sample from.")
    parser.add_argument("--audio-glob", nargs="*", default=[],
                         help="Extra glob pattern(s) for other raw audio pools, e.g. '/work/u1501463/my_corpus/**/*.wav'.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--repetition", type=int, default=1, help="Repeat the whole sample set this many times (for debugging).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dedupe-by-audio", action="store_true")
    parser.add_argument("--min-tools", type=int, default=None, help="Default: tool_config.json's min_tools.")
    parser.add_argument("--max-tools", type=int, default=None, help="Default: tool_config.json's max_tools.")
    parser.add_argument("--output-dir", type=Path, default=Path("/work/u1501463/gen_2nd_2_stage_tool_applied"))
    parser.add_argument("--max-attempts-per-sample", type=int, default=3)
    parser.add_argument("--manifest-file", type=Path, default=None, help="Default: <output-dir>/manifest.json")
    parser.add_argument("--original-subset-file", type=Path, default=None, help="Default: <output-dir>/original_subset.json")
    parser.add_argument("--tool-applied-subset-file", type=Path, default=None, help="Default: <output-dir>/tool_applied_subset.json")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                         help="Resume from <output-dir>/.build_checkpoint.json if present (default on).")
    parser.add_argument("--workers", type=int, default=1,
                         help="Worker processes. 1 = original sequential run (one shared RNG). >1 gives each "
                              "sample its own RNG seeded by (seed, index, id): deterministic regardless of worker "
                              "count or completion order, but a different draw than --workers 1.")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                         help="Save the resume checkpoint every N samples (a resumed run redoes at most N-1).")
    args = parser.parse_args()

    tool_config = ta.load_tool_config(args.tool_config)
    min_tools = args.min_tools if args.min_tools is not None else int(tool_config.get("min_tools", 1))
    max_tools = args.max_tools if args.max_tools is not None else int(tool_config.get("max_tools", 1))

    tool_names = ta.available_tool_names(tool_config)
    if not tool_names:
        raise SystemExit("No enabled+available tools in tool_config.json -- check `enabled` flags and heavy-tool `env` paths.")
    tool_weights_list = ta.tool_weights(tool_config, tool_names)
    print(f"Available tools ({len(tool_names)}): {tool_names}", file=sys.stderr)

    items = load_qa_items(args.input_json) + collect_raw_audio_items(args.sources, args.audio_glob)
    if not items:
        raise SystemExit("No input items -- pass --input-json and/or --sources/--audio-glob.")

    for item in items:
        item.setdefault("id", f"sample_{uuid.uuid4().hex[:12]}")

    sampled = sample_items(items, args.num_samples, args.seed, args.dedupe_by_audio)
    print(f"Sampled {len(sampled)}/{len(items)} item(s)", file=sys.stderr)
    sampled = sampled * args.repetition
    print(f"After repetition={args.repetition}, {len(sampled)} total item(s)", file=sys.stderr)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = args.manifest_file or (args.output_dir / "manifest.json")
    original_subset_file = args.original_subset_file or (args.output_dir / "original_subset.json")
    tool_applied_subset_file = args.tool_applied_subset_file or (args.output_dir / "tool_applied_subset.json")

    records = generate_records(
        sampled, args, tool_names, tool_weights_list, tool_config, min_tools, max_tools)

    write_outputs(records, sampled, manifest_file, original_subset_file, tool_applied_subset_file)


def generate_records(sampled, args, tool_names, tool_weights_list, tool_config, min_tools, max_tools):
    if args.workers > 1:
        return build_parallel(
            sampled, args, tool_names, tool_weights_list, tool_config, min_tools, max_tools)
    # One persistent server per heavy-tool conda env for the whole run (model loaded once).
    # No-op without heavy tools.
    with heavy_tool_pool.serve(ta.heavy_env_pythons(tool_config, tool_names)):
        rng = random.Random(args.seed)
        records: List[Dict[str, Any]] = []
        checkpoint_file = args.output_dir / ".build_checkpoint.json"
        run_key = {
            "seed": args.seed, "num_samples": args.num_samples, "repetition": args.repetition,
            "min_tools": min_tools, "max_tools": max_tools,
            "sampled_ids": [str(item["id"]) for item in sampled],
        }
        start_idx = 1
        if args.resume and checkpoint_file.exists():
            ckpt = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if ckpt.get("run_key") == run_key:
                records = ckpt["records"]
                start_idx = ckpt["next_idx"]
                # Restoring the RNG state makes a resumed run produce exactly what an
                # uninterrupted run would have.
                version, internal, gauss = ckpt["rng_state"]
                rng.setstate((version, tuple(internal), gauss))
                print(f"Resuming from checkpoint: {start_idx - 1}/{len(sampled)} done", file=sys.stderr)
            else:
                print("Checkpoint is from a different configuration; starting fresh.", file=sys.stderr)

        def save_checkpoint(next_idx: int) -> None:
            tmp = checkpoint_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "run_key": run_key, "next_idx": next_idx,
                "rng_state": rng.getstate(), "records": records,
            }), encoding="utf-8")
            os.replace(tmp, checkpoint_file)

        remaining = sampled[start_idx - 1:]
        for idx, item in tqdm(enumerate(remaining, start=start_idx), total=len(sampled), initial=start_idx - 1):
            for attempt in range(args.max_attempts_per_sample):
                try:
                    record = build_one_sample(
                        item, args.output_dir, tool_names, tool_weights_list, tool_config, min_tools, max_tools, rng
                    )
                    records.append(record)
                    ops = [step["tool"] for step in record["tool_chain"]]
                    print(f"[{idx}/{len(sampled)}] {item['id']}: applied {ops}", file=sys.stderr)
                    break
                except Exception:
                    print(f"Attempt {attempt + 1} failed for {item['id']}:", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            else:
                print(f"Giving up on {item['id']} after {args.max_attempts_per_sample} attempts.", file=sys.stderr)
            if (idx - start_idx + 1) % args.checkpoint_every == 0 or idx == len(sampled):
                save_checkpoint(idx + 1)

        return records


def write_outputs(records, sampled, manifest_file, original_subset_file, tool_applied_subset_file) -> None:
    qa_by_id = {item["id"]: item for item in sampled}

    with manifest_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    print(f"Wrote manifest for {len(records)} samples to {manifest_file}", file=sys.stderr)

    original_subset = [to_dcase_entry(qa_by_id[rec["id"]], rec["original_audio"]) for rec in records]
    with original_subset_file.open("w", encoding="utf-8") as handle:
        json.dump(original_subset, handle, indent=2, ensure_ascii=False)
    print(f"Wrote original subset ({len(original_subset)} items) to {original_subset_file}", file=sys.stderr)

    tool_applied_subset = [to_dcase_entry(qa_by_id[rec["id"]], rec["tool_applied_audio"]) for rec in records]
    with tool_applied_subset_file.open("w", encoding="utf-8") as handle:
        json.dump(tool_applied_subset, handle, indent=2, ensure_ascii=False)
    print(f"Wrote tool_applied subset ({len(tool_applied_subset)} items) to {tool_applied_subset_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
