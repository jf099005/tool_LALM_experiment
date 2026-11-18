"""Plan a large tool-applied set by reusing each question with several different tools.

Every question of the pool (a dcase_subset-shaped uq_subset.json, e.g. from
stage3_mcq_rl/build_mcq_dataset.py --format uq) gets K variants, K chosen so the total is
--total. A question's variants all use *different* tools, none of them the tool that question
already got in an earlier build (--exclude-dataset, e.g. the 50k set), so no (question, tool)
pair repeats. Each variant still draws heavy/ML vs light tools like build_dataset.py does
(tool config's heavy_tool_probability), within the tools the question has left.

Output (--output-dir):
    slice_00/input.json ... slice_NN/input.json   items for build_dataset.py --input-json; each
        item is the pool item with id "<question id>__v<k>", base_id = question id, and
        force_tools = [planned tool]. Parameters are drawn at build time.
    plan_summary.json                             counts, tool mix, slice sizes, arguments

Deterministic for a given --seed. Run once; the slice files are the plan.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import tool_appliers as ta  # noqa: E402


def question_key(item: dict) -> str:
    """Stable per-question key across builds: AudioMCQ source + row id (uq ids embed a
    build-specific index, e.g. 'audiocaps/80233/29401' -> 'audiocaps/80233')."""
    sid = item.get("source_id") or item["id"]
    parts = sid.split("/")
    return "/".join(parts[:2]) if len(parts) >= 3 else sid


def excluded_tools(dataset_dirs, uq_subsets) -> dict:
    """question key -> set of tools already applied to it in earlier builds."""
    out = collections.defaultdict(set)
    for ddir, subset in zip(dataset_dirs, uq_subsets):
        key_of = {it["id"]: question_key(it) for it in json.loads(Path(subset).read_text(encoding="utf-8"))}
        for rec in json.loads((Path(ddir) / "manifest.json").read_text(encoding="utf-8")):
            if rec["id"] in key_of:
                out[key_of[rec["id"]]].update(step["tool"] for step in rec["tool_chain"])
    return out


def pick_tool(candidates, weights, tool_config, rng) -> str:
    """One tool: heavy group with heavy_tool_probability (if any heavy left), then weighted."""
    p = tool_config.get("heavy_tool_probability")
    heavy = [(t, w) for t, w in zip(candidates, weights) if tool_config["tools"][t].get("env")]
    light = [(t, w) for t, w in zip(candidates, weights) if not tool_config["tools"][t].get("env")]
    if p is None:
        group = list(zip(candidates, weights))
    else:
        group = heavy if heavy and (not light or rng.random() < p) else light
    total = sum(w for _, w in group)
    r = rng.uniform(0, total)
    upto = 0.0
    for t, w in group:
        upto += w
        if upto >= r:
            return t
    return group[-1][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help="uq_subset.json of the question pool")
    ap.add_argument("--tool-config", type=Path, required=True)
    ap.add_argument("--exclude-dataset", type=Path, nargs="*", default=[],
                    help="earlier build dir(s) (with manifest.json) whose (question, tool) pairs to avoid")
    ap.add_argument("--exclude-uq-subset", type=Path, nargs="*", default=[],
                    help="the uq_subset.json each --exclude-dataset was built from (same order)")
    ap.add_argument("--total", type=int, default=500_000)
    ap.add_argument("--n-slices", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if len(args.exclude_dataset) != len(args.exclude_uq_subset):
        raise SystemExit("--exclude-dataset and --exclude-uq-subset must pair up")

    cfg = ta.load_tool_config(args.tool_config)
    tools = ta.available_tool_names(cfg)
    weights = dict(zip(tools, ta.tool_weights(cfg, tools)))
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    exclude = excluded_tools(args.exclude_dataset, args.exclude_uq_subset)
    rng = random.Random(args.seed)

    n_q = len(pool)
    base, extra = divmod(args.total, n_q)
    plus_one = set(rng.sample(range(n_q), extra))
    variants, short = [], 0
    for qi, item in enumerate(pool):
        k = base + (1 if qi in plus_one else 0)
        left = [t for t in tools if t not in exclude.get(question_key(item), set())]
        if k > len(left):
            short += k - len(left)
            k = len(left)
        for v in range(k):
            tool = pick_tool(left, [weights[t] for t in left], cfg, rng)
            left.remove(tool)
            variants.append({**item, "id": f"{item['id']}__v{v}", "base_id": item["id"], "force_tools": [tool]})
    rng.shuffle(variants)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    size = -(-len(variants) // args.n_slices)
    slices = [variants[i:i + size] for i in range(0, len(variants), size)]
    for si, part in enumerate(slices):
        d = args.output_dir / f"slice_{si:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.json").write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")

    mix = collections.Counter(v["force_tools"][0] for v in variants)
    heavy = sum(c for t, c in mix.items() if cfg["tools"][t].get("env"))
    summary = {
        "questions": n_q, "variants": len(variants), "short_of_total": short,
        "excluded_questions": len(exclude), "slices": [len(p) for p in slices],
        "heavy_share": heavy / len(variants) if variants else 0.0,
        "tool_mix": dict(mix.most_common()),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.output_dir / "plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
