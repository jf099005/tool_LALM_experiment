"""Build the stage-3 (MCQ-with-tools RL) parquet files for verl-omni.

Training sources (all with local audio under $STAGE3_WORK_DIR, default /work/u1501463):
  - clotho / audiocaps  AudioMCQ's Clotho (3.8k) and AudioCaps v2.0 (90.5k) splits: MCQs an LLM
                     wrote from the datasets' text captions, each tagged with audio-contribution
                     like compa_r. Off by default (--clotho-n / --audiocaps-n).
  - compa_r          AudioMCQ's CompA-R split (197k MCQs) re-pointed at the local CompA-R mp3s.
                     AudioMCQ labels every item with `audio-contribution` (weak = answerable
                     from text priors alone, strong = needs the audio); RL defaults to strong.
  - dcase2025        DCASE 2025 task-5 AQA train questions (generate/dcase/dcase_7481.json),
                     re-pointed at /work/u1501463/DCASE_2025_task5 (the original
                     2025_DCASE_AudioQA_Official tree no longer exists).
  - audioset_labels  audioset_20k ships audio + AudioSet labels only, no QA. We synthesize
                     "which sound can be heard" MCQs: the answer is one of the clip's labels,
                     distractors are labels from *other top-level ontology branches* than any of
                     the clip's labels (AudioSet labels are incomplete, so same-branch distractors
                     could actually be audible). Needs audioset_20k/ontology.json.
Validation / evaluation (never trained on):
  - mmau_test_mini   MMAU test-mini (1000; audio extracted from gamma-lab-umd/MMAU-test-mini).
  - mmar             MMAR (1000).

Outputs (in --output-dir): train.parquet, val.parquet (stratified MMAU test-mini subset for
in-training validation), eval_mmau_test_mini.parquet, eval_mmar.parquet, stats.json.

Every row's prompt = stage-1's exact system turn (tool catalogue) + `mcq_protocol.render_user_prompt`;
option order is reshuffled per item (seeded) so the answer letter is uniformly distributed.

Usage (CPU-only; see build_data.sbatch):
    python build_mcq_dataset.py --output-dir /work/u1501463/stage3_mcq_rl/data

`--format uq` instead writes one dcase_subset.json-shaped `uq_subset.json` (choices as
"A. ...", answer as "B. ..."), the input format of gen_2nd_2_data/build_dataset.py and
qwen25_with_tool_chain_evaluation.py; it needs no tool catalogue and skips the eval sets.
See run_local_uq.sh for the local-PC run.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcq_protocol  # noqa: E402

WORK = Path(os.environ.get("STAGE3_WORK_DIR", "/work/u1501463"))
REPO_ROOT = mcq_protocol.REPO_ROOT

AGENT_NAME = "audio_tool_mcq_agent"
ABILITY = "audio_mcq_tool_use"

AUDIOMCQ_DIR = WORK / "AudioMCQ/datasets"
COMPA_R_JSONL = AUDIOMCQ_DIR / "AudioMCQ_CompA-R.jsonl"
CLOTHO_AUDIO_DIR = WORK / "Clotho/audio"
AUDIOCAPS_AUDIO_DIR = WORK / "AudioCaps/audio"
COMPA_R_AUDIO_DIRS = [WORK / "CompA_R/compa_r_train_audios-mp3", WORK / "CompA_R/compa_r_test_audios-mp3"]
DCASE_JSON = REPO_ROOT / "generate/dcase/dcase_7481.json"
DCASE_AUDIO_DIRS = [WORK / "DCASE_2025_task5/train", WORK / "DCASE_2025_task5/dcase_25_task5_audio_part_1/train"]
AUDIOSET_DIR = WORK / "audioset_20k/20k/train"
AUDIOSET_ONTOLOGY = WORK / "audioset_20k/ontology.json"
MMAU_JSON = WORK / "MMAU/mmau-test-mini.json"
MMAU_PARQUET = WORK / "MMAU/test_mini.parquet"
MMAU_AUDIO_DIR = WORK / "MMAU/test-mini-audios"
MMAR_JSON = WORK / "MMAR/MMAR-meta.json"
MMAR_DIR = WORK / "MMAR"


def _duration(path: Path) -> Optional[float]:
    try:
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:  # noqa: BLE001 - unreadable file -> dropped and counted
        return None


def _strip_letter(option: str) -> str:
    """'B. True' -> 'True' (DCASE options carry their own letter prefix)."""
    option = option.strip()
    if len(option) > 2 and option[0] in mcq_protocol.LETTERS and option[1] in ".)":
        return option[2:].strip()
    return option


# ---------------------------------------------------------------------------
# Source loaders: each yields {"id","audio","question","choices","answer", **meta}
# ---------------------------------------------------------------------------


def load_audiomcq(name: str, jsonl: Path, audio_dirs: List[Path], contribution: str) -> List[Dict[str, Any]]:
    """Any AudioMCQ split: rows are matched to local audio by file stem (AudioMCQ's own
    `audio_path` prefix, e.g. clotho_audios/, is not meaningful locally)."""
    index = {p.stem: p for d in audio_dirs if d.exists() for p in d.iterdir()}
    items = []
    missing = 0
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if contribution != "all" and row.get("audio-contribution") != contribution:
                continue
            audio = index.get(Path(row["audio_path"]).stem)
            if audio is None:
                missing += 1
                continue
            items.append({
                "id": f"{name}/{row['id']}/{len(items)}",
                "audio": audio,
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "category": row.get("question_type", ""),
                "audio_contribution": row.get("audio-contribution", ""),
            })
    if missing:
        print(f"{name}: {missing} item(s) skipped, no local audio", file=sys.stderr)
    return items


def load_compa_r(contribution: str) -> List[Dict[str, Any]]:
    return load_audiomcq("compa_r", COMPA_R_JSONL, COMPA_R_AUDIO_DIRS, contribution)


def load_dcase() -> List[Dict[str, Any]]:
    items = []
    for row in json.loads(DCASE_JSON.read_text(encoding="utf-8")):
        name = Path(row["audio_url"]).name
        audio = next((d / name for d in DCASE_AUDIO_DIRS if (d / name).exists()), None)
        if audio is None:
            continue
        items.append({
            "id": f"dcase2025/{row['id']}/{Path(row['_json_path']).stem}",
            "audio": audio,
            "question": row["question"],
            "choices": [_strip_letter(c) for c in row["choice"]],
            "answer": _strip_letter(row["answer"]),
            "category": row.get("question_type", ""),
        })
    return items


def _ontology_top_branches() -> tuple[Dict[str, str], Dict[str, set]]:
    """name -> display name, and name -> set of top-level ancestor names."""
    nodes = {n["id"]: n for n in json.loads(AUDIOSET_ONTOLOGY.read_text(encoding="utf-8"))}
    parents = collections.defaultdict(set)
    for n in nodes.values():
        for child in n.get("child_ids", []):
            parents[child].add(n["id"])

    def tops(node_id: str, seen=None) -> set:
        seen = seen or set()
        if node_id in seen:
            return set()
        seen.add(node_id)
        if not parents[node_id]:
            return {node_id}
        out = set()
        for p in parents[node_id]:
            out |= tops(p, seen)
        return out

    by_name = {n["name"]: nid for nid, n in nodes.items()}
    top_of = {name: {nodes[t]["name"] for t in tops(nid)} for name, nid in by_name.items()}
    return by_name, top_of


# Distractors only come from these ontology roots: "Human sounds", "Music", "Source-ambiguous
# sounds" and "Channel, environment and background" are so often present-but-unlabeled in
# AudioSet (speech, background music, room noise) that using them as "absent" options would
# plant wrong ground truth.
DISTRACTOR_TOPS = {"Animal", "Natural sounds", "Sounds of things"}


def load_audioset_labels(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    by_name, top_of = _ontology_top_branches()
    clips = []
    vocab = set()
    for meta in sorted(AUDIOSET_DIR.glob("*.json")):
        labels = [l for l in json.loads(meta.read_text(encoding="utf-8"))["label"] if l in by_name]
        wav = meta.with_suffix(".wav")
        if labels and wav.exists():
            clips.append((meta.stem, wav, labels))
            vocab.update(labels)
    vocab = sorted(vocab)
    rng.shuffle(clips)
    items = []
    for stem, wav, labels in clips:
        if len(items) >= n:
            break
        clip_tops = set().union(*(top_of[l] for l in labels))
        pool = [v for v in vocab if top_of[v] <= DISTRACTOR_TOPS and not (top_of[v] & clip_tops)]
        if len(pool) < 3:
            continue
        answer = rng.choice(labels)
        items.append({
            "id": f"audioset_labels/{stem}",
            "audio": wav,
            "question": "Which of the following sounds can be heard in the audio?",
            "choices": [answer] + rng.sample(pool, 3),
            "answer": answer,
            "category": "sound_event",
        })
    return items


def load_mmau() -> List[Dict[str, Any]]:
    MMAU_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if len(list(MMAU_AUDIO_DIR.glob("*.wav"))) < 1000:
        import pyarrow.parquet as pq

        # The HF parquet's audio struct has no path; its `other_attributes.id` is the
        # mmau-test-mini.json id, which is also the wav name there.
        table = pq.read_table(MMAU_PARQUET, columns=["context", "other_attributes"])
        for ctx, attrs in zip(table.column("context").to_pylist(), table.column("other_attributes").to_pylist()):
            out = MMAU_AUDIO_DIR / f"{json.loads(attrs)['id']}.wav"
            if not out.exists():
                out.write_bytes(ctx["bytes"])
    items = []
    for row in json.loads(MMAU_JSON.read_text(encoding="utf-8")):
        items.append({
            "id": f"mmau_test_mini/{row['id']}",
            "audio": MMAU_AUDIO_DIR / Path(row["audio_id"]).name,
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "category": row.get("task", ""),
            "sub_category": row.get("sub-category", ""),
            "difficulty": row.get("difficulty", ""),
        })
    return items


def load_mmar() -> List[Dict[str, Any]]:
    return [{
        "id": f"mmar/{row['id']}",
        "audio": (MMAR_DIR / row["audio_path"]).resolve(),
        "question": row["question"],
        "choices": row["choices"],
        "answer": row["answer"],
        "category": row.get("modality", ""),
        "sub_category": row.get("category", ""),
    } for row in json.loads(MMAR_JSON.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------


def to_rows(
    items: Iterable[Dict[str, Any]],
    data_source: str,
    split: str,
    system_prompt: str,
    rng: random.Random,
    max_seconds: Optional[float],
    dropped: collections.Counter,
) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        audio = Path(item["audio"])
        if not audio.exists():
            dropped[f"{data_source}:missing_audio"] += 1
            continue
        dur = _duration(audio)
        if dur is None:
            dropped[f"{data_source}:unreadable_audio"] += 1
            continue
        if max_seconds is not None and dur > max_seconds:
            dropped[f"{data_source}:too_long"] += 1
            continue
        choices = [str(c).strip() for c in item["choices"]]
        answer = str(item["answer"]).strip()
        repaired = ""
        if split == "eval":
            # Benchmarks must keep their full denominator: MMAU test-mini has 27 items with a
            # duplicated option, MMAR 5 whose answer isn't verbatim among the options.
            if len(set(choices)) != len(choices):
                choices = list(dict.fromkeys(choices))
                repaired = "dedup_choices"
            if answer not in choices:
                best = max(choices, key=lambda c: difflib.SequenceMatcher(None, c.lower(), answer.lower()).ratio())
                if difflib.SequenceMatcher(None, best.lower(), answer.lower()).ratio() >= 0.6:
                    answer, repaired = best, "fuzzy_answer"
        if len(set(choices)) != len(choices) or not 2 <= len(choices) <= len(mcq_protocol.LETTERS):
            dropped[f"{data_source}:bad_choices"] += 1
            continue
        try:
            choices, letter = mcq_protocol.shuffle_choices(choices, answer, rng)
        except ValueError:
            dropped[f"{data_source}:answer_not_in_choices"] += 1
            continue
        meta = {k: str(v) for k, v in item.items() if k not in {"audio", "question", "choices", "answer"}}
        rows.append({
            "data_source": data_source,
            "agent_name": AGENT_NAME,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": mcq_protocol.render_user_prompt(item["question"], choices)},
            ],
            "audios": [str(audio)],
            "ability": ABILITY,
            "reward_model": {"style": "rule", "ground_truth": letter},
            "extra_info": {
                **meta,
                "split": split,
                "index": len(rows),
                "question": item["question"],
                "choices": json.dumps(choices, ensure_ascii=False),
                "answer_text": answer,
                "repaired": repaired,
                "duration": round(dur, 3),
            },
        })
    return rows


def _write(rows: List[Dict[str, Any]], path: Path) -> None:
    # use_dictionary=False: same PyArrow nested-column workaround as verl-omni's avqa.py.
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False, use_dictionary=False)


def _safe_id(raw: str, used: set) -> str:
    """Filesystem-safe, unique id (gen_2nd_2_data/build_dataset.py uses ids as dir names)."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")[:120] or "item"
    out, k = base, 1
    while out in used:
        k += 1
        out = f"{base}_{k}"
    used.add(out)
    return out


def to_uq_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """verl rows -> dcase_subset.json-shaped items (the UQ eval checks `answer in prediction`,
    so options and answer carry their letter like DCASE's do)."""
    used: set = set()
    items = []
    for r in rows:
        info = r["extra_info"]
        choices = json.loads(info["choices"])
        letter = r["reward_model"]["ground_truth"]
        items.append({
            "id": _safe_id(info["id"], used),
            "source_id": info["id"],
            "data_source": r["data_source"],
            "question": info["question"],
            "choice": [f"{mcq_protocol.LETTERS[i]}. {c}" for i, c in enumerate(choices)],
            "answer": f"{letter}. {info['answer_text']}",
            "question_type": info.get("category", ""),
            "audio_contribution": info.get("audio_contribution", ""),
            "duration": info["duration"],
            "audio_url": r["audios"][0],
            "audio_path": r["audios"][0],
        })
    return items


def _stratified(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["extra_info"].get("category", "")].append(r)
    picked = []
    for cat_rows in by_cat.values():
        rng.shuffle(cat_rows)
        picked += cat_rows[: max(1, round(n * len(cat_rows) / len(rows)))]
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=WORK / "stage3_mcq_rl/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compa-r-n", type=int, default=8000, help="0 disables the source")
    ap.add_argument("--compa-r-contribution", choices=["strong", "weak", "all"], default="strong")
    ap.add_argument("--clotho-n", type=int, default=0, help="AudioMCQ Clotho split; -1 = all, 0 disables")
    ap.add_argument("--audiocaps-n", type=int, default=0, help="AudioMCQ AudioCaps split; -1 = all, 0 disables")
    ap.add_argument("--audiomcq-contribution", choices=["strong", "weak", "all"], default="strong",
                    help="audio-contribution filter for the Clotho/AudioCaps splits")
    ap.add_argument("--audiomcq-dir", type=Path, default=AUDIOMCQ_DIR, help="dir holding AudioMCQ_*.jsonl")
    ap.add_argument("--clotho-audio-dir", type=Path, default=CLOTHO_AUDIO_DIR)
    ap.add_argument("--audiocaps-audio-dir", type=Path, default=AUDIOCAPS_AUDIO_DIR)
    ap.add_argument("--total-n", type=int, default=None,
                    help="cap the train set at N; AudioCaps (the largest pool) is sized to fill up to N")
    ap.add_argument("--format", choices=["verl", "uq"], default="verl",
                    help="verl: RL parquet files (default). uq: uq_subset.json for the UQ evaluation.")
    ap.add_argument("--skip-eval", action="store_true", help="don't build the MMAU/MMAR val/eval sets")
    ap.add_argument("--dcase-n", type=int, default=-1, help="-1 = all (7481), 0 disables")
    ap.add_argument("--audioset-n", type=int, default=2000, help="0 disables the source")
    ap.add_argument("--val-n", type=int, default=200, help="MMAU test-mini subset used as in-training val")
    ap.add_argument("--max-train-seconds", type=float, default=30.0,
                    help="drop longer training clips (tool outputs re-enter the context as audio tokens)")
    ap.add_argument("--max-eval-seconds", type=float, default=None)
    ap.add_argument("--system-prompt-model-type", default="qwen3_omni",
                    help="official_system_prompt key; stage-1 used qwen3_omni (empty persona)")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap every source at N items")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # The UQ format never shows this prompt, and building it needs the stage-1 tool registry.
    system_prompt = "" if args.format == "uq" else mcq_protocol.build_system_prompt(args.system_prompt_model_type)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dropped: collections.Counter = collections.Counter()

    def take(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        items = list(items)
        rng.shuffle(items)
        if n >= 0:
            items = items[:n]
        return items[: args.limit] if args.limit else items

    train: List[Dict[str, Any]] = []
    if args.compa_r_n:
        # Oversample 1.2x so the duration filter can't leave us short.
        train += to_rows(take(load_compa_r(args.compa_r_contribution), int(args.compa_r_n * 1.2)),
                         "compa_r", "train", system_prompt, rng, args.max_train_seconds, dropped)[: args.compa_r_n]
    if args.dcase_n:
        train += to_rows(take(load_dcase(), args.dcase_n), "dcase2025", "train", system_prompt, rng,
                         args.max_train_seconds, dropped)
    if args.audioset_n:
        train += to_rows(take(load_audioset_labels(args.audioset_n, rng), -1), "audioset_labels", "train",
                         system_prompt, rng, args.max_train_seconds, dropped)
    if args.clotho_n:
        clotho = load_audiomcq("clotho", args.audiomcq_dir / "AudioMCQ_Clotho.jsonl",
                               [args.clotho_audio_dir], args.audiomcq_contribution)
        train += to_rows(take(clotho, args.clotho_n), "clotho", "train", system_prompt, rng,
                         args.max_train_seconds, dropped)
    if args.audiocaps_n:
        n = args.audiocaps_n
        if args.total_n is not None:
            n = max(0, args.total_n - len(train)) if n < 0 else min(n, max(0, args.total_n - len(train)))
        if n:
            audiocaps = load_audiomcq("audiocaps", args.audiomcq_dir / "AudioMCQ_AudioCaps.jsonl",
                                      [args.audiocaps_audio_dir], args.audiomcq_contribution)
            # Oversample 1.2x so the duration filter can't leave us short.
            train += to_rows(take(audiocaps, int(n * 1.2) if n > 0 else -1), "audiocaps", "train",
                             system_prompt, rng, args.max_train_seconds, dropped)[: n if n > 0 else None]
    if args.total_n is not None:
        train = train[: args.total_n]
    rng.shuffle(train)
    for i, row in enumerate(train):
        row["extra_info"]["index"] = i

    if args.format == "uq":
        uq_items = to_uq_items(train)
        out = args.output_dir / "uq_subset.json"
        out.write_text(json.dumps(uq_items, indent=2, ensure_ascii=False), encoding="utf-8")
        stats = {
            "uq_subset": len(uq_items),
            "by_source": dict(collections.Counter(i["data_source"] for i in uq_items)),
            "by_contribution": dict(collections.Counter(i["audio_contribution"] for i in uq_items)),
            "answer_letters": dict(sorted(collections.Counter(i["answer"][0] for i in uq_items).items())),
            "dropped": dict(sorted(dropped.items())),
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))
        print(f"Wrote {len(uq_items)} items to {out}")
        return

    _write(train, args.output_dir / "train.parquet")
    if args.skip_eval:
        print(json.dumps({"train": dict(collections.Counter(r["data_source"] for r in train)),
                          "dropped": dict(sorted(dropped.items()))}, indent=2))
        return

    mmau = to_rows(take(load_mmau(), -1), "mmau_test_mini", "eval", system_prompt, rng, args.max_eval_seconds, dropped)
    mmar = to_rows(take(load_mmar(), -1), "mmar", "eval", system_prompt, rng, args.max_eval_seconds, dropped)
    val = _stratified(mmau, args.val_n, random.Random(args.seed))

    _write(val, args.output_dir / "val.parquet")
    _write(mmau, args.output_dir / "eval_mmau_test_mini.parquet")
    _write(mmar, args.output_dir / "eval_mmar.parquet")

    stats = {
        "train": dict(collections.Counter(r["data_source"] for r in train)),
        "train_answer_letters": dict(sorted(collections.Counter(r["reward_model"]["ground_truth"] for r in train).items())),
        "val": len(val),
        "eval_mmau_test_mini": len(mmau),
        "eval_mmar": len(mmar),
        "dropped": dict(sorted(dropped.items())),
        "eval_repaired": dict(collections.Counter(
            f"{r['data_source']}:{r['extra_info']['repaired']}" for r in mmau + mmar if r["extra_info"]["repaired"]
        )),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
