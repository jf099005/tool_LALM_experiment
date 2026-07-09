#!/usr/bin/env python3
"""Merge two prediction json files (see predictions/qwen25/Dcase_base.json for
the sample format) into one, deciding per-id conflicts with cmp().
"""
import argparse
import json


def load_json(path):
    with open(path) as f:
        return json.load(f)


def accuracy(samples):
    if not samples:
        return 0.0
    return sum(1 for s in samples if s.get("correct")) / len(samples)


def cmp(sample_a, sample_b):
    """Decide which of two same-id samples to keep.

    Default: trust whichever sample the model is more confident in, using
    the P(True) self-verification score from uncertainty quantification.
    Falls back to sample_a when p_true is missing or tied. Edit this
    function to change the merge criteria (e.g. prefer `correct`, lower
    semantic entropy, etc).
    """

    delta = 0.1

    pa = (sample_a.get("uncertainty") or {}).get("p_true")
    pb = (sample_b.get("uncertainty") or {}).get("p_true")
    if pa is None:
        return sample_b if pb is not None else sample_a
    
    if pb is None:
        return sample_a

    return sample_b if pb > pa + delta else sample_a
    # return sample_a if pa >= pb else sample_b


def selection_breakdown(merged, decisions):
    """Count correct/wrong items selected from each source file."""
    samples_by_id = {s["id"]: s for s in merged}
    counts = {
        "file1": {"correct": 0, "wrong": 0},
        "file2": {"correct": 0, "wrong": 0},
    }
    for d in decisions:
        bucket = counts[d["kept_from"]]
        if samples_by_id[d["id"]].get("correct"):
            bucket["correct"] += 1
        else:
            bucket["wrong"] += 1
    return counts


def merge(data1, data2):
    map1 = {s["id"]: s for s in data1}
    map2 = {s["id"]: s for s in data2}

    ids1_only = set(map1) - set(map2)
    ids2_only = set(map2) - set(map1)
    common_ids = set(map1) & set(map2)

    merged = []
    decisions = []

    for id_ in ids1_only:
        merged.append(map1[id_])
        decisions.append({"id": id_, "kept_from": "file1", "reason": "only in file1"})

    for id_ in ids2_only:
        merged.append(map2[id_])
        decisions.append({"id": id_, "kept_from": "file2", "reason": "only in file2"})

    for id_ in common_ids:
        a, b = map1[id_], map2[id_]
        kept = cmp(a, b)
        merged.append(kept)
        decisions.append({
            "id": id_,
            "kept_from": "file1" if kept is a else "file2",
            "reason": "cmp decision",
        })

    merged.sort(key=lambda s: s.get("index", 0))
    decisions.sort(key=lambda d: d["id"])
    return merged, decisions


def main():
    parser = argparse.ArgumentParser(
        description="Merge two prediction json files by id using cmp()."
    )
    parser.add_argument("file1", help="Path to first json file")
    parser.add_argument("file2", help="Path to second json file")
    parser.add_argument("-o", "--output", default="merged.json", help="Path for merged output json")
    parser.add_argument("--log", default="merge_log.json", help="Path for the per-id decision log")
    args = parser.parse_args()

    data1 = load_json(args.file1)
    data2 = load_json(args.file2)

    merged, decisions = merge(data1, data2)

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    with open(args.log, "w") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    print(f"File1 ({args.file1}): n={len(data1)}, accuracy={accuracy(data1):.2%}")
    print(f"File2 ({args.file2}): n={len(data2)}, accuracy={accuracy(data2):.2%}")
    print(f"Merged ({args.output}): n={len(merged)}, accuracy={accuracy(merged):.2%}")

    counts = selection_breakdown(merged, decisions)
    for name in ("file1", "file2"):
        c = counts[name]
        print(f"Selected from {name}: {c['correct']} correct, {c['wrong']} wrong "
              f"(total {c['correct'] + c['wrong']})")

    print(f"Decision log written to {args.log}")


if __name__ == "__main__":
    main()
