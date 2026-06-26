import argparse
import json
from collections import defaultdict
from pathlib import Path

TOOL_SCHEDULES_PATH = Path(__file__).parent / "tool_schedules.json"


def load_results(path: str) -> dict[str, bool]:
    with open(path) as f:
        data = json.load(f)
    return {entry["id"]: entry["correct"] for entry in data}


def load_category_map() -> dict[str, str]:
    with open(TOOL_SCHEDULES_PATH) as f:
        data = json.load(f)
    category_map: dict[str, str] = {}
    for entry in data:
        prob = entry[0]
        pid = prob["id"]
        category_map[pid] = prob.get("category") or "(none)"
    return category_map


def print_confusion(ids: set[str], results1: dict[str, bool], results2: dict[str, bool],
                    name1: str, name2: str, label: str) -> None:
    both_correct = correct1_wrong2 = wrong1_correct2 = both_wrong = 0
    for id_ in ids:
        c1, c2 = results1[id_], results2[id_]
        if c1 and c2:
            both_correct += 1
        elif c1 and not c2:
            correct1_wrong2 += 1
        elif not c1 and c2:
            wrong1_correct2 += 1
        else:
            both_wrong += 1

    total = len(ids)
    print(f"\n{'='*60}")
    print(f"  {label}  (n={total})")
    print(f"{'='*60}")
    print(f"\n{'':30s} {'File2 Correct':>15} {'File2 Wrong':>12}")
    print(f"{'File1 Correct':30s} {both_correct:>15} {correct1_wrong2:>12}")
    print(f"{'File1 Wrong':30s} {wrong1_correct2:>15} {both_wrong:>12}")

    if total > 0:
        print(f"\n  Both correct:              {both_correct:4d}  ({both_correct/total:.1%})")
        print(f"  File1 correct, File2 wrong:{correct1_wrong2:4d}  ({correct1_wrong2/total:.1%})")
        print(f"  File1 wrong, File2 correct:{wrong1_correct2:4d}  ({wrong1_correct2/total:.1%})")
        print(f"  Both wrong:                {both_wrong:4d}  ({both_wrong/total:.1%})")
        print(f"\n  File1 accuracy: {(both_correct + correct1_wrong2)/total:.1%}")
        print(f"  File2 accuracy: {(both_correct + wrong1_correct2)/total:.1%}")


def calculate_confusion(path1: str, path2: str, by_category: bool) -> None:
    results1 = load_results(path1)
    results2 = load_results(path2)

    common_ids = set(results1) & set(results2)
    only_in_1 = set(results1) - set(results2)
    only_in_2 = set(results2) - set(results1)

    name1 = Path(path1).name
    name2 = Path(path2).name

    print(f"File 1: {name1}")
    print(f"File 2: {name2}")
    print(f"\nCommon samples: {len(common_ids)}")
    if only_in_1:
        print(f"Only in file 1: {len(only_in_1)} samples (excluded from confusion)")
    if only_in_2:
        print(f"Only in file 2: {len(only_in_2)} samples (excluded from confusion)")

    if not by_category:
        print_confusion(common_ids, results1, results2, name1, name2, "Overall")
        return

    category_map = load_category_map()

    category_buckets: dict[str, set[str]] = defaultdict(set)
    for id_ in common_ids:
        category_buckets[category_map.get(id_, "(none)")].add(id_)

    print(f"\nCategories found: {len(category_buckets)}")

    for category in sorted(category_buckets):
        print_confusion(category_buckets[category], results1, results2, name1, name2, category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two result files with confusion matrices."
    )
    parser.add_argument("path1", help="Path to first results JSON file")
    parser.add_argument("path2", help="Path to second results JSON file")
    parser.add_argument(
        "-c", action="store_true",
        help="Break down confusion matrices by category"
    )
    args = parser.parse_args()
    calculate_confusion(args.path1, args.path2, args.c)
