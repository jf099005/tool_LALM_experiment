import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def extract_label(text: str) -> str | None:
    """Return the leading choice letter (A–D) from a prediction or answer string."""
    m = re.match(r"^\s*([A-Da-d])[.\s]", text.strip())
    return m.group(1).upper() if m else None


def load_records(path: str) -> dict:
    """Return {id: {answer_label, pred_label}} for parseable records."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    out, skipped = {}, 0
    for r in records:
        true_label = extract_label(r["answer"])
        pred_label = extract_label(r["prediction"])
        if true_label is None or pred_label is None:
            skipped += 1
            continue
        out[r["id"]] = {"answer": true_label, "pred": pred_label}

    if skipped:
        print(f"  [{path}] Skipped {skipped} records with unparseable labels.")
    return out


def plot_cm(ax, y_true, y_pred, title: str, xlabel="Predicted", ylabel="True"):
    all_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    ax.set_title(f"{title}\nAgreement: {acc:.1%}  (n={cm.sum()})", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrices for two prediction files.")
    parser.add_argument("--with_tool", default="predictions_with_tool.json")
    parser.add_argument("--no_tool", default="predictions_no_tool.json")
    parser.add_argument("--output", default="confusion_matrices.png")
    args = parser.parse_args()

    print(f"Loading {args.no_tool} ...")
    no_tool = load_records(args.no_tool)
    print(f"Loading {args.with_tool} ...")
    with_tool = load_records(args.with_tool)

    # Common IDs for the method-vs-method comparison
    common_ids = sorted(set(no_tool) & set(with_tool))
    print(f"Common records: {len(common_ids)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")

    # No-tool vs ground truth
    nt_true = [no_tool[i]["answer"] for i in no_tool]
    nt_pred = [no_tool[i]["pred"]   for i in no_tool]
    plot_cm(axes[0], nt_true, nt_pred, "No Tool  (vs. Ground Truth)")

    # With-tool vs ground truth
    wt_true = [with_tool[i]["answer"] for i in with_tool]
    wt_pred = [with_tool[i]["pred"]   for i in with_tool]
    plot_cm(axes[1], wt_true, wt_pred, "With Tool  (vs. Ground Truth)")

    # With-tool predictions vs No-tool predictions (common subset)
    nt_common = [no_tool[i]["pred"]   for i in common_ids]
    wt_common = [with_tool[i]["pred"] for i in common_ids]
    plot_cm(
        axes[2], nt_common, wt_common,
        "No Tool vs. With Tool\n(prediction agreement on common subset)",
        xlabel="With Tool prediction",
        ylabel="No Tool prediction",
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
