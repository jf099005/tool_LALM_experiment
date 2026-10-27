#!/usr/bin/env python
"""
plot_loss.py

讀取訓練過程中產生的 logging.jsonl（每行一個 JSON，包含 loss / global_step / epoch 等欄位），
畫出 loss 曲線並存成圖檔。

用法:
    python plot_loss.py <logging.jsonl路徑> [--output OUTPUT.png] [--x-axis step|epoch] [--smooth N]

範例:
    python plot_loss.py /work/u1501463/stage1_training/v6-20260815-112315/logging.jsonl
    python plot_loss.py logging.jsonl --output loss_curve.png --x-axis epoch --smooth 5
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 無顯示器環境（如 slurm 計算節點）也能畫圖存檔
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="繪製訓練 loss 曲線")
    parser.add_argument("input", type=str, help="輸入的 logging.jsonl 檔案路徑")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="輸出圖檔路徑（預設: 與輸入同目錄下的 loss_curve.png）"
    )
    parser.add_argument(
        "--x-axis", choices=["step", "epoch"], default="step",
        help="x 軸使用 global_step 或 epoch（預設: step）"
    )
    parser.add_argument(
        "--smooth", type=int, default=1,
        help="移動平均視窗大小，用來平滑 loss 曲線（預設: 1，即不平滑）"
    )
    parser.add_argument(
        "--grad-norm", action="store_true",
        help="額外在副座標軸上畫出 grad_norm"
    )
    return parser.parse_args()


def load_records(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[警告] 第 {lineno} 行解析失敗，已略過: {e}", file=sys.stderr)
    return records


def extract_step(record):
    # global_step/max_steps 格式為 "step/max_steps" 字串，例如 "100/939"
    gs = record.get("global_step/max_steps")
    if gs is not None:
        try:
            return int(str(gs).split("/")[0])
        except (ValueError, IndexError):
            pass
    return None


def moving_average(values, window):
    if window <= 1:
        return values
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"錯誤: 找不到輸入檔案 {input_path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(input_path)
    if not records:
        print("錯誤: 檔案中沒有可用的紀錄", file=sys.stderr)
        sys.exit(1)

    xs, losses, grad_norms = [], [], []
    for r in records:
        if "loss" not in r:
            continue
        if args.x_axis == "step":
            x = extract_step(r)
        else:
            x = r.get("epoch")
        if x is None:
            continue
        xs.append(x)
        losses.append(r["loss"])
        grad_norms.append(r.get("grad_norm"))

    if not xs:
        print("錯誤: 找不到有效的 loss / x 軸資料", file=sys.stderr)
        sys.exit(1)

    smoothed_losses = moving_average(losses, args.smooth)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(xs, losses, color="tab:blue", alpha=0.3, linewidth=1, label="loss (raw)")
    if args.smooth > 1:
        ax1.plot(xs, smoothed_losses, color="tab:blue", linewidth=2,
                  label=f"loss (moving avg, window={args.smooth})")
    else:
        ax1.plot(xs, losses, color="tab:blue", linewidth=1.5, label="loss")

    ax1.set_xlabel("Global Step" if args.x_axis == "step" else "Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    if args.grad_norm and any(g is not None for g in grad_norms):
        ax2 = ax1.twinx()
        gn_xs = [x for x, g in zip(xs, grad_norms) if g is not None]
        gn_ys = [g for g in grad_norms if g is not None]
        ax2.plot(gn_xs, gn_ys, color="tab:red", alpha=0.4, linewidth=1, label="grad_norm")
        ax2.set_ylabel("Grad Norm", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if args.grad_norm and any(g is not None for g in grad_norms):
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    plt.title(f"Training Loss Curve\n{input_path}")
    fig.tight_layout()

    output_path = Path(args.output) if args.output else input_path.parent / "loss_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"已輸出圖檔: {output_path}")
    print(f"共讀取 {len(records)} 筆紀錄，有效繪圖點 {len(xs)} 個")
    print(f"loss 範圍: min={min(losses):.5f}, max={max(losses):.5f}, "
          f"first={losses[0]:.5f}, last={losses[-1]:.5f}")


if __name__ == "__main__":
    main()
