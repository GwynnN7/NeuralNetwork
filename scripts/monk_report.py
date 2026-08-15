#!/usr/bin/env python3
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "_mse"
OUT = "plots"

TRAIN = dict(color="#B03030", linestyle="-", marker="o", label="Training")
TEST = dict(color="#2040A0", linestyle="--", marker="^", label="Test")
plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
                     "legend.fontsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})

def mean_curves(run):
    files = sorted(glob.glob(f"artifacts/{run}/outer0_m*.csv"))
    if not files:
        return None
    if len(files) > 1:
        print(f"  note: {run} has {len(files)} retrained models, plotting {os.path.basename(files[0])}")
    df = pd.read_csv(files[0])
    out = {"epoch": None}
    for col in ["train_mse", "test_mse", "train_acc", "test_acc"]:
        piv = df.pivot_table(index="epoch", columns=["fold", "trial"], values=col)
        # Only epochs every trial reached, so the tail is never a forward-filled minority
        piv = piv[piv.notna().sum(axis=1) == piv.shape[1]]
        out[col] = piv.mean(axis=1).to_numpy()
        out["epoch"] = piv.index.to_numpy()
        out["trials"] = piv.shape[1]
    return out


def panel(ax, c, kind, title):
    tr, te = c[f"train_{kind}"], c[f"test_{kind}"]   # accuracy is already a percentage in the log
    every = max(1, len(c["epoch"]) // 12)
    for series, style in ((tr, TRAIN), (te, TEST)):
        ax.plot(c["epoch"], series, linewidth=1.8, markersize=6, markevery=every, **style)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE" if kind == "mse" else "Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="center right" if kind == "mse" else "lower right", framealpha=0.95)
    if kind == "mse":
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(top=101)


TASKS = [(f"monk1{SUFFIX}", "MONK 1"),
         (f"monk2{SUFFIX}", "MONK 2"),
         (f"monk3{SUFFIX}", "MONK 3 (regularized)"),
         (f"monk3{SUFFIX}_noreg", "MONK 3 (no regularization)")]

os.makedirs(OUT, exist_ok=True)
found = [(r, t) for r, t in TASKS if mean_curves(r) is not None]
missing = [r for r, _ in TASKS if mean_curves(r) is None]
if missing:
    print(f"skipped (no artifacts): {', '.join(missing)}")

fig, axes = plt.subplots(len(found), 2, figsize=(15, 4.6 * len(found)), squeeze=False)
for row, (run, title) in enumerate(found):
    c = mean_curves(run)
    panel(axes[row][0], c, "mse", f"{title} — MSE")
    panel(axes[row][1], c, "acc", f"{title} — Accuracy")
    for kind in ("mse", "acc"):
        f1, a1 = plt.subplots(figsize=(8, 5))
        panel(a1, c, kind, f"{title} — {'MSE' if kind == 'mse' else 'Accuracy'}")
        f1.tight_layout()
        f1.savefig(f"{OUT}/{run}_{kind}.png", dpi=150)
        plt.close(f1)
    model = os.path.basename(glob.glob(f"artifacts/{run}/outer0_m*.csv")[0])[7:-4]
    print(f"  {title:<32} {model:<4} epochs 0..{c['epoch'][-1]} "
          f"(cut at the shortest of {c['trials']} trials)")

fig.tight_layout()
fig.savefig(f"{OUT}/monk_report{SUFFIX}.png", dpi=150)