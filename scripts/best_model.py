import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

NAME = sys.argv[1] if len(sys.argv) > 1 else "model"
FOLDER = f"artifacts/{NAME}"
OUTER = re.compile(r"^outer(\d+)_m(\d+)\.csv$")


def summarize(path, metric):
    df = pd.read_csv(path)
    picks = [run.loc[run["train_error"].idxmin()] for _, run in df.groupby("trial")]
    values = pd.Series([p[metric] for p in picks])
    return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0, len(values)


paths = sorted(p for p in glob.glob(os.path.join(FOLDER, "outer*_m*.csv")) if OUTER.match(os.path.basename(p)))
if not paths:
    sys.exit(f"No retrained model logs found in {FOLDER}")

probe = pd.read_csv(paths[0])
classification = probe["test_acc"].abs().max() > 0
metric, label, higher_is_better = ("test_acc", "Test Accuracy (%)", True) if classification else ("test_mee", "Test MEE", False)

rows = []
for path in paths:
    match = OUTER.match(os.path.basename(path))
    mean, sd, trials = summarize(path, metric)
    rows.append({"fold": int(match.group(1)), "model": int(match.group(2)),
                 "value": mean, "sd": sd, "trials": trials})

results = pd.DataFrame(rows).sort_values("fold")
labels = [f"Fold {f}" for f in results["fold"]]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, results["value"], yerr=results["sd"], capsize=6,
               color="#4C72B0", edgecolor="black", linewidth=1.2, alpha=0.85)

fmt = "{:.1f}%" if classification else "{:.3f}"
for bar, model, value, sd in zip(bars, results["model"], results["value"], results["sd"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sd, f"Model {model}\n" + fmt.format(value),
             ha="center", va="bottom", fontweight="bold", fontsize=10)

average = results["value"].mean()
plt.axhline(average, color="#C44E52", linestyle="--", linewidth=2.5,
            label=("Average: " + fmt).format(average))

headroom = 1.18 * (results["value"] + results["sd"]).max()
plt.ylim(0, 115 if classification else headroom)
plt.ylabel(label, fontsize=12)
direction = "higher is better" if higher_is_better else "lower is better"
plt.title(f"{NAME.upper()} Performance  ({results['trials'].max()} trials per fold, {direction})",
          fontsize=14, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
if len(results) > 1:
    plt.legend(loc="upper right", fontsize=11)
plt.tight_layout()
plt.show()
