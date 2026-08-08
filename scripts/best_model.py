import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

NAME = sys.argv[1] if len(sys.argv) > 1 else "model"
FOLDER = f"artifacts/{NAME}"
OUTER = re.compile(r"^outer(\d+)_m(\d+)\.csv$")


def summarize(path):
    df = pd.read_csv(path)
    picks = [run.loc[run["train_error"].idxmin()] for _, run in df.groupby("trial")]
    acc = pd.Series([p["test_acc"] for p in picks])
    return acc.mean(), acc.std(ddof=1) if len(acc) > 1 else 0.0, len(acc)


rows = []
for path in sorted(glob.glob(os.path.join(FOLDER, "outer*_m*.csv"))):
    match = OUTER.match(os.path.basename(path))
    if not match:
        continue
    mean, sd, trials = summarize(path)
    rows.append({"fold": int(match.group(1)), "model": int(match.group(2)),
                 "acc": mean, "sd": sd, "trials": trials})

if not rows:
    sys.exit(f"No retrained model logs found in {FOLDER}")

results = pd.DataFrame(rows).sort_values("fold")
labels = [f"Fold {f}" for f in results["fold"]]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, results["acc"], yerr=results["sd"], capsize=6,
               color="#4C72B0", edgecolor="black", linewidth=1.2, alpha=0.85)

for bar, model, acc, sd in zip(bars, results["model"], results["acc"], results["sd"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sd + 2,
             f"Model {model}\n{acc:.1f}% ± {sd:.1f}", ha="center", va="bottom",
             fontweight="bold", fontsize=10)

average = results["acc"].mean()
plt.axhline(average, color="#C44E52", linestyle="--", linewidth=2.5,
            label=f"Average: {average:.2f}%")

plt.ylim(0, 115)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.title(f"{NAME.upper()} Performance  ({results['trials'].max()} trials per fold)",
          fontsize=14, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
if len(results) > 1:
    plt.legend(loc="upper right", fontsize=11)
plt.tight_layout()
plt.show()