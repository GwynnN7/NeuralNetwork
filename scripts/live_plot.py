import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons

FOLDER = f"artifacts/{sys.argv[1] if len(sys.argv) > 1 else 'model'}"
NAME = sys.argv[1] if len(sys.argv) > 1 else "model"
INNER = re.compile(r"^outer(\d+)_inner_m(\d+)\.csv$")
OUTER = re.compile(r"^outer(\d+)_m(\d+)\.csv$")
METRICS = ["train_error", "test_error", "train_mse", "test_mse",
           "train_mee", "test_mee", "train_acc", "test_acc"]

VIEWS = {
    "Loss": ("error", "Loss", (0, None), "{:.3f}", "upper right"),
    "MSE": ("mse", "MSE", (0, None), "{:.3f}", "upper right"),
    "MEE": ("mee", "MEE", (0, None), "{:.3f}", "upper right"),
    "Accuracy": ("acc", "Accuracy (%)", (None, 101), "{:.1f}%", "lower right"),
}
CLASSIFICATION_ONLY = {"Accuracy"}

TRAIN = ("purple", "-", "o")
VALID = ("orange", "--", "s")
TEST = ("blue", "-.", "^")
BAND_ALPHA = 0.18
MARKERS_PER_LINE = 12  # Markers are spaced out so they identify the line without hiding it


def load(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


class Runs:
    DIVERGED = 1e6

    def __init__(self, df, retrain):
        pivots = {c: df.pivot_table(index="epoch", columns=["fold", "trial"], values=c) for c in METRICS}
        reference = pivots["train_error"]
        valid = [c for c in reference.columns
                 if np.isfinite(reference[c].dropna()).all() and reference[c].abs().max() < Runs.DIVERGED]
        self.dropped = len(reference.columns) - len(valid)
        if valid:
            pivots = {c: p[valid] for c, p in pivots.items()}
        live = next(iter(pivots.values())).notna().sum(axis=1)
        self.count = int(live.max())
        filled = {c: p.ffill() for c, p in pivots.items()}
        self.mean = pd.DataFrame({c: p.mean(axis=1) for c, p in filled.items()}).reset_index()
        self.sd = pd.DataFrame({c: p.std(axis=1, ddof=1) for c, p in filled.items()}).reset_index()
        complete = (live == self.count).to_numpy()
        if retrain and complete.any():
            self.mean = self.mean[complete]
            self.sd = self.sd[complete]
        elif not retrain:
            enough = (live >= max(1, self.count / 2)).to_numpy()
            if enough.any():
                self.mean = self.mean[enough]
                self.sd = self.sd[enough]
        self.folds = df["fold"].nunique()
        self.trials = df["trial"].nunique()
        self.best = self.mean["train_error" if retrain else "test_error"].idxmin()

    def draw(self, ax, col, style, label, mark=None):
        color, linestyle, marker = style
        epoch = self.mean["epoch"]
        positive = self.mean[col][self.mean[col] > 0]
        floor = positive.min() * 1e-3 if len(positive) else 1e-12
        ax.fill_between(epoch, (self.mean[col] - self.sd[col]).clip(lower=floor), self.mean[col] + self.sd[col],
                        color=color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(epoch, self.mean[col], color=color, linewidth=1.5,
                linestyle=linestyle, label=label,
                marker=marker, markersize=5, markevery=max(1, len(epoch) // MARKERS_PER_LINE))
        if mark is None:
            return
        x, y = self.mean.loc[self.best, "epoch"], self.mean.loc[self.best, col]
        ax.axvline(x, color=color, linestyle=":", alpha=0.6)
        ax.scatter(x, y, color=color, zorder=5)
        ax.annotate(mark.format(y), (x, y), textcoords="offset points",
                    xytext=(5, 5), color=color, fontweight="bold")


class Viewer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 7))
        self.ax = self.fig.add_axes([0.09, 0.16, 0.885, 0.75])
        self.outers, self.models, self.selected = [], {}, {}
        self.outer_i, self.model_i, self.log_scale = 0, 0, False
        self.view, self.classification, self.syncing = "Loss", True, False
        self.user_choice = False

        def button(x, label, action, **kw):
            b = Button(plt.axes([x, 0.04, 0.11, 0.05]), label, **kw)
            b.on_clicked(action)
            return b

        self.prev_fold = button(0.16, "< Prev Fold", self.step(outer=-1))
        self.next_fold = button(0.28, "Next Fold >", self.step(outer=1))
        self.goto_best = button(0.435, "Best Model", self.jump_to_best, color="gold", hovercolor="yellow")
        self.prev_model = button(0.60, "< Prev Model", self.step(model=-1))
        self.next_model = button(0.72, "Next Model >", self.step(model=1))
        self.toggle = button(0.875, "Log Scale", self.flip_scale)

        radio_ax = plt.axes([0.008, 0.028, 0.115, 0.115], frameon=False)
        radio_ax.set_facecolor("none")
        self.radio = RadioButtons(radio_ax, list(VIEWS), active=list(VIEWS).index(self.view))
        self.radio.on_clicked(self.pick_view)

    def step(self, outer=0, model=0):
        def action(_):
            if outer:
                self.outer_i = max(0, min(len(self.outers) - 1, self.outer_i + outer))
                self.model_i = 0
            if model:
                self.model_i = max(0, min(len(self.models[self.outers[self.outer_i]]) - 1, self.model_i + model))
            self.refresh()
        return action

    def jump_to_best(self, _):
        outer = self.outers[self.outer_i]
        if outer in self.selected:
            self.model_i = self.models[outer].index(self.selected[outer])
            self.refresh()

    def flip_scale(self, _):
        self.log_scale = not self.log_scale
        self.refresh()

    def select_view(self, name):
        self.syncing = True
        self.view = name
        self.radio.set_active(list(VIEWS).index(name))
        self.syncing = False

    def pick_view(self, label):
        if self.syncing:
            return
        if not self.classification and label in CLASSIFICATION_ONLY:
            self.select_view(self.view)
            return
        self.view = label
        self.user_choice = True
        self.refresh()

    def scan(self):
        found, selected = {}, {}
        for path in glob.glob(os.path.join(FOLDER, "*.csv")):
            name = os.path.basename(path)
            outer_match = OUTER.match(name)
            match = INNER.match(name) or outer_match
            if not match:
                continue
            outer, model = int(match.group(1)), int(match.group(2))
            found.setdefault(outer, set()).add(model)
            if outer_match:
                selected[outer] = model
        self.outers = sorted(found)
        self.models = {o: sorted(m) for o, m in found.items()}
        self.selected = selected

    def detect_task(self, frames):
        self.classification = any(f[c].abs().max() > 0 for f in frames for c in ("train_acc", "test_acc"))
        if not self.classification and (self.view in CLASSIFICATION_ONLY or not self.user_choice):
            self.select_view("MEE")
        for text in self.radio.labels:
            greyed = not self.classification and text.get_text() in CLASSIFICATION_ONLY
            text.set_alpha(0.3 if greyed else 1.0)

    def enable_buttons(self, models):
        for btn, on in [(self.prev_fold, self.outer_i > 0),
                        (self.next_fold, self.outer_i < len(self.outers) - 1),
                        (self.prev_model, self.model_i > 0),
                        (self.next_model, self.model_i < len(models) - 1),
                        (self.goto_best, self.outers[self.outer_i] in self.selected)]:
            btn.set_active(on)
            btn.label.set_alpha(1.0 if on else 0.3)
        self.toggle.label.set_text("Linear Scale" if self.log_scale else "Log Scale")

    def style(self, title):
        _, ylabel, (bottom, top), _, loc = VIEWS[self.view]
        self.fig.suptitle(title, fontsize=13, fontweight="bold")
        self.ax.set_title(f"{self.view} (log scale)" if self.log_scale else self.view, fontsize=11)
        if self.log_scale:
            self.ax.set_yscale("log", nonpositive="mask")
            values = [v for line in self.ax.get_lines()
                      if line.get_label() and not line.get_label().startswith("_")
                      for v in line.get_ydata() if v > 0]
            if values:
                self.ax.set_ylim(min(values) * 0.7, max(values) * 1.4)
            else:
                self.ax.autoscale(axis="y")
        else:
            self.ax.set_yscale("linear")
            if bottom is not None:
                self.ax.set_ylim(bottom=bottom)
            if top is not None:
                self.ax.set_ylim(top=top)
        self.ax.grid(True, which="both", linestyle="--", alpha=0.7)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel("Epoch")
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc=loc)

    def refresh(self, _=None):
        self.scan()
        self.ax.clear()
        if not self.outers:
            self.style("Waiting for data...")
            return

        self.outer_i = min(self.outer_i, len(self.outers) - 1)
        outer = self.outers[self.outer_i]
        models = self.models[outer]
        self.model_i = min(self.model_i, len(models) - 1)
        model = models[self.model_i]
        self.enable_buttons(models)

        inner_df = load(os.path.join(FOLDER, f"outer{outer}_inner_m{model}.csv"))
        outer_df = load(os.path.join(FOLDER, f"outer{outer}_m{model}.csv"))
        self.detect_task([f for f in (inner_df, outer_df) if f is not None])
        key, _, _, fmt, _ = VIEWS[self.view]

        header = f"Outer Fold: {self.outer_i + 1}/{len(self.outers)}  |  Model: {model}"
        title = header

        if inner_df is not None:
            runs = Runs(inner_df, retrain=False)
            if outer_df is None:
                runs.draw(self.ax, f"train_{key}", TRAIN, f"Training {self.view}")
            runs.draw(self.ax, f"test_{key}", VALID, f"Validation {self.view}",
                      mark=None if outer_df is not None else fmt)
            spread = f"{runs.folds} Folds" + (f" x {runs.trials} Trials" if runs.trials > 1 else "")
            title = f"Grid Search ({spread}):  {header}"

        if outer_df is not None:
            runs = Runs(outer_df, retrain=True)
            runs.draw(self.ax, f"train_{key}", TRAIN, f"Training {self.view}")
            runs.draw(self.ax, f"test_{key}", TEST, f"Test {self.view}", mark=fmt)
            trials = f" ({runs.trials} Trials)" if runs.trials > 1 else ""
            title = f"Best Model Evaluation{trials}:  {header}"

        title = f"{NAME.upper()}: {title}"
        self.style(title)


if __name__ == "__main__":
    viewer = Viewer()
    animation = FuncAnimation(viewer.fig, viewer.refresh, interval=1000, cache_frame_data=False)
    plt.show()
