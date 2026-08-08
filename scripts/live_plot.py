import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

FOLDER = f"artifacts/{sys.argv[1] if len(sys.argv) > 1 else 'model'}"
INNER = re.compile(r"^outer(\d+)_inner_m(\d+)\.csv$")
OUTER = re.compile(r"^outer(\d+)_m(\d+)\.csv$")
METRICS = ["train_error", "test_error", "train_mse", "test_mse", "train_acc", "test_acc"]

TRAIN_COLOR = "purple"
VALID_COLOR = "orange"
TEST_COLOR = "blue"
BAND_LEVELS = [1.0, 0.75, 0.5, 0.25]
BAND_ALPHA = 0.05


def load(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


class Runs:
    def __init__(self, df, retrain):
        pivots = {c: df.pivot_table(index="epoch", columns=["fold", "trial"], values=c) for c in METRICS}
        live = next(iter(pivots.values())).notna().sum(axis=1)
        self.count = int(live.max())
        self.mean = pd.DataFrame({c: p.ffill().mean(axis=1) for c, p in pivots.items()}).reset_index()
        self.sd = pd.DataFrame({c: p.std(axis=1, ddof=1) for c, p in pivots.items()}).reset_index()
        complete = (live == self.count).to_numpy()
        if retrain and complete.any():
            self.mean = self.mean[complete]
            self.sd = self.sd[complete]
            live = live[complete]
        self.live = live.to_numpy()
        self.folds = df["fold"].nunique()
        self.trials = df["trial"].nunique()
        self.best = self.mean["train_error" if retrain else "test_error"].idxmin()

    def draw(self, ax, col, color, label, dashed=False, mark=None):
        epoch = self.mean["epoch"]
        low, high = self.mean[col] - self.sd[col], self.mean[col] + self.sd[col]
        for level in BAND_LEVELS:
            ax.fill_between(epoch, low, high, where=self.live >= max(2, int(self.count * level)),
                            color=color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(epoch, self.mean[col], color=color, linewidth=1.5,
                linestyle="--" if dashed else "-", label=label)
        if mark is None:
            return
        x, y = self.mean.loc[self.best, "epoch"], self.mean.loc[self.best, col]
        ax.axvline(x, color=color, linestyle=":", alpha=0.6)
        ax.scatter(x, y, color=color, zorder=5)
        ax.annotate(mark.format(y), (x, y), textcoords="offset points",
                    xytext=(5, 5), color=color, fontweight="bold")


class Viewer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 7))
        grid = self.fig.add_gridspec(1, 2, left=0.055, right=0.99, bottom=0.16, top=0.91, wspace=0.14)
        self.ax_error = self.fig.add_subplot(grid[0, 0])
        self.ax_acc = self.fig.add_subplot(grid[0, 1], sharex=self.ax_error)
        self.outers, self.models, self.selected = [], {}, {}
        self.outer_i, self.model_i, self.log_scale = 0, 0, False

        def button(x, label, action, **kw):
            b = Button(plt.axes([x, 0.04, 0.1, 0.05]), label, **kw)
            b.on_clicked(action)
            return b

        self.prev_fold = button(0.15, "< Prev Fold", self.step(outer=-1))
        self.next_fold = button(0.26, "Next Fold >", self.step(outer=1))
        self.goto_best = button(0.45, "Best Model", self.jump_to_best, color="gold", hovercolor="yellow")
        self.prev_model = button(0.64, "< Prev Model", self.step(model=-1))
        self.next_model = button(0.75, "Next Model >", self.step(model=1))
        self.toggle = button(0.885, "Log Error", self.flip_scale)

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

    def enable_buttons(self, models):
        for btn, on in [(self.prev_fold, self.outer_i > 0),
                        (self.next_fold, self.outer_i < len(self.outers) - 1),
                        (self.prev_model, self.model_i > 0),
                        (self.next_model, self.model_i < len(models) - 1),
                        (self.goto_best, self.outers[self.outer_i] in self.selected)]:
            btn.set_active(on)
            btn.label.set_alpha(1.0 if on else 0.3)
        self.toggle.label.set_text("Linear Error" if self.log_scale else "Log Error")

    def style(self, title):
        self.fig.suptitle(title, fontsize=13, fontweight="bold")
        self.ax_error.set_title("Error (log scale)" if self.log_scale else "Error", fontsize=11)
        if self.log_scale:
            self.ax_error.set_yscale("log", nonpositive="mask")
            self.ax_error.autoscale(axis="y")
        else:
            self.ax_error.set_yscale("linear")
            self.ax_error.set_ylim(bottom=0)
        self.ax_acc.set_title("Accuracy", fontsize=11)
        self.ax_acc.set_ylim(top=101)
        for ax, ylabel, loc in [(self.ax_error, "Error", "upper right"), (self.ax_acc, "Accuracy (%)", "lower right")]:
            ax.grid(True, which="both", linestyle="--", alpha=0.7)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc=loc)

    def refresh(self, _=None):
        self.scan()
        self.ax_error.clear()
        self.ax_acc.clear()
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
        header = f"Outer Fold: {self.outer_i + 1}/{len(self.outers)}  |  Model: {model}"
        title = header

        if inner_df is not None:
            runs = Runs(inner_df, retrain=False)
            runs.draw(self.ax_error, "train_mse", TRAIN_COLOR, "Training Error", dashed=True)
            runs.draw(self.ax_error, "test_mse", VALID_COLOR, "Validation Error",
                      mark=None if outer_df is not None else "{:.3f}")
            runs.draw(self.ax_acc, "train_acc", TRAIN_COLOR, "Training Accuracy", dashed=True)
            runs.draw(self.ax_acc, "test_acc", VALID_COLOR, "Validation Accuracy",
                      mark=None if outer_df is not None else "{:.1f}%")
            spread = f"{runs.folds} Folds" + (f" x {runs.trials} Trials" if runs.trials > 1 else "")
            title = f"Grid Search ({spread}):  {header}"

        if outer_df is not None:
            runs = Runs(outer_df, retrain=True)
            if inner_df is None:
                runs.draw(self.ax_error, "train_mse", TRAIN_COLOR, "Train Error", dashed=True)
                runs.draw(self.ax_acc, "train_acc", TRAIN_COLOR, "Train Accuracy", dashed=True)
            runs.draw(self.ax_error, "test_mse", TEST_COLOR, "Test Error", mark="{:.3f}")
            runs.draw(self.ax_acc, "test_acc", TEST_COLOR, "Test Accuracy", mark="{:.1f}%")
            trials = f" ({runs.trials} Trials)" if runs.trials > 1 else ""
            title = f"Best Model Evaluation{trials}:  {header}"

        self.style(title)


if __name__ == "__main__":
    viewer = Viewer()
    animation = FuncAnimation(viewer.fig, viewer.refresh, interval=1000, cache_frame_data=False)
    plt.show()
