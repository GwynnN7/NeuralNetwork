import re
import pandas as pd
import sys
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

folder_path = f"artifacts/{sys.argv[1]}" if len(sys.argv) > 1 else "artifacts/model"
fig = plt.figure(figsize=(12, 9))

app_state = {
    'outer_ids': [],
    'models_per_outer': {},
    'best_model_per_outer': {},
    'curr_outer_idx': 0,
    'curr_model_idx': 0,
    'buttons': [],
    'ax_loss': None,
    'ax_acc': None,
    'initialized': False
}

def load_csv(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame()

def prev_fold(event):
    if app_state['curr_outer_idx'] > 0:
        app_state['curr_outer_idx'] -= 1
        app_state['curr_model_idx'] = 0 
        update(0)

def next_fold(event):
    if app_state['curr_outer_idx'] < len(app_state['outer_ids']) - 1:
        app_state['curr_outer_idx'] += 1
        app_state['curr_model_idx'] = 0
        update(0)

def prev_model(event):
    if app_state['curr_model_idx'] > 0:
        app_state['curr_model_idx'] -= 1
        update(0)

def next_model(event):
    if not app_state['outer_ids']: return
    o_id = app_state['outer_ids'][app_state['curr_outer_idx']]
    if app_state['curr_model_idx'] < len(app_state['models_per_outer'][o_id]) - 1:
        app_state['curr_model_idx'] += 1
        update(0)

def goto_best(event):
    if not app_state['outer_ids']: return
    o_id = app_state['outer_ids'][app_state['curr_outer_idx']]
    best_m_id = app_state['best_model_per_outer'].get(o_id)

    if best_m_id is not None:
        models = app_state['models_per_outer'][o_id]
        if best_m_id in models:
            app_state['curr_model_idx'] = models.index(best_m_id)
            update(0)

def init_ui():
    app_state['initialized'] = True

    gs = fig.add_gridspec(2, 1, bottom=0.15, top=0.93, hspace=0.15) 

    app_state['ax_acc'] = fig.add_subplot(gs[1, 0])
    app_state['ax_loss'] = fig.add_subplot(gs[0, 0], sharex=app_state['ax_acc'])

    ax_prev_f = plt.axes([0.15, 0.04, 0.1, 0.05])
    ax_next_f = plt.axes([0.26, 0.04, 0.1, 0.05])

    ax_best   = plt.axes([0.45, 0.04, 0.1, 0.05])

    ax_prev_m = plt.axes([0.64, 0.04, 0.1, 0.05])
    ax_next_m = plt.axes([0.75, 0.04, 0.1, 0.05])

    btn_prev_f = Button(ax_prev_f, '< Prev Fold')
    btn_next_f = Button(ax_next_f, 'Next Fold >')

    btn_best = Button(ax_best, 'Best Model', color='gold', hovercolor='yellow')

    btn_prev_m = Button(ax_prev_m, '< Prev Model')
    btn_next_m = Button(ax_next_m, 'Next Model >')

    btn_prev_f.on_clicked(prev_fold)
    btn_next_f.on_clicked(next_fold)
    btn_best.on_clicked(goto_best)
    btn_prev_m.on_clicked(prev_model)
    btn_next_m.on_clicked(next_model)

    app_state['buttons'].extend([btn_prev_f, btn_next_f, btn_best, btn_prev_m, btn_next_m])

def update(frame):
    try:
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not csv_files:
            if not app_state['initialized']:
                ax = fig.add_subplot(111)
                ax.set_title("Waiting for data...")
            return

        if not app_state['initialized']:
            fig.clf()
            init_ui()

        models_per_outer = {}
        for f in csv_files:
            basename = os.path.basename(f)
            match_inner = re.match(r'^outer(\d+)_inner_m(\d+)\.csv$', basename)
            match_outer = re.match(r'^outer(\d+)_m(\d+)\.csv$', basename)

            if match_inner:
                o_id, m_id = int(match_inner.group(1)), match_inner.group(2)
                models_per_outer.setdefault(o_id, set()).add(m_id)
            elif match_outer:
                o_id, m_id = int(match_outer.group(1)), match_outer.group(2)
                models_per_outer.setdefault(o_id, set()).add(m_id)
                app_state['best_model_per_outer'][o_id] = m_id

        sorted_outer_ids = sorted(list(models_per_outer.keys()))
        app_state['outer_ids'] = sorted_outer_ids
        for o_id in sorted_outer_ids:
            app_state['models_per_outer'][o_id] = sorted(list(models_per_outer[o_id]), key=int)

        if not app_state['outer_ids']:
            return

        if app_state['curr_outer_idx'] >= len(app_state['outer_ids']):
            app_state['curr_outer_idx'] = len(app_state['outer_ids']) - 1

        o_id = app_state['outer_ids'][app_state['curr_outer_idx']]
        models = app_state['models_per_outer'][o_id]

        if app_state['curr_model_idx'] >= len(models):
            app_state['curr_model_idx'] = len(models) - 1

        m_id = models[app_state['curr_model_idx']]

        if len(app_state['buttons']) == 5:
            btn_prev_f, btn_next_f, btn_best, btn_prev_m, btn_next_m = app_state['buttons']

            def toggle_btn(btn, condition):
                btn.set_active(condition)
                btn.label.set_alpha(1.0 if condition else 0.3)

            toggle_btn(btn_prev_f, app_state['curr_outer_idx'] > 0)
            toggle_btn(btn_next_f, app_state['curr_outer_idx'] < len(app_state['outer_ids']) - 1)

            toggle_btn(btn_prev_m, app_state['curr_model_idx'] > 0)
            toggle_btn(btn_next_m, app_state['curr_model_idx'] < len(models) - 1)

            best_m_id = app_state['best_model_per_outer'].get(o_id)
            toggle_btn(btn_best, best_m_id is not None)

        target_inner = os.path.join(folder_path, f"outer{o_id}_inner_m{m_id}.csv")

        target_inner = os.path.join(folder_path, f"outer{o_id}_inner_m{m_id}.csv")
        target_outer = os.path.join(folder_path, f"outer{o_id}_m{m_id}.csv")

        inner_df = load_csv(target_inner)
        outer_df = load_csv(target_outer)

        has_inner = not inner_df.empty
        has_outer = not outer_df.empty

        model_name = f"Outer Fold: {o_id + 1}/{len(app_state['outer_ids'])}  |  Model: {m_id}"
        num_folds = 0

        ax1 = app_state['ax_loss']
        ax2 = app_state['ax_acc']

        ax1.clear()
        ax2.clear()

        if has_inner:
            inner_df['fold_id'] = (inner_df['epoch'] == 0).cumsum()
            num_folds = inner_df['fold_id'].max()
            plot_df = inner_df.groupby('epoch').mean().reset_index()

            if 'train_loss' in plot_df.columns:
                ax1.plot(plot_df['epoch'], plot_df['train_loss'], linewidth=1.5, linestyle='--', color='purple', label='Training Loss')
                ax1.plot(plot_df['epoch'], plot_df['test_loss'], linewidth=1.5, color='orange', label='Validation Loss')

                if not plot_df['test_loss'].isnull().all() and not has_outer:
                    best_idx = plot_df['test_loss'].idxmin()
                    best_ep = plot_df.loc[best_idx, 'epoch']
                    best_val = plot_df.loc[best_idx, 'test_loss']
                    ax1.axvline(x=best_ep, color='orange', linestyle=':', alpha=0.6)
                    ax1.scatter(best_ep, best_val, color='orange', zorder=5)
                    ax1.annotate(f"{best_val:.3f}", (best_ep, best_val), textcoords="offset points", xytext=(5,5), color='orange', fontweight='bold')

            if 'train_acc' in plot_df.columns:
                ax2.plot(plot_df['epoch'], plot_df['train_acc'], linewidth=1.5, linestyle='--', color='purple', label='Training Accuracy')
                ax2.plot(plot_df['epoch'], plot_df['test_acc'], linewidth=1.5, color='orange', label='Validation Accuracy')

                if not plot_df['test_acc'].isnull().all() and not has_outer:
                    best_idx = plot_df['test_acc'].idxmax()
                    best_ep = plot_df.loc[best_idx, 'epoch']
                    best_val = plot_df.loc[best_idx, 'test_acc']
                    ax2.axvline(x=best_ep, color='orange', linestyle=':', alpha=0.6)
                    ax2.scatter(best_ep, best_val, color='orange', zorder=5)
                    ax2.annotate(f"{best_val:.1f}%", (best_ep, best_val), textcoords="offset points", xytext=(5,-12), color='orange', fontweight='bold')

            ax1.set_title(f"Grid Search ({num_folds} Folds):  {model_name}")

        if has_outer:
            if 'train_loss' in outer_df.columns:
                if not has_inner: 
                    ax1.plot(outer_df['epoch'], outer_df['train_loss'], color='purple', linestyle='--', linewidth=1.5, label='Train Loss')
                ax1.plot(outer_df['epoch'], outer_df['test_loss'], color='blue', linewidth=1.5, label='Test Loss')

                if not outer_df['test_loss'].isnull().all():
                    best_idx = outer_df['test_loss'].idxmin()
                    best_ep = outer_df.loc[best_idx, 'epoch']
                    best_val = outer_df.loc[best_idx, 'test_loss']
                    ax1.axvline(x=best_ep, color='blue', linestyle=':', alpha=0.6)
                    ax1.scatter(best_ep, best_val, color='blue', zorder=5)
                    ax1.annotate(f"{best_val:.3f}", (best_ep, best_val), textcoords="offset points", xytext=(5,5), color='blue', fontweight='bold')

            if 'train_acc' in outer_df.columns:
                if not has_inner: 
                    ax2.plot(outer_df['epoch'], outer_df['train_acc'], color='purple', linestyle='--', linewidth=1.5, label='Train Accuracy')
                ax2.plot(outer_df['epoch'], outer_df['test_acc'], color='blue', linewidth=1.5, label='Test Accuracy')

                if not outer_df['test_acc'].isnull().all():
                    best_idx = outer_df['test_acc'].idxmax()
                    best_ep = outer_df.loc[best_idx, 'epoch']
                    best_val = outer_df.loc[best_idx, 'test_acc']
                    ax2.axvline(x=best_ep, color='blue', linestyle=':', alpha=0.6)
                    ax2.scatter(best_ep, best_val, color='blue', zorder=5)
                    ax2.annotate(f"{best_val:.1f}%", (best_ep, best_val), textcoords="offset points", xytext=(5,-12), color='blue', fontweight='bold')

            ax1.set_title(f" Best Model Evaluation:  {model_name}")

        ax1.set_ylim(bottom=-0.05)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylabel("Loss")

        ax2.set_ylim(-5, 105)
        ax2.set_xlabel("Epoch")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylabel("Accuracy")

        handles1, _ = ax1.get_legend_handles_labels()
        if handles1: ax1.legend(loc="upper right")

        handles2, _ = ax2.get_legend_handles_labels()
        if handles2: ax2.legend(loc="lower right")

    except Exception as e:
        print(f"Plotting error: {e}")

if __name__ == "__main__":
    anim = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show()