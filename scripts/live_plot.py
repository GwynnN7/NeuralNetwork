import re
import pandas as pd
import sys
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

folder_path = f"artifacts/{sys.argv[1]}" if len(sys.argv) > 1 else "artifacts/model"
fig = plt.figure(figsize=(10, 8))

def load_csv(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame()

def update(frame):
    try:
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        fig.clf()

        if not csv_files:
            ax = fig.add_subplot(111)
            ax.set_title("Waiting for data...")
            return

        file_info = []
        outer_ids = set()

        for f in csv_files:
            basename = os.path.basename(f)
            match_inner = re.match(r'^outer(\d+)_inner_m(\d+)\.csv$', basename)
            match_outer = re.match(r'^outer(\d+)_m(\d+)\.csv$', basename)
            if match_inner:
                o_id = int(match_inner.group(1))
                outer_ids.add(o_id)
                file_info.append({'path': f, 'outer': o_id, 'model': match_inner.group(2), 'mtime': os.path.getmtime(f)})
            elif match_outer:
                o_id = int(match_outer.group(1))
                outer_ids.add(o_id)
                file_info.append({'path': f, 'outer': o_id, 'model': match_outer.group(2), 'mtime': os.path.getmtime(f)})

        sorted_outer_ids = sorted(list(outer_ids))
        num_cols = len(sorted_outer_ids)

        if num_cols == 0:
            ax = fig.add_subplot(111)
            ax.set_title("Unrecognized file formats...")
            return

        axs = fig.subplots(2, num_cols, squeeze=False)

        for col_idx, o_id in enumerate(sorted_outer_ids):
            ax1 = axs[0, col_idx] 
            ax2 = axs[1, col_idx] 

            files_for_oid = [info for info in file_info if info['outer'] == o_id]
            latest_file = max(files_for_oid, key=lambda x: x['mtime'])
            
            model_id = latest_file['model']
            
            target_inner = os.path.join(folder_path, f"outer{o_id}_inner_m{model_id}.csv")
            target_outer = os.path.join(folder_path, f"outer{o_id}_m{model_id}.csv")

            inner_df = load_csv(target_inner)
            outer_df = load_csv(target_outer)

            has_inner = not inner_df.empty
            has_outer = not outer_df.empty

            model_name = f"Outer {o_id} | Model {model_id}"
            num_folds = 0


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

                ax1.set_title(f"[{model_name}]\nGrid Search ({num_folds} Folds)")

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

                ax1.set_title(f"[{model_name}]\nBest Model Evaluation")

            ax1.set_ylim(bottom=-0.05) 
            ax1.grid(True, linestyle='--', alpha=0.7)

            ax2.set_ylim(-5, 105) 
            ax2.set_xlabel("Epoch")
            ax2.grid(True, linestyle='--', alpha=0.7)

            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                ax1.legend(loc="upper right")

            handles, labels = ax2.get_legend_handles_labels()
            if handles:
                ax2.legend(loc="lower right")

            if col_idx == 0:
                ax1.set_ylabel("Loss")
                ax2.set_ylabel("Accuracy")

        plt.tight_layout()

    except Exception as e:
        print(f"Plotting error: {e}")

if __name__ == "__main__":
    anim = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show()