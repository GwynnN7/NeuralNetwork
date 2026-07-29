import pandas as pd
import sys
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

folder_path = f"artifacts/{sys.argv[1]}" if len(sys.argv) > 1 else "artifacts/model"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

def update(frame):
    ax1.clear()
    ax2.clear()
    try:
        # Get all log files in the directory
        log_files = glob.glob(os.path.join(folder_path, "*_log.csv"))
        if not log_files:
            return

        # Read and plot every individual fold with low opacity
        dataframes = []
        for file in log_files:
            df = pd.read_csv(file)
            dataframes.append(df)

            if 'train_loss' in df.columns and 'test_loss' in df.columns:
                ax1.plot(df['epoch'], df['train_loss'], color='blue', alpha=0.15)
                ax1.plot(df['epoch'], df['test_loss'], color='red', alpha=0.15)

            if 'train_acc' in df.columns and 'test_acc' in df.columns:
                ax2.plot(df['epoch'], df['train_acc'], color='green', alpha=0.15)
                ax2.plot(df['epoch'], df['test_acc'], color='orange', alpha=0.15)

        # Calculate the average across all folds
        combined_df = pd.concat(dataframes)
        mean_df = combined_df.groupby('epoch').mean().reset_index()

        # Plot the thick average lines
        if 'train_loss' in mean_df.columns:
            ax1.plot(mean_df['epoch'], mean_df['train_loss'], linewidth=2, color='blue', label='Avg Train Loss')
            ax1.plot(mean_df['epoch'], mean_df['test_loss'], linewidth=2, color='red', label='Avg Test Loss')

        if 'train_acc' in mean_df.columns:
            ax2.plot(mean_df['epoch'], mean_df['train_acc'], linewidth=2, color='green', label='Avg Train Accuracy')
            ax2.plot(mean_df['epoch'], mean_df['test_acc'], linewidth=2, color='orange', label='Avg Test Accuracy')

        # Formatting ax1 (Loss)
        ax1.set_ylim(bottom=0)
        ax1.set_title(f"Loss Curves ({len(log_files)} folds)")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Formatting ax2 (Accuracy)
        ax2.set_ylim(0, 100) 
        ax2.set_title(f"Accuracy Curves ({len(log_files)} folds)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

    except Exception as e:
        pass

if __name__ == "__main__":
    anim = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show()