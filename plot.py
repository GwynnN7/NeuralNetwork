import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

file_path = f"build/{sys.argv[1]}" if len(sys.argv) > 1 else "build/log.csv"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

def update(frame):
    ax1.clear()
    ax2.clear()
    try:
        df = pd.read_csv(file_path)

        if 'train_loss' in df.columns and 'test_loss' in df.columns:
            ax1.plot(df['epoch'], df['train_loss'], linewidth=2, color='blue', label='Train Loss')
            ax1.plot(df['epoch'], df['test_loss'], linewidth=2, color='red', label='Test Loss')

        ax1.set_ylim(bottom=0)
        ax1.set_title("Loss Curves")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle='--', alpha=0.7)

        if 'train_acc' in df.columns and 'test_acc' in df.columns:
            ax2.plot(df['epoch'], df['train_acc'], linewidth=2, color='green', label='Train Accuracy')
            ax2.plot(df['epoch'], df['test_acc'], linewidth=2, color='orange', label='Test Accuracy')

        ax2.set_ylim(0, 100) 
        ax2.set_title("Accuracy Curves")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

    except Exception as e:
        pass

if __name__ == "__main__":
    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()