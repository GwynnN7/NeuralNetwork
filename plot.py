import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

file_path = f"build/{sys.argv[1]}" if len(sys.argv) > 1 else "build/log.csv"
fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    try:
        df = pd.read_csv(file_path)

        ax.plot(df['epoch'], df['train_loss'], linewidth=2, color='blue', label='Train Loss')
        ax.plot(df['epoch'], df['test_loss'], linewidth=2, color='red', label='Test Loss')
        ax.set_ylim(bottom=0)
        ax.set_title("Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        pass

if __name__ == "__main__":
    anim = FuncAnimation(fig, update, interval=50)
    plt.show()