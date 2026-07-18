import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 5))

def update(frame):
    ax.clear()
    try:
        df = pd.read_csv("build/loss_log.csv")
        
        ax.plot(df['epoch'], df['loss'], linewidth=2, color='blue', label='Loss')
        ax.set_ylim(bottom=0)
        ax.set_title("Live Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        pass

ani = FuncAnimation(fig, update, interval=50)
plt.show()