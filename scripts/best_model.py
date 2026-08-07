import pandas as pd
import sys
import os
import glob
import re
import matplotlib.pyplot as plt

model_name = sys.argv[1] if len(sys.argv) > 1 else "model"
folder_path = f"artifacts/{model_name}"
files = glob.glob(os.path.join(folder_path, "outer*_m*.csv"))

outer_files = [
    f for f in files 
    if re.match(r'^outer(\d+)_m(\d+)\.csv$', os.path.basename(f))
]

if not outer_files:
    print(f"No outer fold files found in {folder_path}")
    sys.exit()

results = []

for file in outer_files:
    basename = os.path.basename(file)

    match = re.search(r'^outer(\d+)_m(\d+)\.csv$', basename)

    outer_idx = int(match.group(1))
    model_idx = int(match.group(2))

    df = pd.read_csv(file)
    if not df.empty:
        final_epoch = df.iloc[-1]
        results.append({
            'Outer Fold': f"Fold {outer_idx}",
            'Sort Index': outer_idx,
            'Model ID': model_idx,
            'Test Acc': final_epoch.get('test_acc', 0.0),
            'Test Error': final_epoch.get('test_error', 0.0)
        })

if not results:
    print(f"Files could not be parsed correctly")
    sys.exit()

results_df = pd.DataFrame(results).sort_values('Sort Index')

plt.figure(figsize=(10, 6))
bars = plt.bar(
    results_df['Outer Fold'], 
    results_df['Test Acc'], 
    color='#4C72B0', 
    edgecolor='black', 
    linewidth=1.2,
    alpha=0.85
)

for bar, model_id, acc in zip(bars, results_df['Model ID'], results_df['Test Acc']):
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        yval + 2, 
        f"Model {model_id}\n({acc:.1f}%)", 
        ha='center', 
        va='bottom', 
        fontweight='bold',
        fontsize=10
    )

grand_average = results_df['Test Acc'].mean()
plt.axhline(
    grand_average, 
    color='#C44E52', 
    linestyle='--', 
    linewidth=2.5, 
    label=f'Average: {grand_average:.2f}%'
)

plt.ylim(0, 115)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.title(f"{model_name.upper()} Performance", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

if len(results_df) > 1:
    plt.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.show()