import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "ml vs dl"
os.makedirs(output_dir, exist_ok=True)

# Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
ml_scores = [0.9485, 0.8147, 0.9682, 0.8848]
dl_scores = [0.962, 0.882, 0.936, 0.909]

# X-axis positions
x = np.arange(len(metrics))
width = 0.35

# Plot
plt.figure(figsize=(8, 6))
bars_ml = plt.bar(x - width/2, ml_scores, width, label='Machine Learning', color='skyblue')
bars_dl = plt.bar(x + width/2, dl_scores, width, label='Deep Learning', color='salmon')

# Add value labels
for bar in bars_ml:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}", ha='center', fontsize=9)

for bar in bars_dl:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}", ha='center', fontsize=9)

# Labels and formatting
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Machine Learning vs Deep Learning')
plt.xticks(x, metrics)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(output_dir, "ml_vs_dl_performance_comparison.png"))
plt.show()



