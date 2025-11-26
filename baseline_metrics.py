import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory with classification report CSVs
report_dir = "baseline_performance_report"

# Metrics to visualize
metrics = ["accuracy", "precision", "recall", "f1-score"]

# Dictionary to collect scores
model_scores = {metric: {} for metric in metrics}

# Loop through classification reports
for filename in os.listdir(report_dir):
    if filename.endswith("_classification_report.csv"):
        model_name = filename.replace("_classification_report.csv", "").replace("_", " ")
        df = pd.read_csv(os.path.join(report_dir, filename), index_col=0)

        # Accuracy (stored as a separate row)
        if "accuracy" in df.index:
            model_scores["accuracy"][model_name] = df.loc["accuracy", "precision"]
        else:
            print(f"Warning: 'accuracy' not found in {filename}")

        # Macro avg for precision, recall, f1-score
        for metric in ["precision", "recall", "f1-score"]:
            if "macro avg" in df.index:
                model_scores[metric][model_name] = df.loc["macro avg", metric]
            else:
                print(f"Warning: 'macro avg' not found for {metric} in {filename}")

# Plot vertical bar charts
for metric, scores in model_scores.items():
    models = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color='cornflowerblue', edgecolor='black')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Comparison Among Models')
    plt.ylim(0, 1)

    # Annotate each bar with the score
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.2f}", ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"{metric}.png"))
    plt.show()
