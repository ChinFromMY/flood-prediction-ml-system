import pickle
from tabulate import tabulate

# Baseline metrics from your previous runs
baseline_metrics = {
    "KNN": {"Accuracy": 0.9175, "F1": 0.7968, "Precision": 0.8394, "Recall": 0.7583},
    "Logistic Regression": {"Accuracy": 0.8812, "F1": 0.7663, "Precision": 0.6599, "Recall": 0.9135},
    "SVM": {"Accuracy": 0.8817, "F1": 0.7646, "Precision": 0.6642, "Recall": 0.9008},
    "Decision Tree": {"Accuracy": 0.8899, "F1": 0.7717, "Precision": 0.6915, "Recall": 0.8728},
    "Random Forest": {"Accuracy": 0.8931, "F1": 0.7814, "Precision": 0.6929, "Recall": 0.8957}
}

# Load nested CV results
output_dir = "tuning_model"
with open(f"{output_dir}/nested_cv_results.pkl", "rb") as f:
    nested_cv_results = pickle.load(f)

# Prepare comparison table
table = []
for model in baseline_metrics.keys():
    nested = nested_cv_results.get(model, {})
    row = [
        model,
        baseline_metrics[model]["Accuracy"],
        round(nested.get('mean_accuracy', 0), 4),
        baseline_metrics[model]["F1"],
        round(nested.get('mean_f1', 0), 4),
        baseline_metrics[model]["Precision"],
        round(nested.get('mean_precision', 0), 4),
        baseline_metrics[model]["Recall"],
        round(nested.get('mean_recall', 0), 4)
    ]
    table.append(row)

# Print nicely
headers = ["Model", "Baseline Accuracy", "Nested CV Accuracy", "Baseline F1", "Nested CV F1",
           "Baseline Precision", "Nested CV Precision", "Baseline Recall", "Nested CV Recall"]

print(tabulate(table, headers=headers, tablefmt="github"))
