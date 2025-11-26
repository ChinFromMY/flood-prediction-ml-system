import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# random forest baseline model vs rf final model 
# accuracy, precision, recall and f1 score
# auc pr curve

output_dir = "base vs final"
os.makedirs(output_dir, exist_ok = True)

# Load base model
with open("baseline_pkl/Random Forest_model.pkl", "rb") as f:     
    base_model = pickle.load(f)

# Load final model
with open("random_forest_final_model/random_forest_model.pkl", "rb") as f:    
    final_model = pickle.load(f)

# Load the already split dataset
with open("baseline_pkl/full_data_splits.pkl", "rb") as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

# Predictions
y_pred_base = base_model.predict(X_test)
y_pred_final = final_model.predict(X_test)

# Metrics
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }

metrics_base = compute_metrics(y_test, y_pred_base)
metrics_final = compute_metrics(y_test, y_pred_final)


# Prepare metrics
metric_names = list(metrics_base.keys())
base_values = list(metrics_base.values())
final_values = list(metrics_final.values())

# Set position of bar on X axis
x = np.arange(len(metric_names))  # label locations
width = 0.35  # width of the bars

# Create plot
plt.figure(figsize=(8, 6))
bars_base = plt.bar(x - width/2, base_values, width=width, label='Base Model', color='skyblue')
bars_final = plt.bar(x + width/2, final_values, width=width, label='Final Model', color='orange')

# Add text labels on top of bars
for bar in bars_base:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

for bar in bars_final:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", ha='center', va='bottom', fontsize=9)


# Labels and formatting
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Performance Metrics Comparison: Base vs Final Model')
plt.xticks(x, metric_names)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "performance_metric_comparison.png"))
plt.show()


# Confusion matrix
cm_base = confusion_matrix(y_test, y_pred_base, labels=[1,0])
cm_final = confusion_matrix(y_test, y_pred_final, labels=[1,0])

# plot confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Base Model CM
sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Flood", "No Flood"], 
            yticklabels=["Flood", "No Flood"], ax=axes[0])
axes[0].set_title("Base Model")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Plot Final Model CM
sns.heatmap(cm_final, annot=True, fmt="d", cmap="Greens", 
            xticklabels=["Flood", "No Flood"], 
            yticklabels=["Flood", "No Flood"], ax=axes[1])
axes[1].set_title("Final Model")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_comparison.png"))
plt.show()


# compute the pr auc
def compute_pr_auc(model, X, y_true):
    y_scores = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)
    return auc(recall, precision)

pr_auc_base = compute_pr_auc(base_model, X_test, y_test)
pr_auc_final = compute_pr_auc(final_model, X_test, y_test)

# precision-recall curve line graph
y_scores_base = base_model.predict_proba(X_test)[:, 1]
y_scores_final = final_model.predict_proba(X_test)[:, 1]

# Compute PR curve data
precision_base, recall_base, _ = precision_recall_curve(y_test, y_scores_base, pos_label=1)
precision_final, recall_final, _ = precision_recall_curve(y_test, y_scores_final, pos_label=1)

# Plot PR curves
plt.figure(figsize=(7, 5))
plt.plot(recall_base, precision_base, label=f"Base Model (AUC = {pr_auc_base:.4f})", color="skyblue")
plt.plot(recall_final, precision_final, label=f"Final Model (AUC = {pr_auc_final:.4f})", color="orange")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_curve_comparison.png"))
plt.show()

# get pr-curve for final model
# compute pr curve values
precision, recall, _ = precision_recall_curve(y_test, y_scores_final)
pr_auc = auc(recall, precision)

# Plot PR curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"Final Model (AUC = {pr_auc:.4f})", color="orange")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Final Random Forest Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "final_model_precision_recall_curve.png"))
plt.show()