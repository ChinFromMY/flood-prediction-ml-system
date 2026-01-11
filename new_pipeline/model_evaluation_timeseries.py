# model_evaluation_timeseries.py
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# Path setup
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

PKL_DIR = os.path.join(PIPELINE_DIR, "baseline_timeseries_pkl")
TUNING_DIR = os.path.join(PIPELINE_DIR, "tuning_model_timeseries")
MODEL_DIR = os.path.join(PIPELINE_DIR, "best_model_timeseries")
REPORT_DIR = os.path.join(PIPELINE_DIR, "best_model_performance-report")

# Directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load preprocessed dataset
with open(os.path.join(PKL_DIR, "full_dataset.pkl"), "rb") as f:
    X, y = pickle.load(f)

# Split dataset into train+val and test (e.g., 80% train+val, 20% test)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # TimeSeries: no shuffle!
)
print(f"Train+Val shape: {X_trainval.shape}, Test shape: {X_test.shape}")

# Load nested CV results
with open(os.path.join(TUNING_DIR, "nested_cv_results_timeseries.pkl"), "rb") as f:
    nested_cv_results = pickle.load(f)

# Select best model based on F1 + Recall
best_model = max(
    nested_cv_results.items(),
    key=lambda x: x[1]["mean_f1"] + x[1]["mean_recall"]
)[0]

best_params = next(
    p for p in nested_cv_results[best_model]["best_params_per_fold"] if p is not None
)

clean_params = {k.replace("clf__", ""): v for k, v in best_params.items()}

print(f"Best Model: {best_model}")
print(f"Best Parameters: {clean_params}")


# Build final model
model_mapping = {
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier
}

# Decide if scaling is needed
needs_scaling = best_model in ["KNN", "SVM", "Logistic Regression"]

# Create final model
if needs_scaling:
    final_model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', model_mapping[best_model](**clean_params))

    ])
else:
    final_model = model_mapping[best_model](**clean_params)


# Train on full train+val
final_model.fit(X_trainval, y_trainval)

# Save trained model
model_path = os.path.join(MODEL_DIR, f"{best_model.replace(' ', '_')}_final_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)

print(f"\nTrained {best_model} saved to {model_path}")


# Evaluate on test set
y_pred = final_model.predict(X_test)

print("\nFinal Test Performance")
print(classification_report(y_test, y_pred, zero_division=0))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n=== {best_model} Performance on Test Set ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{best_model} Confusion Matrix - TimeSeries")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, f"{best_model.replace(' ', '_')}_confusion_matrix.png"))
plt.show()

# Classification report
clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
df_report = pd.DataFrame(clf_report).transpose()
report_path = os.path.join(REPORT_DIR, f"{best_model.replace(' ', '_')}_classification_report.csv")
df_report.to_csv(report_path, index=True)

print(f"\nClassification report saved to {report_path}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))
