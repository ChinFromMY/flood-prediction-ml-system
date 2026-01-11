# model_evaluation.py (AUTO MODEL SELECTION - OLD PIPELINE)

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
BASE_DIR = os.path.dirname(__file__)
PKL_DIR = os.path.join(BASE_DIR, "baseline_pkl")
TUNING_DIR = os.path.join(BASE_DIR, "tuning_model")
MODEL_DIR = os.path.join(BASE_DIR, "best_model")
REPORT_DIR = os.path.join(BASE_DIR, "best_model_report")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load data splits
with open(os.path.join(PKL_DIR, "full_data_splits.pkl"), "rb") as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

print(f"Train+Val shape: {X_trainval.shape}, Test shape: {X_test.shape}")

# Load nested CV results
with open(os.path.join(TUNING_DIR, "nested_cv_results.pkl"), "rb") as f:
    nested_cv_results = pickle.load(f)

# Select best model (F1 + Recall)
best_model_name = max(
    nested_cv_results.items(),
    key=lambda x: x[1]["mean_f1"] + x[1]["mean_recall"]
)[0]

best_params = next(
    p for p in nested_cv_results[best_model_name]["best_params_per_fold"]
    if p is not None
)

clean_params = {k.replace("clf__", ""): v for k, v in best_params.items()}

print(f"\nBest Model Selected: {best_model_name}")
print(f"Best Parameters: {clean_params}")

# Model mapping
model_mapping = {
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier
}

needs_scaling = best_model_name in ["KNN", "SVM", "Logistic Regression"]

# Build final model
if needs_scaling:
    final_model = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", model_mapping[best_model_name](**clean_params))
    ])
else:
    final_model = model_mapping[best_model_name](**clean_params)

# Train final model
final_model.fit(X_trainval, y_trainval)

# Save model
model_path = os.path.join(
    MODEL_DIR, f"{best_model_name.replace(' ', '_')}_final_model.pkl"
)
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)

print(f"Final model saved to: {model_path}")

# Evaluate on test set
y_pred = final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n=== Final Test Performance ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{best_model_name} Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    REPORT_DIR, f"{best_model_name.replace(' ', '_')}_confusion_matrix.png"
)
plt.savefig(cm_path)
plt.close()

# Save classification report
report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True, zero_division=0)
).transpose()

report_path = os.path.join(
    REPORT_DIR, f"{best_model_name.replace(' ', '_')}_classification_report.csv"
)
report_df.to_csv(report_path)

print(f"\nReports saved to: {REPORT_DIR}")
