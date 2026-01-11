# baseline_model_timeseries.py

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

import os
import pickle

# path
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PIPELINE_DIR, "../dataset/FloodPrediction(excel).csv")

PKL_DIR = os.path.join(PIPELINE_DIR, "baseline_timeseries_pkl")
REPORT_DIR = os.path.join(PIPELINE_DIR, "baseline_timeseries_report")

# create folder if they dont exist
os.makedirs(PKL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load dataset
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: '{DATA_PATH}' not found. Please check the file path.")
    exit()

print("--- 1. Initial Data Inspection ---")
print(data.info())

# Features and label data
X = data[['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage']]
y = data['Flood?'].astype(int)

# Handle missing values
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

# save dataset for next stages
with open(os.path.join(PKL_DIR, "full_dataset.pkl"), "wb") as f:
    pickle.dump((X, y), f)

print(f"Saved dataset to {PKL_DIR}")

# TimeSeriesSplit configuration
tscv = TimeSeriesSplit(n_splits=5)

# Helper functions
def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=[1,0], yticklabels=[1,0])
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()
    
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(REPORT_DIR, f"{name}_classification_report.csv"))

# Models
models_with_scaling = {
    "KNN": KNeighborsClassifier(n_neighbors=30, weights='distance'),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "SVM": SVC(kernel='linear', class_weight='balanced', probability=True)
}

models_no_scaling = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=80, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=80, class_weight='balanced')
}

# Run TimeSeriesSplit for baseline models
for name, clf in {**models_with_scaling, **models_no_scaling}.items():
    print(f"\nTraining model: {name}")
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Use pipeline for scaling if needed
        if name in models_with_scaling:
            pipe = Pipeline([('scaler', MinMaxScaler()), ('model', clf)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            model_to_save = pipe
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            model_to_save = clf
        
        all_acc.append(accuracy_score(y_val, y_pred))
        all_prec.append(precision_score(y_val, y_pred))
        all_rec.append(recall_score(y_val, y_pred))
        all_f1.append(f1_score(y_val, y_pred))
        
        # Save model per fold
        with open(os.path.join(PKL_DIR, f"{name}_fold{fold+1}.pkl"), 'wb') as f:
            pickle.dump(model_to_save, f)
        
        # Evaluate
        evaluate_model(f"{name}_fold{fold+1}", y_val, y_pred)
    
    print(f"{name} Average across folds:")
    print(f"Accuracy: {np.mean(all_acc):.4f}, Precision: {np.mean(all_prec):.4f}, Recall: {np.mean(all_rec):.4f}, F1: {np.mean(all_f1):.4f}")
