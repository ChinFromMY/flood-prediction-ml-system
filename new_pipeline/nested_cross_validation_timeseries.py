# nested_cross_validation_timeseries.py
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer

# Path setup
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

PKL_DIR = os.path.join(PIPELINE_DIR, "baseline_timeseries_pkl")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "tuning_model_timeseries")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load preprocessed dataset from baseline pickles
with open(os.path.join(PKL_DIR, "full_dataset.pkl"), "rb") as f:
    X, y =pickle.load(f)    

print(f"Loaded preprocessed dataset: X shape = {X.shape}, y shape = {y.shape}")

# TimeSeriesSplit configuration
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 2

outer_cv = TimeSeriesSplit(n_splits=N_OUTER_SPLITS)
inner_cv = TimeSeriesSplit(n_splits=N_INNER_SPLITS)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='weighted')
}

# Models and hyperparameter grids
models_and_grids = {
    "KNN": {
        "pipeline": Pipeline([('scaler', MinMaxScaler()), ('clf', KNeighborsClassifier())]),
        "param_grid": {
            'clf__n_neighbors': [3, 7, 10],
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan']
        }
    },
    "SVM": {
        "pipeline": Pipeline([('scaler', MinMaxScaler()), ('clf', SVC(class_weight='balanced', probability=True, random_state=80))]),
        "param_grid": {
            'clf__C': [1, 10, 100],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': ['scale', 'auto', 0.1]
        }
    },
    "Logistic Regression": {
        "pipeline": Pipeline([('scaler', MinMaxScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=80))]),
        "param_grid": {
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'lbfgs']
        }
    },
    "Decision Tree": {
        "pipeline": Pipeline([('clf', DecisionTreeClassifier(class_weight='balanced', random_state=80))]),
        "param_grid": {
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy']
        }
    },
    "Random Forest": {
        "pipeline": Pipeline([('clf', RandomForestClassifier(class_weight='balanced', random_state=80))]),
        "param_grid": {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy']
        }
    }
}

# Nested CV loop
nested_cv_results = {}

for name, config in models_and_grids.items():
    print(f"\nEvaluating model: {name}")

    grid = GridSearchCV(
        estimator=config['pipeline'],
        param_grid=config['param_grid'],
        cv=inner_cv,
        scoring=scoring,
        refit='f1',
        n_jobs=-1,
        error_score='raise'
    )

    outer_accuracy, outer_f1, outer_precision, outer_recall = [], [], [], []
    best_params_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"\n Outer Fold {fold + 1}/{N_OUTER_SPLITS}")

        X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
        y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]

        try:
            grid.fit(X_outer_train, y_outer_train)
            y_pred = grid.predict(X_outer_test)

            acc = accuracy_score(y_outer_test, y_pred)
            f1 = f1_score(y_outer_test, y_pred, average='weighted')
            prec = precision_score(y_outer_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_outer_test, y_pred, average='weighted', zero_division=0)

            outer_accuracy.append(acc)
            outer_f1.append(f1)
            outer_precision.append(prec)
            outer_recall.append(rec)
            best_params_per_fold.append(grid.best_params_)

            print(f"  Fold {fold + 1} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
            print(f"  Best Params: {grid.best_params_}")

        except ValueError as e:
            print(f"  Error in Outer Fold {fold + 1}: {e}")
            outer_accuracy.append(np.nan)
            outer_f1.append(np.nan)
            best_params_per_fold.append(None)

    nested_cv_results[name] = {
        'mean_accuracy': np.mean(outer_accuracy),
        'mean_f1': np.mean(outer_f1),
        'mean_precision': np.mean(outer_precision),
        'mean_recall': np.mean(outer_recall),
        'std_accuracy': np.std(outer_accuracy),
        'std_f1': np.std(outer_f1),
        'std_precision': np.std(outer_precision),
        'std_recall': np.std(outer_recall),
        'outer_accuracy': outer_accuracy,
        'outer_f1': outer_f1,
        'outer_precision': outer_precision,
        'outer_recall': outer_recall,
        'best_params_per_fold': best_params_per_fold
    }

    # Save fold results
    results_df = pd.DataFrame({
        'Outer Fold': list(range(1, N_OUTER_SPLITS + 1)),
        'Accuracy': outer_accuracy,
        'F1 Score': outer_f1,
        'Precision': outer_precision,
        'Recall': outer_recall,
        'Best Params': best_params_per_fold
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}_fold_results.csv"), index=False)

# Save full nested CV dictionary
with open(os.path.join(OUTPUT_DIR, "nested_cv_results_timeseries.pkl"), "wb") as f:
    pickle.dump(nested_cv_results, f)

print(f"\nNested CV TimeSeries results saved to '{OUTPUT_DIR}'")
