#Contributor: Tiong
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from tabulate import tabulate

output_dir = "tuning_model"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
data = pd.read_csv("FloodPrediction(excel).csv")

# define X and y (type the columns inside the datasets)
X = data[['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage']]
y = data['Flood?'].astype(int)     #convert the 'Flood?' column into integers, 0: no flood, 1: flood

# Split into training and test sets (80% train_val, 20% test) with stratification                                              # stratify=y (to maintain same class distribution in train/test sets)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=80, shuffle=False, stratify=None)  # shuffle=False: prevents data leak
# Save: (X_trainval, X_test, y_trainval, y_test)

with open("data_splits.pkl", "wb") as f:
    pickle.dump((X_trainval, X_test, y_trainval, y_test), f)

# Contributor: Chin 
models_and_grids = {
    "KNN": {
        "pipeline": Pipeline([                             # with pipeline, scaling is automatically done after splitting and only on training data
            ('scaler', MinMaxScaler()),                    # this avoid data leak when do model training and cross-validation
            ('clf', KNeighborsClassifier())                # model training is done during nested cross validation
        ]),
        "param_grid": {
            'clf__n_neighbors': [3, 7, 10],
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan']
        }
    },
    "SVM": {
        "pipeline": Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', SVC(class_weight='balanced', random_state=80))
        ]),
        "param_grid": {
            'clf__C': [1, 10, 100],
            'clf__kernel': ['linear', 'rbf', 'poly'],
            'clf__gamma': ['scale', 'auto', 0.1]
        }
    },
    "Logistic Regression": {
        "pipeline": Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=80))
        ]),
        "param_grid": {
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l2'],  # Corrected penalty
            'clf__solver': ['liblinear', 'lbfgs']
        }
    },
    "Decision Tree": {
        "pipeline": Pipeline([
            ('clf', DecisionTreeClassifier(class_weight='balanced', random_state=80))
        ]),
        "param_grid": {
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy']
        }
    },
    "Random Forest": {
        "pipeline": Pipeline([
            ('clf', RandomForestClassifier(class_weight='balanced', random_state=80))
        ]),
        "param_grid": {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy']
        }
    }
}

# Contributor: Chan
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 2

outer_cv = KFold(n_splits=N_OUTER_SPLITS, shuffle=False)
inner_cv = KFold(n_splits=N_INNER_SPLITS, shuffle=False)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='weighted')   # dataset is imbalanced
}

nested_cv_results = {}

# loop through each model
for name, config in models_and_grids.items():
    print(f"\nEvaluating model: {name}")

    grid = GridSearchCV(
        estimator=config['pipeline'],
        param_grid=config['param_grid'],
        cv=inner_cv,
        scoring=scoring,
        refit='f1',
        n_jobs=-1,    # to make the gridsearchcv run faster since the grid search is sequential, it will be slow for rf and svm
        error_score='raise'   # if have any error during model fitting or scoring, will raise exception
    )


    outer_accuracy = []
    outer_f1 = []
    outer_precision = []
    outer_recall = []

    best_params_per_fold = []
    inner_fold_results = []  # To store results of each inner fold

    # outer cv loop: evaluate generalization
    for fold, (train_index, test_index) in enumerate(outer_cv.split(X_trainval, y_trainval)):
        print(f"\n Outer Fold {fold + 1}/{N_OUTER_SPLITS}")

        X_outer_train, X_outer_test = X_trainval.iloc[train_index], X_trainval.iloc[test_index]
        y_outer_train, y_outer_test = y_trainval.iloc[train_index], y_trainval.iloc[test_index]

        try:
            # Inner loop: do hyperparameter tuning using GridSearchCV
            grid.fit(X_outer_train, y_outer_train)

            # Store results of each inner fold
            print("  Inner Fold Results:")
            for i in range(N_INNER_SPLITS):
                accuracy_inner = grid.cv_results_[f'split{i}_test_accuracy'][grid.best_index_]
                f1_inner = grid.cv_results_[f'split{i}_test_f1'][grid.best_index_]
                print(f"   Fold {i + 1}: Accuracy = {accuracy_inner:.4f}, F1 = {f1_inner:.4f}")

            # predict (evaluate using the x_outer_test of each fold)
            y_pred = grid.predict(X_outer_test)       # evaluate model on outer test set

            # Evaluate and store performance
            acc = accuracy_score(y_outer_test, y_pred)
            f1 = f1_score(y_outer_test, y_pred, average='weighted')
            precision = precision_score(y_outer_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_outer_test, y_pred, average='weighted', zero_division=0)

            outer_accuracy.append(acc)
            outer_f1.append(f1)
            outer_precision.append(precision)
            outer_recall.append(recall)

            best_params_per_fold.append(grid.best_params_)   # save the best hyperparameter for each outer fold

            print(f"  Outer Fold {fold + 1} Performance:")
            print(f"   Accuracy: {acc:.4f}, F1: {f1:.4f}")
            print(f"   Best Params: {grid.best_params_}")

        except ValueError as e:
            print(f"  Error in Outer Fold {fold + 1}: {e}")
            outer_accuracy.append(np.nan)
            outer_f1.append(np.nan)
            best_params_per_fold.append(None)

    # store results for this model
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


    # summary of results
    summary_table = [[
        f"{nested_cv_results[name]['mean_accuracy']:.4f} ± {nested_cv_results[name]['std_accuracy']:.4f}",
        f"{nested_cv_results[name]['mean_f1']:.4f} ± {nested_cv_results[name]['std_f1']:.4f}",
        f"{nested_cv_results[name]['mean_precision']:.4f} ± {nested_cv_results[name]['std_precision']:.4f}",
        f"{nested_cv_results[name]['mean_recall']:.4f} ± {nested_cv_results[name]['std_recall']:.4f}"
    ]]
    print("\n>>> Performance Summary:")
    print(tabulate(summary_table, headers=["Mean Accuracy", "Mean F1 Score", "Mean Precision", "Mean Recall"], tablefmt="github"))


    # Best parameters for each fold
    print(">>> Best" \
    " Params per Outer Fold:")
    for i, params in enumerate(best_params_per_fold):
        print(f"  Fold {i + 1}: {params}")

    results_df = pd.DataFrame({
        'Outer Fold': list(range(1, N_OUTER_SPLITS + 1)),
        'Accuracy': outer_accuracy,
        'F1 Score': outer_f1,
        'Precision': outer_precision,
        'Recall': outer_recall,
        'Best Params': best_params_per_fold
    })
    results_df.to_csv(os.path.join(output_dir, f"{name.replace(' ', '_')}_fold_results.csv"), index=False)
    

# Save the nested_cv_results dictionary to a file
with open(os.path.join(output_dir, "nested_cv_results.pkl"), "wb") as f:
    pickle.dump(nested_cv_results, f)

print(f"\nNested CV results saved to '{os.path.join(output_dir, 'nested_cv_results.pkl')}'")

