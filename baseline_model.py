import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,recall_score, f1_score)
from sklearn.metrics import classification_report
import pickle
import os 

# load dataset
try:
    data = pd.read_csv("FloodPrediction(excel).csv")
except FileNotFoundError:
    print("Error: 'FloodPrediction(excel).csv' not found. Please check the file path.")
    exit()

print("--- 1. Initial Data Inspection ---")
print(data.info())

# Features and label data
X = data[['Max_Temp', 'Min_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage']]
y = data['Flood?'].astype(int)  

# Check and handle missing values
print("\nChecking for missing values...")
if data.isnull().values.any():
    print("Missing values found. Imputing with column median.")
    for col in X.columns:
        if data[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
else:
    print("No missing values found.")

# Check class distribution 
print("\nClass Distribution (Flood=1):\n", y.value_counts(normalize=True))

# Chronological split: 80% train+val, 20% test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=80, shuffle=False, stratify=None)

# Split trainval into 60% train and 20% val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=80, shuffle=False, stratify=None)

# Create directories
report_dir = "baseline_performance_report"
pkl_dir = "baseline_pkl"
os.makedirs(report_dir, exist_ok = True)
os.makedirs(pkl_dir, exist_ok = True)

# Save splits
with open(os.path.join(pkl_dir, "full_data_splits.pkl"), "wb") as f: 
    pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

# Evaluate model to get the baseline performance
def evaluate_model(name, y_true, y_pred):
    print(f"\n----- {name} Evaluation -----")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred) 
    rec = recall_score(y_true, y_pred) 
    f1 = f1_score(y_true, y_pred) 
    

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0]) 
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (for class 1 - Flood): {prec:.4f}") 
    print(f"Recall (for class 1 - Flood): {rec:.4f}")     
    print(f"F1 Score (for class 1 - Flood): {f1:.4f}")     
    print("Confusion Matrix:\n", cm)

    sns.heatmap(cm, cmap="Greens", annot=True, fmt="d", xticklabels=[1,0], yticklabels=[1,0]) # Added fmt="d" for integer annotations
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")

    # Save plot to PNG file inside report directory
    filename = f"{name}_confusion_matrix.png"
    filepath = os.path.join(report_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Confusion matrix for {name} saved to {filepath}")
    plt.show()

    plt.close()  # Close the figure to free memory

# create report for each evaluation metric for each model
def create_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True) 
    df_report = pd.DataFrame(report).transpose()

    # Save report inside baseline_performance_report_directory
    filename = f"{model_name.replace(' ', '_')}_classification_report.csv"
    filepath = os.path.join(report_dir, filename)
    # Using .replace(' ', '_') for consistent filenames if model_name has spaces
    df_report.to_csv(filepath) 
    
    print(f"\n--- Classification Report for {model_name} ---") 
    print(classification_report(y_true, y_pred, zero_division=0)) 
    print(f"Classification report for {model_name} saved to {filepath}")

# Models with Scaling (Pipeline)
models_with_scaling = {
    "KNN": KNeighborsClassifier(n_neighbors=30, weights='distance'),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "SVM": SVC(kernel='linear', class_weight='balanced'),
}

for name, clf in models_with_scaling.items():
    pipe = Pipeline([('scaler', MinMaxScaler()), ('model', clf)])
    pipe.fit(X_train, y_train)   
    y_pred = pipe.predict(X_val)
    # Evaluate the model
    evaluate_model(name, y_val, y_pred)
    # Create and save the classification report
    create_report(y_val, y_pred, name) 
    # Save the model to a file
    model_path = os.path.join(pkl_dir, f"{name}_pipe.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(pipe, f)

# Models without Scaling
models_no_scaling = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=80, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=80, class_weight='balanced'),
}

for name, clf in models_no_scaling.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    # Evaluate the model
    evaluate_model(name, y_val, y_pred)
    # Create and save the classification report
    create_report(y_val, y_pred, name)  

    model_path = os.path.join(pkl_dir, f"{name}_model.pkl")
    # Save the model to a file
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)



