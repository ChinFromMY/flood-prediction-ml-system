import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import os
import pickle

# load the saved data splits 
with open("data_splits.pkl", "rb") as f:
    X_trainval, X_test, y_trainval, y_test = pickle.load(f)

rf_final_dir = "random_forest_final_model"
os.makedirs(rf_final_dir, exist_ok = True)

# best hyperparameters from GridSearchCV (Outer Fold 3)
best_params = {
    'clf__n_estimators': 200,
    'clf__max_depth': None,
    'clf__min_samples_split': 2,
    'clf__min_samples_leaf': 1,
    'clf__criterion': 'entropy'
}

# train model using best parameters
rf_model = RandomForestClassifier(
    n_estimators=best_params['clf__n_estimators'],
    max_depth=best_params['clf__max_depth'],
    min_samples_split=best_params['clf__min_samples_split'],
    min_samples_leaf=best_params['clf__min_samples_leaf'],
    criterion=best_params['clf__criterion'],
    random_state=80,
    class_weight='balanced'
)

# fit the model （using full training+validation dataset）
rf_model.fit(X_trainval, y_trainval)

# make predictions (using test dataset)
y_pred = rf_model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=[f"Actual_{cls}" for cls in sorted(set(y_test))], 
                     columns=[f"Predicted_{cls}" for cls in sorted(set(y_test))])

# print results
print("Model Performance on Test Set:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("Confusion Matrix:")
print(cm_df)

with open(os.path.join(rf_final_dir, "random_forest_model.pkl"), "wb") as f: 
    pickle.dump(rf_model, f)





