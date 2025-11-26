# Flood-Prediction-Machine-Learning-Model

## Overview
This project aims to predict flood occurences using Machine Learning models, including: 
- K-Nearest Neighbors (KNN) 
- Support Vector Machine (SVM)  
- Logistic Regression 
- Random Forest 
- Decision Tree

The performance of these models is compared using evaluation metrics such as accuracy, precision, recall and F1 score.

The dataset used in this project is the  Bangladesh flood dataset and the models are trained to classfify whether a flood will occur or not.


## Folder Structure

- `baseline_model.py` — Builds baseline models using default parameters
- `baseline_metrics.py` — Calculates evaluation metrics and generates visualizations for baseline models
- `nested_cross_validation.py` — Performs hyperparameter tuning and cross validation
- `model_evaluation.py` — Evaluates the best selected model using the test dataset
- `base_vs_final.py` — Compare baseline vs final model performance
- `requirements.txt` — List of all required Python packages
- `README.txt` — User guide for running the project
- `tuning_model/` — Contains tuning results for each model and fold
- `random_forest_final_model/` — Contains the saved final Random Forest model with tuned hyperparameters
- `baseline_pkl/` — Contains saved baseline models (.pkl)
- `baseline_performance_report/` — Stored baseline performance results for each model
- `base vs final/` — Contains visual comparison plots of baseline vs final model
- `data_splits.pkl` — Stores the train/test split dataset
- `FloodPrediction(excel)` — The main dataset used (Bangladesh flood dataset)


## Requirements

— Python 3.10
— Recommend to use a virtual environment

### Python Packages (from `requirements.txt`)
— pandas
— matplotlib
— numpy
— scikit-learn
— seaborn
— tabulate


  
