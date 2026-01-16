# Flood-Prediction-Machine-Learning-System

## Features:
- Flood risk prediction using supervised machine learning
- RESTful API built with FastAPI
- Interactive web interface using Streamlit
- Multiple model training and benchmarking
- Cross-validation and hyperparameter tuning (Grid Search)
- Automated API testing with PyTest and Requests
- Continuous integration using GitHub Actions
- Cloud deployment on Render

## Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Project Structure:

flood-prediction-ml-system/
│
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── model_prod.pkl          # Trained production model
│   ├── requirements.txt
│   └── tests/
│       └── test_api.py         # Automated API tests
│
├── frontend/
│   ├── streamlit_app.py        # Streamlit UI
│   └── requirements.txt
│
├── new_pipeline/               # ML training & evaluation scripts
├── old_pipeline/               # Earlier ML experiments
│
├── .github/workflows/
│   └── api-tests.yml           # CI workflow (GitHub Actions)
│
└── README.md


## Machine Learning Pipeline:
The project evaluates and compares multiple classification models:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

Training workflow includes:
- Time series cross-validation
- Nested cross-validation
- Hyperparameter tuning using Grid Search

The best-performing model (Random Forest) is saved as model_prod.pkl and used for production inference.

## Automated Testing:
API tests are implemented using:
- PyTest
- Requests

Tests verify:
- API availability
- Input validation

## Deployment
- Backend and frontend services are deployed on Render



  
