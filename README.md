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


## How to run locally?
1. Clone the repository
git clone https://github.com/ChinFromMY/flood-prediction-ml-system.git
cd flood-prediction-ml-system

2. Create and activate virtual environment (at project root)
python -m venv venv
venv\Scripts\activate   #Windows user

3. Install dependencies
pip install -r backend/requirments.txt
pip install -r frontend/requirements.txt

4. Start the Backend (FastAPI)
cd backend
python -m uvicorn main:app --reload

6. Start the Frontend (Streamlit)
Open a new terminal:
cd flood-prediction-ml-system
venv\Scripts\activate
cd frontend   
streamlit run streamlit_app.py

8. Run API tests
Open another terminal:
venv\Scripts\activate
cd backend
pytest tests

**Backend must be running before testing
   
   



  
