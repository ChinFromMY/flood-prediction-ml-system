import streamlit as st
import requests
import json

# --- CONFIGURATION ---
# The URL where your FastAPI server is running locally
# FASTAPI_URL = "http://127.0.0.1:8000/predict-flood/"
FASTAPI_URL = "https://flood-backend-api.onrender.com/predict-flood/"


st.set_page_config(
    page_title="Flood Prediction Dashboard",
    layout="wide"
)

# --- HELPER FUNCTION: API CALL ---
def get_flood_prediction(data):
    """Sends a POST request to the FastAPI endpoint."""
    try:
        # Use requests to send the data as JSON
        response = requests.post(
            FASTAPI_URL, 
            json=data
        )
        
        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: Server responded with status code {response.status_code}")
            st.error(f"Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"Connection Error: Could not connect to FastAPI server at {FASTAPI_URL}."
            f" Please ensure your FastAPI server is running."
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- STREAMLIT UI ---
st.title("🌊 Rainfall & Flood Risk Predictor")
st.markdown("Enter the meteorological data below to predict the likelihood of a flood.")

with st.form("flood_form"):
    st.header("Input Meteorological Data")
    
    # Use columns to organize inputs horizontally
    col1, col2, col3 = st.columns(3)
    
    # Collect all 6 input features
    max_temp = col1.number_input("Max Temperature (°C)", min_value=-50.0, max_value=50.0, value=30.5, step=0.1, key='Max_Temp')
    min_temp = col1.number_input("Min Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.1, step=0.1, key='Min_Temp')
    rainfall = col2.number_input("Rainfall (mm)", min_value=0.0, max_value=2000.0, value=15.0, step=0.1, key='Rainfall')
    relative_humidity = col2.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1, key='Relative_Humidity')
    wind_speed = col3.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=5.2, step=0.1, key='Wind_Speed')
    cloud_coverage = col3.number_input("Cloud Coverage (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1, key='Cloud_Coverage')
    
    # Form submission button
    submitted = st.form_submit_button("Get Flood Prediction")

if submitted:
    # 1. Structure data to match FastAPI's FloodData Pydantic model
    input_data = {
        "Max_Temp": max_temp,
        "Min_Temp": min_temp,
        "Rainfall": rainfall,
        "Relative_Humidity": relative_humidity,
        "Wind_Speed": wind_speed,
        "Cloud_Coverage": cloud_coverage,
    }
    
    with st.spinner('Contacting FastAPI server and predicting risk...'):
        # 2. Call the helper function
        result = get_flood_prediction(input_data)

    # 3. Display the results
    if result:
        prediction = result.get("flood_prediction")
        probability = result.get("probability")

        st.subheader("Prediction Results")
        
        # Display Probability Gauge/Bar
        prob_percent = probability * 100
        st.metric(label="Probability of Flood", value=f"{prob_percent:.2f}%")

        # Display Final Verdict
        if prediction == 1:
            st.error(f"High Risk of Flood: {prob_percent:.2f}% probability is above the 35% threshold.")
        else:
            st.success(f"Low Risk of Flood: {prob_percent:.2f}% probability is below the 35% threshold.")

        st.info(f"Model Threshold (FastAPI): {st.session_state.get('threshold', 0.35) * 100}%")