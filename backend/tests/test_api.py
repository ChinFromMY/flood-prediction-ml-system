import requests

API_URL = "http://127.0.0.1:8000/predict-flood/"

def test_prediction_success():
    data={
        "Max_Temp": 30,
        "Min_Temp": 20,
        "Rainfall": 120,
        "Relative_Humidity": 80,
        "Wind_Speed": 5,
        "Cloud_Coverage": 60
    }

    response = requests.post(API_URL, json=data)

    assert response.status_code == 200

    result = response.json()
    assert "flood_prediction" in result
    assert "probability" in result

def test_invalid_temperature():
    data = {
        "Max_Temp": 500,  # invalid
        "Min_Temp": 20,
        "Rainfall": 10,
        "Relative_Humidity": 50,
        "Wind_Speed": 2,
        "Cloud_Coverage": 30
    }

    response = requests.post(API_URL, json=data)

    # FastAPI returns 422 for validation errors
    assert response.status_code in [400, 422]

    def test_negative_rainfall():
        data={
            "Max_Temp": 30,
            "Min_Temp": 20,
            "Rainfall": -5,
            "Relative_Humidity": 50,
            "Wind_Speed": 2,
            "Cloud_Coverage": 30
        }

        response = requests.post(API_URL, json=data)

        assert response.status_code in [400, 422]

