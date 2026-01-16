import requests

API_URL = "http://127.0.0.1:8000/predict-flood/"

VALID_DATA = {
    "Max_Temp": 30,
    "Min_Temp": 20,
    "Rainfall": 120,
    "Relative_Humidity": 80,
    "Wind_Speed": 5,
    "Cloud_Coverage": 60
}

def test_prediction_success():
    response = requests.post(API_URL, json=VALID_DATA, timeout=5)
    assert response.status_code == 200

    result = response.json()
    assert "flood_prediction" in result
    assert "probability" in result


def test_invalid_temperature():
    data = VALID_DATA.copy()
    data["Max_Temp"] = 500

    response = requests.post(API_URL, json=data, timeout=5)
    assert response.status_code in [400, 422]


def test_negative_rainfall():
    data = VALID_DATA.copy()
    data["Rainfall"] = -5

    response = requests.post(API_URL, json=data, timeout=5)
    assert response.status_code in [400, 422]


def test_missing_field():
    r = requests.post(API_URL, json={"Rainfall": 20}, timeout=5)
    assert r.status_code == 422


def test_probability_range():
    r = requests.post(API_URL, json=VALID_DATA, timeout=5)
    prob = r.json()['probability']
    assert 0 <= prob <= 1




    

