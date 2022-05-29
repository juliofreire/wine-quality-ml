"""
Creator: Joao Farias and Julio Freire
Date: 18 April 2022
API testing
"""
# Run using this comand python -m pytest -vv -s
# is necessary to put python -m because it adds the pythonpath
from fastapi.testclient import TestClient
import os
import sys
import pathlib
from source.api.main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# a unit test that tests the status code of the root path
def test_root():
    r = client.get("/")
    assert r.status_code == 200

# a unit test that tests the status code and response 
# for an instance with a low income
def test_get_inference_bad_wine():

    wine = {
            "fixed_acidity": 7.2,
            "volatile_acidity": 0.8,
            "citric_acid": 0.05,
            "residual_sugar": 2,
            "chlorides": 0.07,
            "free_sulfur_dioxide": 12.0,
            "total_sulfur_dioxide": 39.0,
            "density": 0.9977,
            "ph": 3.78,
            "sulphates": 0.56,
            "alcohol": 9.3
    }

    r = client.post("/predict", json=wine)
    # print(r.json())
    assert r.status_code == 200
    assert r.json() == "This is a BAD wine"

# a unit test that tests the status code and response 
# for an instance with a high income
def test_get_inference_good_wine():

    wine = {
            "fixed_acidity": 9.5,
            "volatile_acidity": 0.5,
            "citric_acid": 0.09,
            "residual_sugar": 2.6,
            "chlorides": 0.09,
            "free_sulfur_dioxide": 20.0,
            "total_sulfur_dioxide": 60.0,
            "density": 0.9987,
            "ph": 3.3,
            "sulphates": 0.72,
            "alcohol": 13.3
    }

    r = client.post("/predict", json=wine)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == "This is a GOOD wine"