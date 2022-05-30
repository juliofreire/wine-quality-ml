"""
Creators: Joao Farias and Julio Freire
Date: 28 May 2022
Script that POSTS to the API using the requests 
module and returns both the result of 
model inference and the status code
"""
import requests
import json
# import pprint

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

#url = "http://127.0.0.1:8000"
url = "https://red-wine-quality-ml.herokuapp.com"
response = requests.post(f"{url}/predict",
                         json=wine)

print(f"Request: {url}/predict")
print(f"Wine: \n fixed_acidity: {wine['fixed_acidity']}\n volatile_acidity: {wine['volatile_acidity']}\n"\
      f" citric_acid: {wine['citric_acid']}\n residual_sugar: {wine['residual_sugar']}\n"\
      f" chlorides: {wine['chlorides']}\n"\
      f" free_sulfur_dioxide: {wine['free_sulfur_dioxide']}\n"\
      f" total_sulfur_dioxide: {wine['total_sulfur_dioxide']}\n"\
      f" density: {wine['density']}\n"\
      f" ph: {wine['ph']}\n"\
      f" sulphates: {wine['sulphates']}\n"\
      f" alcohol: {wine['alcohol']}\n"\
     )
print(f"Result of model inference: {response.json()}")
print(f"Status code: {response.status_code}")