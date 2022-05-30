"""
Creators: João Farias and Julio Freire
Date: 28 May 2022
Create API
"""

# from typing import Union
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import pandas as pd
import joblib
import os
import wandb
import sys
from source.api.pipeline import FeatureSelector, NumericalTransformer#, CategoricalTransformer, NumericalTransformer

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
#setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)
setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# name of the model artifact
artifact_model_name = "red_wine_quality/model_export:latest"

# initiate the wandb project
run = wandb.init(project="red_wine_quality",job_type="api")

# create the api
app = FastAPI()

# declare request example data using pydantic
# a wine in our dataset has the following attributes
class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

    class Config:
        schema_extra = {
            "example": {
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
        }

# give a greeting using GET
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Hello Welcome to Red Wine Quality Predict </strong></span></p>"""\
    """<p><span style="font-size:20px">In this project, we will apply the skills """\
        """acquired through the first stage of ML course about the Deploying a Scalable ML Pipeline in Production to develop """\
        """a classification model.</p></span>"""\
    """<p><span style="font-size:20px">For this step, we brought a Decision Tree as classifier model and to predict some quality of wine """\
        """is necessary acess this link: """\
        """<a href="https://red-wine-quality-ml.herokuapp.com/docs"> predict</a> and execute one try.</span></p>"""\
    """<p><span style="font-size:20px"> Our dataset was taken from: """\
        """<a href="https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009"> Kaggle: Red Wine Quality</a>.</span></p>"""

# run the model inference and use a Wine data structure via POST to the API.
@app.post("/predict")
async def get_inference(wine: Wine):
    
    # Download inference artifact
    model_export_path = run.use_artifact(artifact_model_name).file()
    pipe = joblib.load(model_export_path)
    
    # Create a dataframe from the input feature
    # note that we could use pd.DataFrame.from_dict
    # but due be only one instance, it would be necessary to
    # pass the Index.
    df = pd.DataFrame([wine.dict()])

    # Predict test data
    predict = pipe.predict(df)

    return "This is a GOOD wine" if predict[0] else "This is a BAD wine"