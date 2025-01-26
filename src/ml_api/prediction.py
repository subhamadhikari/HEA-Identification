import json
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

def load_model():
    """
    Load machine learning models from disk.
    """
    # xgb_interpolate_model = pickle.load(open("model/model_xgb.pkl", "rb"))
    # xgb_extrapolate_model = pickle.load(open("model/xgb-extrapoltae.pkl", "rb"))
    # rf_extrapolate_model = pickle.load(open("model/rf-extrapolate.pkl", "rb"))
    # rf_interpolate_model = pickle.load(open("model/rf-interpolate.pkl", "rb"))
    # app.xgb_inter = xgb_interpolate_model
    # app.xgb_extra = xgb_extrapolate_model
    # app.rf_inter = rf_interpolate_model
    # app.rf_extra = rf_extrapolate_model
    with open("model/model_xgb.pkl", "rb") as file:
        app.xgb_inter = pickle.load(file)
    with open("model/xgb-extrapoltae.pkl", "rb") as file:
        app.xgb_extra = pickle.load(file)
    with open("model/rf-extrapolate.pkl", "rb") as file:
        app.rf_extra = pickle.load(file)
    with open("model/rf-interpolate.pkl", "rb") as file:
        app.rf_inter = pickle.load(file)


load_model()


class ResponseModel(BaseModel):
    """
    Model for API response containing a name and value.
    """

    name: str
    value: float


# interpolate - xgb


@app.get("/predict-xgb_interpolate", response_model=ResponseModel)
async def make_prediction_xgb_in(x_in):
    """
    Predict the energy formation based on the model type and input index.
    """
    print("input i below:")
    input_data = app.interpolate_dataset[int(x_in)]
    print(input_data)
    print("xgb interpoltae predict")
    print(type(input_data))
    input_data = pd.DataFrame.from_dict(data=input_data, orient="index")
    print(type(input_data))
    input_data = input_data.T
    input_data = input_data.drop(columns=["formula", "Ef_per_atom"])
    input_data = input_data.apply(pd.to_numeric, errors="coerce")
    result = app.xgb_inter.predict(input_data)
    print(result)
    print("xgb interpolate model has been successfully loaded!")
    return ResponseModel(name="prediction", value=result[0])


@app.get("/predict-rf_interpolate", response_model=ResponseModel)
async def make_prediction_rf_in(x_in):
    """
    Predict the energy formation based on the model type and input index.
    """
    print("input i below:")
    input_data = app.interpolate_dataset[int(x_in)]
    print(input_data)
    print("xgb interpoltae predict")
    print(type(input_data))
    input_data = pd.DataFrame.from_dict(data=input_data, orient="index")
    print(type(input_data))
    input_data = input_data.T
    input_data = input_data.drop(columns=["formula", "Ef_per_atom"])
    input_data = input_data.apply(pd.to_numeric, errors="coerce")
    result = app.rf_inter.predict(input_data)
    print(result)
    print("RF interpolate model has been successfully loaded!")
    return ResponseModel(name="prediction", value=result[0])


@app.get("/predict-xgb_extrapolate", response_model=ResponseModel)
async def make_prediction_xgb_ex(x_in):
    """
    Predict the energy formation based on the model type and input index.
    """
    print("input i below:")
    input_data = app.extrapolate_dataset[int(x_in)]
    print(input_data)
    print("xgb extrapoltae predict")
    print(type(input_data))
    input_data = pd.DataFrame.from_dict(data=input_data, orient="index")
    print(type(input_data))
    input_data = input_data.T
    input_data = input_data.drop(columns=["formula", "Ef_per_atom"])
    input_data = input_data.apply(pd.to_numeric, errors="coerce")
    result = app.xgb_extra.predict(input_data)
    print(result)
    print("xgb extrapolate model has been successfully loaded!")
    return ResponseModel(name="prediction", value=result[0])


@app.get("/predict-rf_extrapolate", response_model=ResponseModel)
async def make_prediction_rf_ex(x_in):
    """
    Predict the energy formation based on the model type and input index.
    """
    print("input i below:")
    input_data = app.extrapolate_dataset[int(x_in)]
    print(input_data)
    print("rf extrapoltae predict")
    print(type(input_data))
    input_data = pd.DataFrame.from_dict(data=input_data, orient="index")
    print(type(input_data))
    input_data = input_data.T
    input_data = input_data.drop(columns=["formula", "Ef_per_atom"])
    input_data = input_data.apply(pd.to_numeric, errors="coerce")
    result = app.rf_extra.predict(input_data)
    print(result)
    print("xgb extrapolate model has been successfully loaded!")
    return ResponseModel(name="prediction", value=result[0])


def get_data():
    """
    Prepare input data for prediction by converting it to DataFrame and cleaning it.]
    """
    testdata_inter = pd.read_csv("datasets/interpolate_test.csv").sample(n=5)
    testdata_extra = pd.read_csv("datasets/extrapolate_test.csv").sample(n=5)
    testdata_inter = json.loads(testdata_inter.to_json(orient="records"))
    testdata_extra = json.loads(testdata_extra.to_json(orient="records"))
    app.interpolate_dataset = testdata_inter
    app.extrapolate_dataset = testdata_extra


get_data()


@app.get("/get_interpol_testdata")
async def get_interpol_data():
    """
    Fetch and return test data for interpolation.
    """
    # testdata = pd.read_csv("datasets/interpolate_test.csv").sample(n=5)
    # testdata = json.loads(testdata.to_json(orient="records"))
    # # print(testdata)
    # print(type(testdata))
    # return testdata
    return app.interpolate_dataset


@app.get("/get_extrapol_testdata")
async def get_extrapol_data():
    """
    Fetch and return test data for extrapolation.
    """
    # testdata = pd.read_csv("datasets/interpolate_test.csv").sample(n=5)
    # testdata = json.loads(testdata.to_json(orient="records"))
    # # print(testdata)
    # print(type(testdata))
    # return testdata
    return app.extrapolate_dataset
