import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.components.data_transformation import DataTransformer
from src.utils import load_object

class Predict_pipline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            model=load_object(model_path)
            tranform=DataTransformer()
            scaled_data=tranform.transformations(features)
            preds=model.predict(scaled_data)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
class CustomDataClass:
    def __init__(self,Date,Hour,Temperature,Humidity,Wind_speed,Visibility,Dew_point_temperature,Solar_Radiation,
                 Rainfall,Snowfall,Seasons,Holiday,Functioning_Day):
        self.Date=Date
        self.Hour=Hour
        self.Temperature=Temperature
        self.Humidity=Humidity
        self.Wind_speed=Wind_speed
        self.Visibility=Visibility
        self.Dew_point_temperature=Dew_point_temperature
        self.Solar_Radiation=Solar_Radiation
        self.Rainfall=Rainfall
        self.Snowfall=Snowfall
        self.Seasons=Seasons
        self.Holiday=Holiday
        self.Functioning_Day=Functioning_Day

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "Date":[self.Date],
                "Hour":[self.Hour],
                "Temperature":[self.Temperature],
                "Humidity":[self.Humidity],
                "Wind_speed":[self.Wind_speed],
                "Visibility":[self.Visibility],
                "Dew_point_temperature":[self.Dew_point_temperature],
                "Solar_Radiation":[self.Solar_Radiation],
                "Rainfall":[self.Rainfall],
                "Snowfall":[self.Snowfall],
                "Seasons":[self.Seasons],
                "Holiday":[self.Holiday],
                "Functioning_Day":[self.Functioning_Day]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
