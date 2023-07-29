import sys
import os

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformConfig:
    preprocess_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_tranformer_config=DataTransformConfig()

    def transformations(df):
        

    # def get_data_transformer_obj(self):
    #     try:
    #         # Separate numerical and categorical features and apply transformers
    #         categorical_features=['Seasons','Holiday', 'Functioning Day']
    #         numerical_features=['Hour','Temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)',
    #                     'Dew point temperature(°C)','Solar Radiation (MJ/m2)','Rainfall(mm)', 'Snowfall (cm)']
            
    #         # categorical_pipeline=Pipeline(steps=[("label",LabelEncoder())])
    #         # logging.info("Pipelines created.")
    #         ordinal = OrdinalEncoder()

    #         preprocessor=ColumnTransformer([("ordinal",ordinal,categorical_features)])

    #         return preprocessor
    #     except Exception as e:
    #         raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Importing data from path
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Imported Train and Test data from path.")
            # Processing date column
            train_df['Date'] = pd.to_datetime(train_df["Date"].iloc[0:], format='%d/%m/%Y') # Converts values into timestamp.
            test_df['Date'] = pd.to_datetime(test_df["Date"].iloc[0:], format='%d/%m/%Y')

            # Adding Month column from date and dropping date column.
            train_df['Month'] = pd.DatetimeIndex(train_df['Date']).month
            train_df.drop('Date',axis=1,inplace=True)

            test_df['Month'] = pd.DatetimeIndex(test_df['Date']).month
            test_df.drop('Date',axis=1,inplace=True)

            logging.info("Date column transformation complete.")
            
            # Importing preprocessor
            processing_obj=self.get_data_transformer_obj()

            target_column=("Rented Bike Count")
            categorical_features=['Seasons','Holiday', 'Functioning Day']

            # Seperating Independent and Dependent features for Train and test data
            train_input=train_df.drop(columns=[target_column],axis=1)
            train_target=train_df[target_column]

            test_input=test_df.drop(columns=[target_column],axis=1)
            test_target=test_df[target_column]

            logging.info("Applying transformations on Train and Test input data.")
            train_input_array = processing_obj.fit_transform(train_input)
            test_input_array = processing_obj.transform(test_input)

            # Converting transformed data as array into dataframe
            train_array=np.c_[train_input_array,np.array(train_target)]
            test_array=np.c_[test_input_array,np.array(test_target)]

            logging.info("Preprocessing completed and file saved.")

            save_object(
                file_path=self.data_tranformer_config.preprocess_file_path,
                obj=processing_obj
            )

            return(
                train_array,
                test_array,
                self.data_tranformer_config.preprocess_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
            

    




