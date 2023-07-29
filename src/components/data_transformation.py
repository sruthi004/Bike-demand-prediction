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

    def transformations(self,df):
        try:
            # Creating month column
            df['Date'] = pd.to_datetime(df["Date"].iloc[0:], format='%d/%m/%Y') # Converts values into timestamp.
            df['Month'] = pd.DatetimeIndex(df['Date']).month
            df.drop('Date',axis=1,inplace=True)
            
            # Label encoder and get dummies
            cat_feat = df.loc[:,['Seasons','Holiday', 'Functioning Day']]
            cat = pd.get_dummies(df.loc[:,['Holiday', 'Functioning Day']])

            df['Holiday'] = cat['Holiday_Holiday'] # 0 Not a Holiday, 1 means Holiday
            df['Functioning Day'] = cat['Functioning Day_Yes'] # 1 is Yes, 0 is No

            lbl = LabelEncoder()
            df['Seasons'] = lbl.fit_transform(df['Seasons'])
            return df
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Importing data from path
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Imported Train and Test data from path.")
            logging.info("Transformations started.")

            train_df = self.transformations(train_df)
            test_df = self.transformations(test_df)

            logging.info("Transformations complete.")

            return(
                train_df,
                test_df
            )

        except Exception as e:
            raise CustomException(e,sys)
            

    




