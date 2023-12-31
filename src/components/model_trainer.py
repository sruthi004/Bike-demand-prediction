import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn as skl

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def best_model(self,X_train,y_train,X_test,y_test,model):
        model.fit(X_train, y_train)
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate Train and Test dataset
        mae= mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_square = r2_score(y_test, y_test_pred)
        return mae,rmse,r2_square

    def train_model(self,train_df,test_df):
        try:
            logging.info("Split data into Train and Test.")
            X_train, y_train, X_test, y_test = (
                train_df.drop("Rented Bike Count",axis=1),
                train_df.iloc[:, 0],
                test_df.drop("Rented Bike Count",axis=1),
                test_df.iloc[:, 0]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)

            # Best model and score 
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            # if best_model_score<0.6:
            #     raise CustomException("No best model.")

            logging.info("Best model selected.")

            save_object(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )

        
            mae,rmse,r2_square=self.best_model(X_train,y_train,X_test,y_test,best_model)

            # predicted=best_model.predict(X_test)
            # r2 = r2_score(y_test,predicted)

            return best_model_name,mae,rmse,r2_square

        except Exception as e:
            raise CustomException(e,sys)
