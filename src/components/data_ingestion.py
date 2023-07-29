import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformer
from src.components.data_transformation import DataTransformConfig

from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started.")
        try:
            df=pd.read_csv("notebook\data\SeoulBikeData.csv",encoding= 'unicode_escape')
            logging.info("Imported data as dataframe.")

            # Creates file path for our Train, Test and Raw files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test split initiated.")
            # This is done to avoid snooping bias.
            train_data,test_data=train_test_split(df,test_size=0.20,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion completed.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    #obj.initiate_data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transform=DataTransformer()
    train_array,test_array,_=data_transform.initiate_data_transformation(train_data,test_data)

    modeltrain=ModelTrainer()
    print(modeltrain.train_model(train_array,test_array))

            