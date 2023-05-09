import os
import sys
from src.exception import CustomExeption
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path : str= os.path.join('artifacts', "train.csv")
    test_data_path : str= os.path.join('artifacts', "test.csv")
    raw_data_path : str= os.path.join('artifacts', "data.csv")

class DataIngestion: 
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            train_set = pd.read_csv('Dataset/train.csv')
            test_set = pd.read_csv('Dataset/test.csv')
            logging.info("Read the train and test data")

            train_set.to_csv(self.ingestion_config.train_data_path,index= False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomExeption(e,sys)

if __name__ =="__main__":
    obj= DataIngestion()
    obj.initiate_data_ingestion()

