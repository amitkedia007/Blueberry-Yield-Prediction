import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.exception import CustomExeption
from src.logger import logging
from src.utils import save_object  

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            Variables_to_remove = ['fruitmass', 'seeds', 'AverageOfLowerTRange', 'MinOfLowerTRange',
                                   'AverageRainingDays', 'MinOfUpperTRange', 'MaxOfLowerTRange', 'AverageOfUpperTRange']

            target_variable = "yield"

            input_feature_train_df = train_df.drop(columns=Variables_to_remove + [target_variable])
            target_feature_train_df = train_df[target_variable]

            input_feature_test_df = test_df.drop(columns=Variables_to_remove)

            logging.info(f"Dropped the highly correlated variables from the train and test data")

            train_arr = np.c_[np.array(input_feature_train_df), np.array(target_feature_train_df)]

            test_arr = np.array(input_feature_test_df)

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=None  # Update this with the proper preprocessor object as needed
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomExeption(e,sys)
