import sys
import pandas as pd 
from src.exception import CustomExeption
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            preds = model.predict(features)
            return preds
        
        except Exception as e:
            raise CustomExeption(e,sys)


class CustomData:
    def __init__(self,
                 cloneSize: int,
                 honeybee: int,
                 bumbles: int,
                 andrena: int,
                 osmia: int,
                 maxUpperTRange: int,
                 rainingDays: int,
                 fruitset: int):
        self.cloneSize = cloneSize
        self.honeybee = honeybee
        self.bumbles = bumbles
        self.andrena =andrena
        self.osmia =osmia
        self.maxUpperTRange = maxUpperTRange
        self.rainingDays =rainingDays
        self.fruitset = fruitset
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "cloneSize" : [self.cloneSize],
                "honeybee" : [self.honeybee],
                "bumbles" : [self.bumbles],
                "andrena" : [self.andrena],
                "osmia" : [self.osmia],
                "maxUpperTRange" : [self.maxUpperTRange],
                "rainingDays" : [self.rainingDays],
                "fruitset" : [self.fruitset]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomExeption(e,sys)

