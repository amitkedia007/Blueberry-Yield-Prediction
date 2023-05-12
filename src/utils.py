import os
import sys
import dill

import numpy as np
import pandas as pd
from src.exception import CustomExeption
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True )

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomExeption(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            
            if params[model_name]:  # If there are parameters to tune for this model
                grid_search = GridSearchCV(model, params[model_name], cv=5)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_  # Use the best estimator found by GridSearchCV
            
            else:  # If there are no parameters to tune, just fit the model
                model.fit(X_train, y_train)  

            # Update the model in the dictionary with the fitted model
            models[model_name] = model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
        
        return report, models  # return the models dictionary as well

    except Exception as e:
        raise CustomExeption(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomExeption(e,sys)
