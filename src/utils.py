# import os
# import sys
# import dill
# import numpy as np 
# import pandas as pd
# import pickle
# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV

# from src.exception import CustomException

# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)

#         os.makedirs(dir_path, exist_ok=True)

#         with open(file_path, "wb") as file_obj:
#             pickle.dump(obj, file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)
    
# def evaluate_models(X_train, y_train,X_test,y_test,models,param):
#     try:
#         report = {}

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para=param[list(models.keys())[i]]

#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,y_train)

#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)

#             #model.fit(X_train, y_train)  # Train model

#             y_train_pred = model.predict(X_train)

#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train, y_train_pred)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)
    
# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return pickle.load(file_obj)
#     except FileNotFoundError as e:
#         raise CustomException(f"No such file or directory: '{file_path}'", sys)
#     except Exception as e:
#         raise CustomException(e, sys)
import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Parameters:
        file_path (str): The file path where the object will be saved.
        obj (object): The object to be saved.
    
    Raises:
        CustomException: If an error occurs while saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate machine learning models using GridSearchCV and return the performance report.

    Parameters:
        X_train (array-like): The training feature matrix.
        y_train (array-like): The training target vector.
        X_test (array-like): The testing feature matrix.
        y_test (array-like): The testing target vector.
        models (dict): A dictionary containing the models to be evaluated.
        param (dict): A dictionary containing the parameters for GridSearchCV.

    Returns:
        dict: A dictionary containing the model names as keys and their corresponding test model scores as values.
    
    Raises:
        CustomException: If an error occurs during model evaluation.
    """
    try:
        report = {}

        for model_name, model in models.items():
            gs = GridSearchCV(model, param[model_name], cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using pickle.

    Parameters:
        file_path (str): The file path from which the object will be loaded.

    Returns:
        object: The loaded object.
    
    Raises:
        CustomException: If the file cannot be found or an error occurs while loading the object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except FileNotFoundError as e:
        raise CustomException(f"No such file or directory: '{file_path}'", sys)
    except Exception as e:
        raise CustomException(e, sys)
