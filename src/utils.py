import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any

def save_object(file_path: str, obj: Any) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, float]:
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name)

            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"Model: {model_name}, Train Score: {train_model_score:.4f}, Test Score: {test_model_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
