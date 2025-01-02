import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.utils import save_object 
from src.exception import CustomException
from src.logger import logging
from typing import Any

class TrainPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        '''Loads data from a CSV file'''
        try:
            logging.info(f"Loading data from {file_path}...")
            data = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomException(f"Error loading data: {e}", sys)

    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        '''Prepares data by separating features and target, and applying transformations'''
        try:
            logging.info("Preprocessing data...")
            X = data.drop("math_score", axis=1)
            y = data["math_score"]

            categorical_cols = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            numerical_cols = ["reading_score", "writing_score"]

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="mean")),  
                        ("scaler", StandardScaler()) 
                    ]), numerical_cols),
                    ("cat", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ]), categorical_cols)
                ]
            )

            logging.info("Fitting preprocessor to the data...")
            X_processed = self.preprocessor.fit_transform(X)

            logging.info("Preprocessing complete.")
            return X_processed, y
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise CustomException(f"Error in preprocessing data: {e}", sys)

    def train_model(self, X: Any, y: Any) -> Any:
        '''Train the model using RandomForestRegressor (can be changed to any model)'''
        try:
            logging.info("Training model...")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            logging.info("Model training complete.")
            return self.model
        except Exception as e:
            logging.error(f"Error training the model: {e}")
            raise CustomException(f"Error training the model: {e}", sys)

    def save_model_and_preprocessor(self):
        '''Save the trained model and the preprocessor to disk'''
        try:
            logging.info("Saving model and preprocessor...")
            if self.model and self.preprocessor:
                model_path = os.path.join("artifacts", "model.pkl")
                save_object(file_path=model_path, obj=self.model)

                preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
                save_object(file_path=preprocessor_path, obj=self.preprocessor)

                logging.info(f"Model and preprocessor saved to {model_path} and {preprocessor_path}.")
            else:
                raise CustomException("Model or preprocessor is not trained or loaded.", sys)
        except Exception as e:
            logging.error(f"Error saving model or preprocessor: {e}")
            raise CustomException(f"Error saving model or preprocessor: {e}", sys)

    def run(self, file_path: str):
        '''Main method to run the full training pipeline'''
        try:
            logging.info("Running the training pipeline...")
            data = self.load_data(file_path)

            X, y = self.preprocess_data(data)

            self.train_model(X, y)

            self.save_model_and_preprocessor()

            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error running the training pipeline: {e}")
            raise CustomException(f"Error running the training pipeline: {e}", sys)

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run("notebook/data/stud.csv")