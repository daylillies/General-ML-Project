import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from typing import List


class PredictPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def load_model_and_preprocessor(self):
        '''Loads the model and preprocessor if they are not already loaded'''
        if self.model is None or self.preprocessor is None:
            try:
                model_path = os.path.join("artifacts", "model.pkl")
                preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

                print("Before Loading")
                self.model = load_object(file_path=model_path)
                self.preprocessor = load_object(file_path=preprocessor_path)
                print("After Loading")
            except Exception as e:
                raise CustomException(f"Error loading model or preprocessor: {e}", sys)

    def predict(self, features: pd.DataFrame) -> List[float]:
        '''Makes predictions after ensuring model and preprocessor are loaded'''
        try:
            self.load_model_and_preprocessor()

            data_scaled = self.preprocessor.transform(features)

            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(f"Error during prediction: {e}", sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        '''Converts the input data to a DataFrame for prediction'''
        custom_data_input_dict = {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        }
        return pd.DataFrame(custom_data_input_dict)