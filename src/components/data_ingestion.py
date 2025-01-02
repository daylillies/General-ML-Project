import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple:
        logging.info("Entered the data ingestion method/component")
        try:
            data_path = 'notebook/data/stud.csv'
            
            # Check if the file exists before proceeding
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"The file {data_path} does not exist.")
            
            df = pd.read_csv(data_path)
            logging.info('Read the dataset as DataFrame')

            # Ensure the directory for saving files exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data to CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            logging.info("Train-test split initiated")
            # Perform train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except FileNotFoundError as fnf_error:
            logging.error(fnf_error)
            raise CustomException(fnf_error, sys)
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model R2 Score: {r2_score}")
    
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise CustomException(e, sys)