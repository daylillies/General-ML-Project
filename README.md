# Machine Learning Model Pipeline

## Project Summary

This project implements a full **Machine Learning pipeline** for predicting student scores based on various features like gender, race, and test preparation. The pipeline involves **data ingestion**, **data transformation**, and **model training**, using **sklearn pipelines** for preprocessing. After model training, a **Flask web application** is used to serve predictions based on user input.

## Key Components

### 1. **Data Ingestion**
   - The `data_ingestion.py` component reads a dataset (`stud.csv`), splits it into training and testing sets, and stores the data in the `artifacts/` directory.
   - **Files saved**:
     - Raw data: `raw.csv`
     - Training data: `train.csv`
     - Testing data: `test.csv`

### 2. **Data Transformation**
   - **Data Preprocessing Pipeline**:
     - **Numerical Features**: Imputation (handling missing values), Scaling (StandardScaler).
     - **Categorical Features**: One-Hot Encoding, Imputation (handling missing values).
   - This transformation is done using **sklearn pipelines**, which are saved for future use.
   - **Preprocessing Object**: `preprocessor.pkl` stored in the `artifacts/` directory.
   - **Data Transformation** performed by `data_transformation.py`.

### 3. **Model Training**
   - Multiple regression models are trained using the following algorithms:
     - **Random Forest**
     - **Decision Tree**
     - **Gradient Boosting**
     - **XGBoost**
     - **Linear Regression**
     - **K-Neighbors**
     - **CatBoost**
     - **AdaBoost**
   - **Hyperparameter tuning** is performed to optimize the models.
   - The best model is selected based on **R² score** and saved to `model.pkl`.

### 4. **Flask Application**
   - The project includes a **Flask web application** for serving model predictions.
   - **Inputs**:
     - User provides values for features like `gender`, `race_ethnicity`, etc.
   - **Outputs**:
     - The Flask app returns predicted scores for the given inputs using the trained model.
   - Flask app is implemented in the `app.py` file and serves as the interface for users to interact with the model.


## Workflow

1. **Data Ingestion**: The script `data_ingestion.py` reads the dataset, splits it into training and testing sets, and saves them to the `artifacts/` directory.
   
2. **Data Transformation**: The `data_transformation.py` script handles preprocessing using **sklearn pipelines**, including:
   - Imputation for missing data.
   - Scaling numerical features.
   - One-hot encoding for categorical features.
   
3. **Model Training**: The `model_trainer.py` script trains multiple regression models, evaluates their performance using **R² score**, and performs **hyperparameter tuning** to find the best model. The best model is saved in `model.pkl`.

4. **Flask Application**: The `app.py` file sets up a web service using **Flask**, where users can input their features (e.g., `gender`, `test_preparation_course`), and the model will predict the corresponding score. To load the app, run:

```bash
python app.py
```
and go to the webpage:

http://127.0.0.1:5000

## Requirements

This project uses python 3.12.8.

To install the required libraries, run:

```bash
pip install .
```