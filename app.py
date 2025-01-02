from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import logging

logging.basicConfig(level=logging.INFO)

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            try:
                reading_score = float(request.form.get('reading_score'))
                writing_score = float(request.form.get('writing_score'))
            except ValueError:
                raise ValueError("Reading and Writing scores must be numeric.")

            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            pred_df = data.get_data_as_data_frame()
            logging.info(f"Prediction data prepared: {pred_df}")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])

        except ValueError as e:
            logging.error(f"Error in form data: {e}")
            return render_template('home.html', error_message=f"Error: {str(e)}")

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('home.html', error_message="An error occurred during prediction. Please try again.")

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")