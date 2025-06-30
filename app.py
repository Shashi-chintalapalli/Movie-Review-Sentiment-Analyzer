# app.py
from flask import Flask, render_template, request
import joblib
import re

# Load the model and components
MODEL_PATH = 'logistic_regression_model.joblib'
TFIDF_PATH = 'tfidf_vectorizer.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'

try:
    model_trained = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Model and components loaded.")
except Exception as e:
    print(f"❌ Error loading model/components: {e}")
    exit()

# Initialize Flask app
app = Flask(__name__)

# Text cleaning function
def clean_text(text):
    text = str(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form['text']
        cleaned_text = clean_text(input_text)
        text_vector = tfidf_vectorizer.transform([cleaned_text])
        prediction_encoded = model_trained.predict(text_vector)
        predicted_sentiment = label_encoder.inverse_transform(prediction_encoded)[0]

        return render_template('index.html',
                               input_text=input_text,
                               cleaned_text=cleaned_text,
                               prediction=predicted_sentiment)
    except Exception as e:
        return f"<h2>Error occurred: {e}</h2>"

# Run server
if __name__ == "__main__":
    app.run(debug=True, port=5055)
