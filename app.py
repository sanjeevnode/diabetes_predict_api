from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the pre-trained model (replace 'your_model.joblib' with your actual model file)
model = load('./model/diabetes.joblib')

# Define a route to handle POST requests with JSON data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the JSON request
        data = request.get_json()

        # Extract the parameters: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction
        pregnancies = float(data.get('Pregnancies', 0.0))
        glucose = float(data.get('Glucose', 0.0))
        blood_pressure = float(data.get('BloodPressure', 0.0))
        skin_thickness = float(data.get('SkinThickness', 0.0))
        insulin = float(data.get('Insulin', 0.0))
        bmi = float(data.get('BMI', 0.0))
        diabetes_pedigree = float(data.get('DiabetesPedigreeFunction', 0.0))

        # Perform predictions using the loaded model and received data
        prediction = model.predict(np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree]).reshape(1, -1))

        # Return the prediction as a JSON response
        return jsonify({'prediction': str(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def home():
    return "This is Diabetes Prediction API by Sanjeev"
    

if __name__ == '__main__':
    app.run(debug=True)
