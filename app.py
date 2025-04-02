from flask import Flask, render_template, request, redirect, url_for
from preprocessing import predict_fire_risk
import joblib
import os

app = Flask(__name__)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Check if model needs to be trained
if not os.path.exists('models/random_forest_model.joblib'):
    from preprocessing import preprocess_data
    print("Training model...")
    preprocess_data()

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = {
            'month': request.form['month'],
            'day': request.form['day'],
            'FFMC': float(request.form['FFMC']),
            'DMC': float(request.form['DMC']),
            'DC': float(request.form['DC']),
            'ISI': float(request.form['ISI']),
            'temp': float(request.form['temp']),
            'RH': float(request.form['RH']),
            'wind': float(request.form['wind']),
            'rain': float(request.form['rain'])
        }
        
        # Make prediction
        area, risk = predict_fire_risk(data)
        
        # Prepare result
        result = {
            'prediction': area,
            'risk_level': risk,
            'input_data': data
        }
        
        return render_template('output.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)