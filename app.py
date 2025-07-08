import pickle
from flask import Flask, render_template, request
import numpy as np
import os

app= Flask(__name__)

# Load trained ElasticNet model and Scaler
MODEL_PATH = os.path.join('models', 'elasticcv.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
elasticcv = pickle.load(open(MODEL_PATH, 'rb'))
scaler = pickle.load(open(SCALER_PATH, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    fire_status = None
    if request.method == "POST":
        try:
            temp = float(request.form.get("Temperature"))
            rh = float(request.form.get("RH"))
            ws = float(request.form.get("Ws"))
            rain = float(request.form.get("Rain"))
            ffmc = float(request.form.get("FFMC"))
            dmc = float(request.form.get("DMC"))
            isi = float(request.form.get("ISI"))
            classes = float(request.form.get("Classes"))
            region = float(request.form.get("Region"))

            input_data = np.array([[temp, rh, ws, rain, ffmc, dmc, isi, classes, region]])
            scaled_input = scaler.transform(input_data)
            pred_value = elasticcv.predict(scaled_input)[0]
            prediction = round(pred_value, 2)
            # Simple threshold for demo: if FWI > 20, fire risk
            fire_status = "Fire" if prediction > 20 else "No Fire"
        except Exception as e:
            prediction = None
            fire_status = None
    return render_template('predict.html', prediction=prediction, fire_status=fire_status)

if __name__ == '__main__':
    app.run(debug=True)
