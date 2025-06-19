import pickle
from flask import Flask, render_template, request
import numpy as np

application = Flask(__name__)
app = application

# Load trained ElasticNet model and Scaler
elasticcv = pickle.load(open('models/elasticcv.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))  # Renamed for clarity

@app.route('/')
def index():
    return render_template('home.html')  # Changed to match actual home.html

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # Collect data from the form
        temp = float(request.form.get("Temperature"))
        rh = float(request.form.get("RH"))
        ws = float(request.form.get("Ws"))
        rain = float(request.form.get("Rain"))
        ffmc = float(request.form.get("FFMC"))
        dmc = float(request.form.get("DMC"))
        # dc = float(request.form.get("DC"))
        isi = float(request.form.get("ISI"))
        classes = float(request.form.get("Classes"))
        region = float(request.form.get("Region"))

        # Create numpy array
        input_data = np.array([[temp, rh, ws, rain, ffmc, dmc, isi, classes, region]])

        # Scale input data
        scaled_input = scaler.transform(input_data)

        # Predict using the model
        prediction = elasticcv.predict(scaled_input)

        # Return result
        result = f"🔥 Fire Weather Index (FWI): {round(prediction[0], 2)}"
        return render_template('home.html', prediction=result)

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
