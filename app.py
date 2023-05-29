from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained XGBoost model
model = pickle.load(open("model.pkl", "rb"))

# Mapping dictionaries for label encoding
gender_mapping = {'female': 0, 'male': 1}
cholesterol_mapping = {'normal': 1, 'above_normal': 2, 'well_above_normal': 3}
glucose_mapping = {'normal': 1, 'above_normal': 2, 'well_above_normal': 3}
smoke_mapping = {'no': 0, 'yes': 1}
alco_mapping = {'no': 0, 'yes': 1}
active_mapping = {'not active': 0, 'active': 1}


@app.route('/')
def home():
    return render_template('in.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = float(data['age'])
    gender = gender_mapping[data['gender']]
    height = float(data['height'])
    weight = float(data['weight'])
    ap_hi = float(data['ap_hi'])
    ap_lo = float(data['ap_lo'])
    cholesterol = cholesterol_mapping[data['cholesterol']]
    glucose = glucose_mapping[data['glucose']]
    smoke = smoke_mapping[data['smoke']]
    alco = alco_mapping[data['alco']]
    active = active_mapping[data['active']]

    # Preprocess the input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol],
        'gluc': [glucose],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active]
    })

    # Make the prediction
    prediction = model.predict(input_data)[0]

    response = {'prediction': int(prediction)}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
