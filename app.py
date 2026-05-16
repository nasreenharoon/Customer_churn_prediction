from flask import Flask, render_template, request, jsonify
import pickle
from pathlib import Path
import pandas as pd

app = Flask(__name__)

# =========================================================
# Load model & encoders
# =========================================================
MODEL_PATH = Path("customer_churn_model.pkl")
ENCODER_PATH = Path("encoders.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
encoders = pickle.load(open(ENCODER_PATH, "rb"))

# =========================================================
# Feature list (must match training)
# =========================================================
FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# =========================================================
# DEFAULT VALUES (from dataset – VERY IMPORTANT)
# =========================================================
DEFAULTS = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.0,
    'TotalCharges': 70.0
}

# =========================================================
# Preprocessing
# =========================================================
def preprocess_input(input_data: dict) -> pd.DataFrame:
    row = {}

    for feature in FEATURES:
        value = input_data.get(feature)
        if value in [None, "", "None"]:
            value = DEFAULTS[feature]
        row[feature] = value

    df = pd.DataFrame([row])

    # Numeric conversion
    df['SeniorCitizen'] = int(df['SeniorCitizen'])
    df['tenure'] = int(df['tenure'])
    df['MonthlyCharges'] = float(df['MonthlyCharges'])
    df['TotalCharges'] = float(df['TotalCharges'])

    # Encode categorical columns
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    return df

# =========================================================
# Prediction
# =========================================================
def predict_churn(input_data: dict):
    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    return {
        "Churn": "Yes" if prediction == 1 else "No",
        "Probability": round(float(probability), 2)
    }

# =========================================================
# Routes
# =========================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        result = predict_churn(data)
        return render_template(
            "index.html",
            prediction=f"{result['Churn']} (Probability: {result['Probability']})",
            form_values=data
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =========================================================
# Run app
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
