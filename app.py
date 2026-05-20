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
        if col in df.columns:
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
    try:
        # Load dataset to extract real statistics for the business dashboard
        df_raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        
        total_customers = len(df_raw)
        churn_count = len(df_raw[df_raw['Churn'] == 'Yes'])
        churn_rate = round((churn_count / total_customers) * 100, 2)
        
        # Calculate charges
        avg_monthly_charges = round(df_raw['MonthlyCharges'].mean(), 2)
        df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
        total_revenue_m = round(df_raw['TotalCharges'].sum() / 1_000_000, 2)
        
        # Internet service splits
        internet_groups = df_raw.groupby('InternetService')['Churn'].value_counts().unstack().fillna(0)
        internet_labels = list(internet_groups.index)
        internet_total = [int(x) for x in internet_groups.sum(axis=1)]
        internet_churn = [int(x) for x in internet_groups.get('Yes', [0,0,0])]
        
        # Contract splits
        contract_groups = df_raw.groupby('Contract')['Churn'].value_counts().unstack().fillna(0)
        contract_labels = list(contract_groups.index)
        contract_total = [int(x) for x in contract_groups.sum(axis=1)]
        contract_churn = [int(x) for x in contract_groups.get('Yes', [0,0,0])]
        
        # Gender splits
        gender_groups = df_raw.groupby('gender')['Churn'].value_counts().unstack().fillna(0)
        gender_labels = list(gender_groups.index)
        gender_total = [int(x) for x in gender_groups.sum(axis=1)]
        gender_churn = [int(x) for x in gender_groups.get('Yes', [0,0])]

        stats = {
            "total_customers": total_customers,
            "churn_count": churn_count,
            "churn_rate": churn_rate,
            "avg_monthly_charges": avg_monthly_charges,
            "total_revenue_m": total_revenue_m,
            "internet": {
                "labels": internet_labels,
                "total": internet_total,
                "churn": internet_churn
            },
            "contract": {
                "labels": contract_labels,
                "total": contract_total,
                "churn": contract_churn
            },
            "gender": {
                "labels": gender_labels,
                "total": gender_total,
                "churn": gender_churn
            }
        }
    except Exception as e:
        # Fallback default statistics if dataset is missing
        stats = {
            "total_customers": 7043,
            "churn_count": 1869,
            "churn_rate": 26.54,
            "avg_monthly_charges": 64.76,
            "total_revenue_m": 16.06,
            "internet": {"labels": ["DSL", "Fiber optic", "No"], "total": [2421, 3096, 1526], "churn": [459, 1297, 113]},
            "contract": {"labels": ["Month-to-month", "One year", "Two year"], "total": [3875, 1473, 1695], "churn": [1655, 166, 48]},
            "gender": {"labels": ["Female", "Male"], "total": [3488, 3555], "churn": [939, 930]}
        }

    return render_template("dashboard.html", stats=stats)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.form.to_dict()
            result = predict_churn(data)
            return render_template(
                "predict.html",
                prediction=f"{result['Churn']} (Probability: {result['Probability']})",
                form_values=data
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    return render_template("predict.html", form_values=None)

@app.route("/explain", methods=["GET"])
def explain():
    return render_template("explain.html")

# =========================================================
# Run app
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
