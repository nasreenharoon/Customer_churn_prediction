import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

# Scikit-learn & Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# SMOTE for imbalance handling
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    import subprocess
    import sys
    print("imbalanced-learn not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
    from imblearn.over_sampling import SMOTE

# Suppress annoying warnings
warnings.filterwarnings("ignore")

print("=====================================================")
print("STARTING ELITE ML PIPELINE FOR CUSTOMER CHURN")
print("=====================================================\n")

# 1. Load Dataset
data_path = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# Drop CustomerID as it has no predictive power
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# 2. Data Cleaning
print("Cleaning data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 3. Feature Engineering
print("Engineering advanced features...")
# Convert some Yes/No categorical variables to 1/0 where appropriate, or just Label Encode everything.
# Let's keep it simple and apply LabelEncoder to all categorical columns for consistency with the existing web app.
categorical_columns = df.select_dtypes(include=['object']).columns

encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Prepare X and y
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Original Dataset Shape: {X.shape}")
print(f"Original Churn Distribution: \n{y.value_counts()}")

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Handling Imbalance with SMOTE
print("\nApplying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE Traning Dataset Shape: {X_train_smote.shape}")
print(f"SMOTE Churn Distribution: \n{pd.Series(y_train_smote).value_counts()}")

# 6. Model Training & Hyperparameter Tuning
print("\nTraining Random Forest with Hyperparameter Tuning (GridSearch)...")
rf = RandomForestClassifier(random_state=42)

# Simple grid to keep training time reasonable (usually takes 10-20 seconds on modern CPUs)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train_smote, y_train_smote)

best_model = grid_search.best_estimator_
print(f"\nBest Parameters Found: {grid_search.best_params_}")

# 7. Evaluation
print("\nEvaluating Elite Model on Test Data...")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# Generate and Save Plots as SVGs (Pure Python, No external dependencies like matplotlib)
try:
    # 1. Confusion Matrix SVG Plot
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    svg_cm = f"""<svg width="420" height="320" xmlns="http://www.w3.org/2000/svg">
  <style>
    .title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 16px; font-weight: bold; fill: #f0f2ff; }}
    .label {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #8b93b0; font-weight: 500; }}
    .val {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 18px; font-weight: bold; fill: #ffffff; }}
    .cell {{ stroke: rgba(255,255,255,0.06); stroke-width: 1.5; }}
  </style>
  <rect width="100%" height="100%" fill="#0a0e1a" rx="12"/>
  <text x="20" y="35" class="title">Confusion Matrix</text>
  
  <!-- Headers -->
  <text x="175" y="70" class="label" text-anchor="middle">PREDICTED: NO</text>
  <text x="305" y="70" class="label" text-anchor="middle">PREDICTED: YES</text>
  
  <text x="20" y="145" class="label">ACTUAL: NO</text>
  <text x="20" y="235" class="label">ACTUAL: YES</text>
  
  <!-- Cells -->
  <!-- Top Left: TN -->
  <rect x="120" y="95" width="110" height="85" fill="#1e3a8a" class="cell" rx="6"/>
  <text x="175" y="135" class="val" text-anchor="middle">{tn}</text>
  <text x="175" y="155" class="label" text-anchor="middle" font-size="10" fill="#93c5fd">True Negative</text>
  
  <!-- Top Right: FP -->
  <rect x="250" y="95" width="110" height="85" fill="#312e81" class="cell" rx="6"/>
  <text x="305" y="135" class="val" text-anchor="middle">{fp}</text>
  <text x="305" y="155" class="label" text-anchor="middle" font-size="10" fill="#a5b4fc">False Positive</text>
  
  <!-- Bottom Left: FN -->
  <rect x="120" y="195" width="110" height="85" fill="#4c1d95" class="cell" rx="6"/>
  <text x="175" y="235" class="val" text-anchor="middle">{fn}</text>
  <text x="175" y="255" class="label" text-anchor="middle" font-size="10" fill="#c084fc">False Negative</text>
  
  <!-- Bottom Right: TP -->
  <rect x="250" y="195" width="110" height="85" fill="#065f46" class="cell" rx="6"/>
  <text x="305" y="235" class="val" text-anchor="middle">{tp}</text>
  <text x="305" y="255" class="label" text-anchor="middle" font-size="10" fill="#34d399">True Positive</text>
</svg>"""
    
    with open('static/confusion_matrix.svg', 'w') as f:
        f.write(svg_cm)
    print("Saved static/confusion_matrix.svg")

    # 2. Feature Importance SVG Plot
    importances = best_model.feature_importances_
    features = list(X.columns)
    indices = np.argsort(importances)[::-1]
    
    svg_fe = """<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <style>
    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 16px; font-weight: bold; fill: #f0f2ff; }
    .label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #8b93b0; }
    .bar { fill: url(#barGrad); }
    .bar-val { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 10px; fill: #f0f2ff; font-weight: bold; }
  </style>
  <defs>
    <linearGradient id="barGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7c6df9" />
      <stop offset="100%" stop-color="#a855f7" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="#0a0e1a" rx="12"/>
  <text x="20" y="35" class="title">Top 10 Feature Importances</text>
"""
    
    max_importance = importances[indices[0]]
    y_offset = 70
    for i in range(10):
        feat_name = features[indices[i]]
        feat_val = importances[indices[i]]
        bar_width = int((feat_val / max_importance) * 350)
        svg_fe += f"""  <text x="20" y="{y_offset + 14}" class="label">{feat_name}</text>\n"""
        svg_fe += f"""  <rect x="180" y="{y_offset}" width="{bar_width}" height="20" class="bar" rx="3"/>\n"""
        svg_fe += f"""  <text x="{180 + bar_width + 10}" y="{y_offset + 14}" class="bar-val">{feat_val:.4f}</text>\n"""
        y_offset += 30
        
    svg_fe += "</svg>"
    
    with open('static/feature_importances.svg', 'w') as f:
        f.write(svg_fe)
    print("Saved static/feature_importances.svg")
except Exception as e:
    print(f"Could not generate SVG plots: {e}")

# 8. Exporting Artifacts
model_path = "customer_churn_model.pkl"
encoder_path = "encoders.pkl"

pickle.dump(best_model, open(model_path, "wb"))
pickle.dump(encoders, open(encoder_path, "wb"))

print(f"\nSUCCESS! Elite Model saved to {model_path}")
print(f"SUCCESS! Encoders saved to {encoder_path}")
