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
data_path = Path(r"C:\Users\Nasreen M H\Desktop\project\WA_Fn-UseC_-Telco-Customer-Churn.csv")
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

# 8. Exporting Artifacts
model_path = "customer_churn_model.pkl"
encoder_path = "encoders.pkl"

pickle.dump(best_model, open(model_path, "wb"))
pickle.dump(encoders, open(encoder_path, "wb"))

print(f"\nSUCCESS! Elite Model saved to {model_path}")
print(f"SUCCESS! Encoders saved to {encoder_path}")
