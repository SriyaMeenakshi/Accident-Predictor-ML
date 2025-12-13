import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from xgboost import XGBClassifier

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Added more road features as requested
FEATURES = [
    "Hour", "Month", "DayOfWeek", 
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
    "Junction", "Traffic_Signal", "Crossing", "Station", "Stop", "Bump", # New Features
    "Is_Rush_Hour"
]
TARGET = "Risk_Level"

def calculate_metrics(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1_Score": round(f1_score(y_true, y_pred, average="macro"), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro"), 4),
        "Recall": round(recall_score(y_true, y_pred, average="macro"), 4)
    }

print("\n[1/7] Loading dataset...")
df = pd.read_csv("processed_accidents.csv")

# Ensure all new features exist (fill missing with False/0 if not found)
for col in ["Crossing", "Station", "Stop", "Bump"]:
    if col not in df.columns:
        df[col] = False

df = df[FEATURES + [TARGET]].dropna()

print("[2/7] Balancing Dataset (Upsampling)...")
# ---------------------------------------------------------
# CRITICAL FIX: BALANCE THE DATA
# ---------------------------------------------------------
df_medium = df[df[TARGET] == "Medium"]
df_high = df[df[TARGET] == "High"]
df_low = df[df[TARGET] == "Low"]

# Upsample High and Low to match Medium
df_high_upsampled = resample(df_high, replace=True, n_samples=len(df_medium), random_state=42)
df_low_upsampled = resample(df_low, replace=True, n_samples=len(df_medium), random_state=42)

# Combine back
df_balanced = pd.concat([df_medium, df_high_upsampled, df_low_upsampled])
print(f"   Original counts: {df[TARGET].value_counts().to_dict()}")
print(f"   Balanced counts: {df_balanced[TARGET].value_counts().to_dict()}")

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
print("[3/7] Preprocessing...")
label_encoder = LabelEncoder()
df_balanced[TARGET] = label_encoder.fit_transform(df_balanced[TARGET])
joblib.dump(label_encoder, "label_encoder.pkl")

X = df_balanced[FEATURES]
y = df_balanced[TARGET]

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

model_metrics = []

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
print("[4/7] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
model_metrics.append(calculate_metrics(y_test, rf_pred, "Random Forest"))
joblib.dump(rf, "rf_model.pkl")

print("[5/7] Training XGBoost...")
xgb = XGBClassifier(
    objective="multi:softprob", eval_metric="mlogloss", num_class=len(label_encoder.classes_),
    learning_rate=0.1, max_depth=10, n_estimators=200, random_state=42
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
model_metrics.append(calculate_metrics(y_test, xgb_pred, "XGBoost"))
joblib.dump(xgb, "xgb_model.pkl")

print("[6/7] Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, solver="lbfgs")
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
model_metrics.append(calculate_metrics(y_test, lr_pred, "Logistic Regression"))
joblib.dump(lr, "lr_model.pkl")

print("[7/7] Training Voting Classifier...")
voting = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("lr", lr)],
    voting="soft"
)
voting.fit(X_train, y_train)
v_pred = voting.predict(X_test)
model_metrics.append(calculate_metrics(y_test, v_pred, "Voting Ensemble"))
joblib.dump(voting, "voting_model.pkl")

# Save metrics
with open("metrics.json", "w") as f:
    json.dump(model_metrics, f)

print("\nSUCCESS: Models trained with BALANCED data!")