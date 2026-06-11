"""
Usage:  python train_model.py
Outputs: model.pkl  encoders.pkl  explainer.pkl  feature_order.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import shap
import joblib, json

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# ── 2. Clean ───────────────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ── 3. Encode categoricals ─────────────────────────────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── 4. Split ───────────────────────────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Train XGBoost ──────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss", random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\n" + "="*50)
print(classification_report(y_test, y_pred, target_names=["Stay","Churn"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ── 7. SHAP explainer ──────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model)

# ── 8. Save ────────────────────────────────────────────────────────────────────
joblib.dump(model,    "models/model.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(explainer,"models/explainer.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
with open("feature_order.json","w") as f:
    json.dump(list(X.columns), f)

print("\n Done! model.pkl  encoders.pkl  explainer.pkl  feature_order.json")