import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# =========================
# PATHS
# =========================
DATA_DIR = "processed_data"
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("\n📥 Loading training data...")
df = pd.read_csv(TRAIN_PATH)
print(f"✅ Train data shape: {df.shape}")

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Print feature information
print("\n📊 Feature Information:")
print(f"Number of features: {X.shape[1]}")
print("Feature names:")
print(X.columns.tolist())
print("\nFeature types:")
print(X.dtypes)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# PCA
# =========================
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\n📉 PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")

# =========================
# MODELS
# =========================
print("\n🚀 Training models...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_pca, y)
rf_model.fit(X_pca, y)

# =========================
# EVALUATION
# =========================
print("\n📊 Model Evaluation:")

xgb_probs = xgb_model.predict_proba(X_pca)[:, 1]
rf_probs = rf_model.predict_proba(X_pca)[:, 1]

ensemble_probs = (xgb_probs + rf_probs) / 2

print("\n🔹 XGBoost AUC:", roc_auc_score(y, xgb_probs))
print("🔹 Random Forest AUC:", roc_auc_score(y, rf_probs))
print("🔹 Ensemble AUC:", roc_auc_score(y, ensemble_probs))

# =========================
# SAVE MODELS AND METADATA
# =========================
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Save feature names for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

print("\n💾 Models saved successfully:")
print(f"✔ xgboost_model.pkl")
print(f"✔ random_forest_model.pkl")
print(f"✔ pca.pkl")
print(f"✔ scaler.pkl")
print(f"✔ feature_names.pkl")
print(f"\nFeatures used: {feature_names}")

print("\n✅ Training completed.")