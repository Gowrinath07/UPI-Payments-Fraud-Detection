from data_loader import load_data
from preprocessing import scale_data, apply_pca
from models import train_xgb, train_rf
from evaluate import evaluate_models
from save_models import save_model

# ---------------------------
# Load Data
# ---------------------------
X, y = load_data()

# ---------------------------
# Preprocessing
# ---------------------------
X_scaled, scaler = scale_data(X)
X_pca, pca = apply_pca(X_scaled)

# ---------------------------
# Train Models
# ---------------------------
print("\n🚀 Training models...")
xgb_model = train_xgb(X_pca, y)
rf_model = train_rf(X_pca, y)

# ---------------------------
# Evaluate Models
# ---------------------------
xgb_probs, rf_probs, ensemble_probs = evaluate_models(xgb_model, rf_model, X_pca, y)

# ---------------------------
# Save Models
# ---------------------------
print("\n💾 Saving models...")
save_model(xgb_model, "xgboost_model.pkl")
save_model(rf_model, "random_forest_model.pkl")
save_model(pca, "pca.pkl")
save_model(scaler, "scaler.pkl")

print("\n✅ Training completed.")
