from sklearn.metrics import roc_auc_score

def evaluate_models(xgb_model, rf_model, X, y):
    print("\n📊 Model Evaluation:")
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    rf_probs = rf_model.predict_proba(X)[:, 1]
    ensemble_probs = (xgb_probs + rf_probs) / 2

    print("\n🔹 XGBoost AUC:", roc_auc_score(y, xgb_probs))
    print("🔹 Random Forest AUC:", roc_auc_score(y, rf_probs))
    print("🔹 Ensemble AUC:", roc_auc_score(y, ensemble_probs))
    
    return xgb_probs, rf_probs, ensemble_probs
