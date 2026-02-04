from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def train_xgb(X, y):
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
    xgb_model.fit(X, y)
    return xgb_model

def train_rf(X, y):
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    return rf_model
