import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

# =========================
# PATHS
# =========================
DATASET_PATH = r"C:/Users/TECH/Desktop/Online-Payments-Fraud-Detection/dataset/PS_20174392719_1491204439457_log.csv"
OUTPUT_DIR = "processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
print("\n📥 Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print(f"✅ Dataset Loaded: {df.shape}")

# =========================
# TARGET CHECK
# =========================
if "isFraud" not in df.columns:
    raise ValueError("❌ 'isFraud' column not found in dataset")

# =========================
# DROP STRING COLUMNS
# =========================
string_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\n🗑️ Removing non-numeric columns: {string_cols}")
df.drop(columns=string_cols, inplace=True)

# =========================
# CLASS DISTRIBUTION (BEFORE)
# =========================
print("\n📊 Class Distribution BEFORE Balancing:")
class_counts = Counter(df["isFraud"])
total = sum(class_counts.values())

for cls, cnt in class_counts.items():
    print(f"Class {cls}: {cnt} ({(cnt/total)*100:.4f}%)")

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# =========================
# IMBALANCE CHECK
# =========================
ratio = min(class_counts.values()) / max(class_counts.values())

if ratio < 0.2:
    print("\n⚠️ Severe imbalance detected. Applying SMOTE...")
    
    smote = SMOTE(
        sampling_strategy=0.5,  # realistic fraud ratio
        random_state=42,
        k_neighbors=3
    )
    
    X_res, y_res = smote.fit_resample(X, y)
else:
    print("\n✅ Dataset already balanced. No SMOTE applied.")
    X_res, y_res = X, y

# =========================
# BALANCED DATAFRAME
# =========================
balanced_df = pd.concat(
    [pd.DataFrame(X_res, columns=X.columns),
     pd.Series(y_res, name="isFraud")],
    axis=1
)

# =========================
# CLASS DISTRIBUTION (AFTER)
# =========================
print("\n📊 Class Distribution AFTER Balancing:")
balanced_counts = Counter(balanced_df["isFraud"])
total_bal = sum(balanced_counts.values())

for cls, cnt in balanced_counts.items():
    print(f"Class {cls}: {cnt} ({(cnt/total_bal)*100:.4f}%)")

# =========================
# SAVE BALANCED CSV
# =========================
balanced_path = os.path.join(OUTPUT_DIR, "balanced_dataset.csv")
balanced_df.to_csv(balanced_path, index=False)

print(f"\n💾 Balanced dataset saved at: {balanced_path}")

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df.drop("isFraud", axis=1),
    balanced_df["isFraud"],
    test_size=0.2,
    random_state=42,
    stratify=balanced_df["isFraud"]
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_path = os.path.join(OUTPUT_DIR, "train_data.csv")
test_path = os.path.join(OUTPUT_DIR, "test_data.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n📁 Files Generated Successfully:")
print(f"➡ Train CSV: {train_path}")
print(f"➡ Test CSV : {test_path}")

print("\n✅ Data preparation completed successfully.")
