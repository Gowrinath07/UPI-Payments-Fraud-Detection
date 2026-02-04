import pandas as pd
import os

DATA_DIR = "processed_data"
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")

def load_data():
    print("\n📥 Loading training data...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"✅ Train data shape: {df.shape}")
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    return X, y
