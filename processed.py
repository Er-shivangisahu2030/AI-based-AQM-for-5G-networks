"""
processed_data.py
Data Preprocessing Pipeline
------------------------------
Loads raw traffic_data.csv and:
  1. Checks for missing values
  2. Removes duplicates
  3. Detects and removes outliers (IQR method)
  4. Normalises features (StandardScaler)
  5. Analyses class balance
  6. Saves processed_data.csv

Run this BEFORE train_model.py
"""

import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

RAW_PATH       = "data/traffic_data.csv"
PROCESSED_PATH = "data/processed_data.csv"

FEATURES = [
    "queue_length",
    "arrival_rate",
    "service_rate",
    "avg_delay_ms",
    "throughput",
    "drop_rate",
]
TARGET = "congestion"


# ─────────────────────────────────────────────
# Step 1 — Load raw data
# ─────────────────────────────────────────────

def load_raw(path=RAW_PATH):

    print("\n" + "=" * 50)
    print("  📂  Loading raw dataset...")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found at '{path}'.\n"
            f"Run dataset_generator.py first."
        )

    df = pd.read_csv(path)

    print(f"  Loaded   : {df.shape[0]} rows, "
          f"{df.shape[1]} columns")
    print(f"  Columns  : {list(df.columns)}")

    return df


# ─────────────────────────────────────────────
# Step 2 — Check and handle missing values
# ─────────────────────────────────────────────

def handle_missing(df):

    print("\n  🔍  Checking missing values...")

    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        print("  ✅  No missing values found.")
    else:
        print(f"  ⚠️   Found {total_missing} missing values:")
        print(missing[missing > 0])
        df = df.dropna()
        print(f"  Rows after drop : {len(df)}")

    return df


# ─────────────────────────────────────────────
# Step 3 — Remove duplicates
# ─────────────────────────────────────────────

def remove_duplicates(df):

    print("\n  🔍  Checking duplicates...")

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    if removed == 0:
        print("  ✅  No duplicates found.")
    else:
        print(f"  Removed {removed} duplicate rows.")
        print(f"  Rows remaining : {len(df)}")

    return df


# ─────────────────────────────────────────────
# Step 4 — Remove outliers (IQR method)
# ─────────────────────────────────────────────

def remove_outliers(df, features=FEATURES):

    print("\n  🔍  Removing outliers (IQR method)...")

    before = len(df)

    for col in features:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    removed = before - len(df)
    print(f"  Removed  : {removed} outlier rows")
    print(f"  Remaining: {len(df)} rows")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Step 5 — Class balance analysis
# ─────────────────────────────────────────────

def analyse_balance(df):

    print("\n  📊  Class balance analysis...")

    counts = df[TARGET].value_counts()
    total  = len(df)

    for label, count in counts.items():
        name = "Congested" if label == 1 else "Normal   "
        pct  = count / total * 100
        bar  = "█" * int(pct / 2)
        print(f"  {name} [{label}] : "
              f"{count:>5} samples  ({pct:.1f}%)  {bar}")

    ratio = counts.min() / counts.max()
    if ratio < 0.4:
        print("  ⚠️   Dataset is imbalanced. "
              "Consider oversampling.")
    else:
        print("  ✅  Class balance is acceptable.")

    return df


# ─────────────────────────────────────────────
# Step 6 — Feature statistics
# ─────────────────────────────────────────────

def show_stats(df):

    print("\n  📈  Feature statistics:")
    print(df[FEATURES].describe().round(3).to_string())


# ─────────────────────────────────────────────
# Step 7 — Normalise features
# ─────────────────────────────────────────────

def normalise(df, features=FEATURES):

    print("\n  ⚙️   Normalising features (StandardScaler)...")

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    print("  ✅  Features normalised.")
    print(f"  Mean after scaling : "
          f"{df_scaled[features].mean().round(4).to_dict()}")

    return df_scaled, scaler


# ─────────────────────────────────────────────
# Step 8 — Save processed data
# ─────────────────────────────────────────────

def save_processed(df, path=PROCESSED_PATH):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\n  💾  Saved → '{path}'")
    print(f"  Shape    : {df.shape}")


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def run_pipeline(raw_path=RAW_PATH,
                 processed_path=PROCESSED_PATH):

    print("\n" + "=" * 50)
    print("  ⚙️   Data Preprocessing Pipeline")
    print("=" * 50)

    df = load_raw(raw_path)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = analyse_balance(df)
    show_stats(df)
    df, scaler = normalise(df)
    save_processed(df, processed_path)

    print("\n  ✅  Preprocessing complete!")
    print("=" * 50)

    return df, scaler


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    run_pipeline(
        raw_path=RAW_PATH,
        processed_path=PROCESSED_PATH
    )