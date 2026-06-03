"""
train_model.py
ML Model Training for Congestion Prediction
----------------------------------------------
Loads processed_data.csv and:
  1. Splits into train / test sets
  2. Trains three models:
       - Logistic Regression
       - Decision Tree
       - Random Forest  ← usually best
  3. Evaluates with accuracy, classification
     report, and confusion matrix
  4. Prints feature importances
  5. Saves the best model + scaler as .pkl
"""

import os
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# Paths and config
# ─────────────────────────────────────────────

PROCESSED_PATH = "data/processed_data.csv"
RAW_PATH       = "data/traffic_data.csv"
MODEL_PATH     = "models/congestion_model.pkl"
SCALER_PATH    = "models/scaler.pkl"

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
# Step 1 — Load data
# ─────────────────────────────────────────────

def load_data():

    print("\n" + "=" * 50)
    print("  📂  Loading dataset...")

    # prefer processed; fall back to raw
    if os.path.exists(PROCESSED_PATH):
        df   = pd.read_csv(PROCESSED_PATH)
        path = PROCESSED_PATH
    elif os.path.exists(RAW_PATH):
        df   = pd.read_csv(RAW_PATH)
        path = RAW_PATH
        print("  ⚠️   Processed data not found — using raw.")
    else:
        raise FileNotFoundError(
            "No dataset found. "
            "Run dataset_generator.py first."
        )

    print(f"  Loaded   : {df.shape}  from '{path}'")
    print(f"  Congested: {df[TARGET].sum()} / {len(df)}")

    return df


# ─────────────────────────────────────────────
# Step 2 — Split
# ─────────────────────────────────────────────

def split(df):

    print("\n  ✂️   Splitting data  (80% train / 20% test)...")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"  Train    : {X_train_sc.shape[0]} samples")
    print(f"  Test     : {X_test_sc.shape[0]} samples")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────
# Step 3 — Train & compare models
# ─────────────────────────────────────────────

def train_models(X_train, X_test,
                 y_train, y_test):

    print("\n  🤖  Training models...\n")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),
    }

    results    = {}
    best_acc   = 0.0
    best_model = None
    best_name  = ""

    # Header
    print(f"  {'Model':<25} {'Accuracy':>10} "
          f"{'CV Mean':>10} {'CV Std':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8}")

    for name, clf in models.items():

        # Train
        clf.fit(X_train, y_train)

        # Test accuracy
        preds = clf.predict(X_test)
        acc   = accuracy_score(y_test, preds)

        # 5-fold cross-validation on training set
        cv_scores = cross_val_score(
            clf, X_train, y_train,
            cv=5, scoring="accuracy"
        )

        print(
            f"  {name:<25} {acc*100:>9.2f}%"
            f" {cv_scores.mean()*100:>9.2f}%"
            f" {cv_scores.std()*100:>7.2f}%"
        )

        results[name] = {
            "model"   : clf,
            "acc"     : acc,
            "cv_mean" : cv_scores.mean(),
            "preds"   : preds,
        }

        if acc > best_acc:
            best_acc   = acc
            best_model = clf
            best_name  = name

    print(f"\n  🏆  Best model → {best_name} "
          f"({best_acc*100:.2f}%)")

    return results, best_model, best_name


# ─────────────────────────────────────────────
# Step 4 — Detailed evaluation of best model
# ─────────────────────────────────────────────

def evaluate(best_name, best_model,
             X_test, y_test):

    print(f"\n  📋  Detailed report — {best_name}")
    print("─" * 50)

    preds = best_model.predict(X_test)

    print(classification_report(
        y_test, preds,
        target_names=["Normal", "Congested"]
    ))

    cm = confusion_matrix(y_test, preds)

    print("  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Normal  Congested")
    print(f"  Actual Normal  : {cm[0][0]:>6}  {cm[0][1]:>9}")
    print(f"  Actual Congest.: {cm[1][0]:>6}  {cm[1][1]:>9}")


# ─────────────────────────────────────────────
# Step 5 — Feature importances
# ─────────────────────────────────────────────

def feature_importance(model, name):

    if not hasattr(model, "feature_importances_"):
        return

    print(f"\n  📊  Feature Importances — {name}")
    print("─" * 40)

    importances = model.feature_importances_
    pairs = sorted(
        zip(FEATURES, importances),
        key=lambda x: -x[1]
    )

    for feat, imp in pairs:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<20} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────
# Step 6 — Save model + scaler
# ─────────────────────────────────────────────

def save_model(model, scaler):

    os.makedirs("models", exist_ok=True)

    pickle.dump(model,  open(MODEL_PATH,  "wb"))
    pickle.dump(scaler, open(SCALER_PATH, "wb"))

    print(f"\n  💾  Model saved  → '{MODEL_PATH}'")
    print(f"  💾  Scaler saved → '{SCALER_PATH}'")


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def train(processed_path=PROCESSED_PATH):

    df = load_data()

    X_train, X_test, y_train, y_test, scaler = split(df)

    results, best_model, best_name = train_models(
        X_train, X_test, y_train, y_test
    )

    evaluate(best_name, best_model, X_test, y_test)

    feature_importance(best_model, best_name)

    save_model(best_model, scaler)

    print("\n  ✅  Training complete!")
    print("=" * 50)

    return best_model, scaler


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    train()