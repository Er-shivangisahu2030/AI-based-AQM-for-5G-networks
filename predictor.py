"""
predictor.py
Real-time Congestion Predictor
--------------------------------
Loads the saved ML model and scaler,
then predicts congestion probability
for live traffic features.

Used by ml_aqm.py to decide whether
to drop incoming packets proactively.
"""

import os
import pickle

import numpy as np


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

MODEL_PATH  = r"E:\\PROJECT FILE\\AI-based-AQM-for-5G-networks\\models\\congestion_model.pkl"
SCALER_PATH = r"E:\\PROJECT FILE\\AI-based-AQM-for-5G-networks\\models\\scaler.pkl"

FEATURES = [
    "queue_length",
    "arrival_rate",
    "service_rate",
    "avg_delay_ms",
    "throughput",
    "drop_rate",
]


# ─────────────────────────────────────────────
# Predictor class
# ─────────────────────────────────────────────

class CongestionPredictor:

    def __init__(self,
                 model_path=MODEL_PATH,
                 scaler_path=SCALER_PATH):

        self.model  = None
        self.scaler = None

        self._load(model_path, scaler_path)

    # -----------------------------------------
    # Load model and scaler from disk
    # -----------------------------------------

    def _load(self, model_path, scaler_path):

        if not os.path.exists(model_path):
            print(
                "  ⚠️   Model not found. "
                "Training now — this may take a moment..."
            )
            from train_model import train
            self.model, self.scaler = train()
            return

        self.model  = pickle.load(
            open(model_path,  "rb")
        )
        self.scaler = pickle.load(
            open(scaler_path, "rb")
        )

        print(
            f"  ✅  Model loaded  → '{model_path}'\n"
            f"  ✅  Scaler loaded → '{scaler_path}'"
        )

    # -----------------------------------------
    # Predict congestion
    # -----------------------------------------

    def predict(self,
                queue_length,
                arrival_rate,
                service_rate,
                avg_delay_ms,
                throughput,
                drop_rate):
        """
        Returns:
          prediction  : 0 = Normal, 1 = Congested
          probability : float confidence (0.0 – 1.0)
        """

        features = np.array([[
            queue_length,
            arrival_rate,
            service_rate,
            avg_delay_ms,
            throughput,
            drop_rate,
        ]])

        if self.scaler is None or self.model is None:
            raise RuntimeError("Model or scaler not initialized. Check model training.")

        scaled     = self.scaler.transform(features)
        prediction = int(self.model.predict(scaled)[0])
        probability = float(
            self.model.predict_proba(scaled)[0][1]
        )

        return prediction, round(probability, 4)

    # -----------------------------------------
    # Boolean check — is it congested?
    # -----------------------------------------

    def is_congested(self,
                     queue_length,
                     arrival_rate,
                     service_rate,
                     avg_delay_ms,
                     throughput,
                     drop_rate,
                     threshold=0.5):
        """
        Returns (bool, probability).
        True if probability >= threshold.
        """

        _, prob = self.predict(
            queue_length,
            arrival_rate,
            service_rate,
            avg_delay_ms,
            throughput,
            drop_rate,
        )

        return prob >= threshold, prob

    # -----------------------------------------
    # Explain a prediction
    # -----------------------------------------

    def explain(self,
                queue_length,
                arrival_rate,
                service_rate,
                avg_delay_ms,
                throughput,
                drop_rate):
        """
        Print a human-readable prediction summary.
        """

        pred, prob = self.predict(
            queue_length,
            arrival_rate,
            service_rate,
            avg_delay_ms,
            throughput,
            drop_rate,
        )

        status  = "🔴 CONGESTED" if pred == 1 else "🟢 NORMAL"
        risk    = (
            "HIGH"   if prob > 0.75 else
            "MEDIUM" if prob > 0.45 else
            "LOW"
        )

        print(f"\n  ┌── Prediction Result {'─'*20}")
        print(f"  │  Status      : {status}")
        print(f"  │  Probability : {prob:.4f} ({prob*100:.1f}%)")
        print(f"  │  Risk Level  : {risk}")
        print(f"  ├── Input Features {'─'*22}")
        print(f"  │  Queue Length  : {queue_length}")
        print(f"  │  Arrival Rate  : {arrival_rate}")
        print(f"  │  Service Rate  : {service_rate}")
        print(f"  │  Avg Delay ms  : {avg_delay_ms}")
        print(f"  │  Throughput    : {throughput}")
        print(f"  │  Drop Rate     : {drop_rate}")
        print(f"  └{'─'*40}")

        return pred, prob


# ─────────────────────────────────────────────
# Entry point — demo predictions
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("  🤖  Congestion Predictor — Demo")
    print("=" * 50)

    predictor = CongestionPredictor()

    # Test scenarios
    scenarios = [
        {
            "label"        : "Light Traffic",
            "queue_length" : 3,
            "arrival_rate" : 5,
            "service_rate" : 10,
            "avg_delay_ms" : 2.0,
            "throughput"   : 5.0,
            "drop_rate"    : 0.0,
        },
        {
            "label"        : "Moderate Traffic",
            "queue_length" : 15,
            "arrival_rate" : 12,
            "service_rate" : 10,
            "avg_delay_ms" : 12.0,
            "throughput"   : 9.5,
            "drop_rate"    : 0.05,
        },
        {
            "label"        : "Heavy Congestion",
            "queue_length" : 45,
            "arrival_rate" : 22,
            "service_rate" : 8,
            "avg_delay_ms" : 40.0,
            "throughput"   : 5.0,
            "drop_rate"    : 0.30,
        },
        {
            "label"        : "Burst Traffic",
            "queue_length" : 30,
            "arrival_rate" : 18,
            "service_rate" : 10,
            "avg_delay_ms" : 25.0,
            "throughput"   : 7.0,
            "drop_rate"    : 0.15,
        },
    ]

    for s in scenarios:
        print(f"\n  📡  Scenario: {s['label']}")
        predictor.explain(
            queue_length = s["queue_length"],
            arrival_rate = s["arrival_rate"],
            service_rate = s["service_rate"],
            avg_delay_ms = s["avg_delay_ms"],
            throughput   = s["throughput"],
            drop_rate    = s["drop_rate"],
        )

    print("\n  ✅  Demo complete!")
    print("=" * 50)