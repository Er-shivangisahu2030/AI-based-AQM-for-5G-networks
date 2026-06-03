"""
main.py
AI-Based Active Queue Management in 5G Networks
-------------------------------------------------
Full pipeline controller — runs everything in order:

  Step 1  →  FIFO Simulation
  Step 2  →  RED Simulation
  Step 3  →  CoDel Simulation
  Step 4  →  Dataset Generation
  Step 5  →  Data Preprocessing
  Step 6  →  ML Model Training
  Step 7  →  Congestion Predictor Demo
  Step 8  →  Performance Comparison + Graphs

Run:
    python main.py
"""

import os
import sys
import time


# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────

def banner():
    print("\n" + "═" * 55)
    print("   AI-Based Active Queue Management")
    print("   in 5G Networks")
    print("   Full Pipeline — main.py")
    print("═" * 55)


# ─────────────────────────────────────────────
# Step header
# ─────────────────────────────────────────────

def step(n, title):
    print(f"\n\n{'━' * 55}")
    print(f"   STEP {n} — {title}")
    print(f"{'━' * 55}")
    time.sleep(0.3)


# ─────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────

SIM_STEPS    = 20
ARRIVAL_RATE = 6
SERVICE_RATE = 4
NUM_SAMPLES  = 2000


# ─────────────────────────────────────────────
# Step 1 — FIFO
# ─────────────────────────────────────────────

def run_fifo():
    from fifo import FIFOQueue

    fifo = FIFOQueue(max_size=15)

    result = fifo.simulate(
        simulation_steps = SIM_STEPS,
        arrival_rate     = ARRIVAL_RATE,
        service_rate     = SERVICE_RATE,
    )

    return result, fifo.metrics.queue_lengths[:]


# ─────────────────────────────────────────────
# Step 2 — RED
# ─────────────────────────────────────────────

def run_red():
    from red import REDQueue

    red = REDQueue(
        max_size   = 50,
        min_thresh = 10,
        max_thresh = 30,
        max_prob   = 0.5,
        weight     = 0.002,
    )

    result = red.simulate(
        simulation_steps = SIM_STEPS,
        arrival_rate     = ARRIVAL_RATE,
        service_rate     = SERVICE_RATE,
    )

    return result, red.metrics.queue_lengths[:]


# ─────────────────────────────────────────────
# Step 3 — CoDel
# ─────────────────────────────────────────────

def run_codel():
    from codel import CoDelQueue

    codel = CoDelQueue(
        max_size     = 200,
        target_delay = 5.0,
        interval     = 100.0,
    )

    result = codel.simulate(
        simulation_steps = SIM_STEPS,
        arrival_rate     = ARRIVAL_RATE,
        service_rate     = SERVICE_RATE,
    )

    return result, codel.metrics.queue_lengths[:]


# ─────────────────────────────────────────────
# Step 4 — Dataset generation
# ─────────────────────────────────────────────

def run_dataset():
    from dataset_generator import generate_dataset

    os.makedirs("data", exist_ok=True)

    df = generate_dataset(
        num_samples  = NUM_SAMPLES,
        output_path  = "data/traffic_data.csv",
    )

    return df


# ─────────────────────────────────────────────
# Step 5 — Data preprocessing
# ─────────────────────────────────────────────

def run_preprocessing():
    from processed import run_pipeline

    df, scaler = run_pipeline(
        raw_path       = "data/traffic_data.csv",
        processed_path = "data/processed_data.csv",
    )

    return df, scaler


# ─────────────────────────────────────────────
# Step 6 — ML training
# ─────────────────────────────────────────────

def run_training():
    from train_model import train

    os.makedirs("models", exist_ok=True)

    model, scaler = train(
        processed_path = "data/processed_data.csv",
    )

    return model, scaler


# ─────────────────────────────────────────────
# Step 7 — Predictor demo
# ─────────────────────────────────────────────

def run_predictor():
    from predictor import CongestionPredictor

    print("\n  🤖  Running predictor demo...\n")

    predictor = CongestionPredictor()

    test_cases = [
        (3,  5,  10, 2.0,  5.0,  0.00, "Light traffic   "),
        (15, 12, 10, 12.0, 9.5,  0.05, "Moderate traffic"),
        (45, 22, 8,  40.0, 5.0,  0.30, "Heavy congestion"),
    ]

    print(f"  {'Scenario':<20} {'Congested':<12} "
          f"{'Probability':<14} {'Risk'}")
    print(f"  {'─'*20} {'─'*12} {'─'*14} {'─'*8}")

    for ql, ar, sr, dl, tp, dr, label in test_cases:
        pred, prob = predictor.predict(
            ql, ar, sr, dl, tp, dr
        )
        status = "🔴 YES" if pred == 1 else "🟢 NO"
        risk   = (
            "HIGH"   if prob > 0.75 else
            "MEDIUM" if prob > 0.45 else
            "LOW"
        )
        print(f"  {label:<20} {status:<12} "
              f"{prob:.4f} ({prob*100:.1f}%)  {risk}")

    return predictor


# ─────────────────────────────────────────────
# Step 8 — Performance comparison + graphs
# ─────────────────────────────────────────────

def run_comparison(results, queue_histories):

    print("\n  📊  Performance Comparison Table")
    print("─" * 65)
    print(
        f"  {'Algorithm':<15} {'Loss%':>8} "
        f"{'Avg Delay':>12} {'Avg Queue':>12} "
        f"{'Throughput':>12}"
    )
    print(f"  {'─'*15} {'─'*8} {'─'*12} {'─'*12} {'─'*12}")

    for name, r in results.items():
        print(
            f"  {name:<15} "
            f"{r['loss_rate']:>7.2f}% "
            f"{r['avg_delay_ms']:>11.4f} "
            f"{r['avg_queue']:>12.2f} "
            f"{r['avg_throughput']:>12.2f}"
        )

    print("─" * 65)

    # ── Graphs ────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("graphs", exist_ok=True)
        names  = list(results.keys())
        colors = ["#E24B4A", "#BA7517", "#1D9E75"]

        # Graph 1 — Queue length over time
        fig, ax = plt.subplots(figsize=(9, 4))
        for (name, qlens), c in zip(
            queue_histories.items(), colors
        ):
            ax.plot(qlens, label=name, color=c,
                    linewidth=1.8)
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Queue Length")
        ax.set_title("Queue Length Over Time")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("graphs/queue_length.png", dpi=150)
        plt.close()
        print("\n  📈  Saved → graphs/queue_length.png")

        # Graph 2 — Packet loss comparison
        loss_vals = [
            results[n]["loss_rate"] for n in names
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names, loss_vals,
                      color=colors, width=0.4,
                      edgecolor="white")
        for bar, val in zip(bars, loss_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.2f}%",
                ha="center", fontsize=10
            )
        ax.set_ylabel("Packet Loss (%)")
        ax.set_title("Packet Loss Rate Comparison")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("graphs/packet_loss.png", dpi=150)
        plt.close()
        print("  📈  Saved → graphs/packet_loss.png")

        # Graph 3 — Average delay comparison
        delay_vals = [
            results[n]["avg_delay_ms"] for n in names
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names, delay_vals,
                      color=colors, width=0.4,
                      edgecolor="white")
        for bar, val in zip(bars, delay_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", fontsize=10
            )
        ax.set_ylabel("Avg Delay (ms)")
        ax.set_title("Average Delay Comparison")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("graphs/avg_delay.png", dpi=150)
        plt.close()
        print("  📈  Saved → graphs/avg_delay.png")

        print("\n  ✅  All graphs saved to graphs/ folder.")

    except ImportError:
        print("  ⚠️   matplotlib not installed — "
              "skipping graphs.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    banner()

    # ── Step 1: FIFO ──────────────────────────
    step(1, "FIFO Simulation")
    fifo_result, fifo_qlens = run_fifo()

    # ── Step 2: RED ───────────────────────────
    step(2, "RED Simulation")
    red_result, red_qlens = run_red()

    # ── Step 3: CoDel ─────────────────────────
    step(3, "CoDel Simulation")
    codel_result, codel_qlens = run_codel()

    # ── Step 4: Dataset ───────────────────────
    step(4, "Dataset Generation")
    run_dataset()

    # ── Step 5: Preprocessing ─────────────────
    step(5, "Data Preprocessing")
    run_preprocessing()

    # ── Step 6: Training ──────────────────────
    step(6, "ML Model Training")
    run_training()

    # ── Step 7: Predictor ─────────────────────
    step(7, "Congestion Predictor Demo")
    run_predictor()

    # ── Step 8: Comparison + Graphs ───────────
    step(8, "Performance Comparison & Graphs")

    results = {
        "FIFO" : fifo_result,
        "RED"  : red_result,
        "CoDel": codel_result,
    }

    queue_histories = {
        "FIFO" : fifo_qlens,
        "RED"  : red_qlens,
        "CoDel": codel_qlens,
    }

    run_comparison(results, queue_histories)

    # ── Final summary ─────────────────────────
    print("\n\n" + "═" * 55)
    print("   ✅  Pipeline Complete!")
    print("   Output files:")
    print("     data/traffic_data.csv")
    print("     data/processed_data.csv")
    print("     models/congestion_model.pkl")
    print("     models/scaler.pkl")
    print("     graphs/queue_length.png")
    print("     graphs/packet_loss.png")
    print("     graphs/avg_delay.png")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()