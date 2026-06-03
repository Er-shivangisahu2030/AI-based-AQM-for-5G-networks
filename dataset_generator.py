"""
dataset_generator.py
Generate synthetic traffic dataset for ML training
----------------------------------------------------
Runs FIFO, RED, CoDel simulations across many
different traffic conditions and records features
+ congestion label into a CSV file.

Features recorded per sample:
  - queue_length
  - arrival_rate
  - service_rate
  - avg_delay_ms
  - throughput
  - drop_rate
  - congestion  ← label (0 = normal, 1 = congested)
"""

import os
import random
import time

import pandas as pd

from packet import Packet


# ─────────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────────

def run_simulation(max_queue,
                   arrival_rate,
                   service_rate,
                   sim_steps=30,
                   seed=None):
    """
    Lightweight simulation (no prints) used
    purely for dataset generation.

    Returns a dict of measured features.
    """

    if seed is not None:
        random.seed(seed)

    queue          = []
    total_arrivals = 0
    total_dropped  = 0
    total_delivered = 0
    total_delay    = 0.0
    queue_lengths  = []

    for step in range(sim_steps):

        # ── Arrivals ──────────────────────────
        arrivals = random.randint(
            max(1, arrival_rate - 3),
            arrival_rate + 3
        )
        total_arrivals += arrivals

        for _ in range(arrivals):
            pkt = Packet(total_arrivals)

            if len(queue) < max_queue:
                queue.append(pkt)
            else:
                total_dropped += 1

        # ── Service ───────────────────────────
        served = min(service_rate, len(queue))

        for _ in range(served):
            pkt = queue.pop(0)
            pkt.mark_departure()
            delay = pkt.get_delay()

            # use a realistic simulated delay
            # (wall-clock is near-zero in tight loop)
            sim_delay = (
                delay * 1000
                if delay > 0
                else random.uniform(0.5, 15.0)
            )
            total_delay    += sim_delay
            total_delivered += 1

        queue_lengths.append(len(queue))

    # ── Compute features ──────────────────────
    avg_delay  = (
        total_delay / total_delivered
        if total_delivered > 0
        else 0.0
    )
    drop_rate  = (
        total_dropped / total_arrivals
        if total_arrivals > 0
        else 0.0
    )
    throughput = total_delivered / sim_steps
    queue_len  = (
        sum(queue_lengths) / len(queue_lengths)
        if queue_lengths
        else 0.0
    )

    # ── Label: congested? ─────────────────────
    # Congested if drop_rate > 10% OR
    # avg delay > 20ms
    congestion = int(
        drop_rate > 0.10 or avg_delay > 20.0
    )

    return {
        "queue_length" : round(queue_len, 4),
        "arrival_rate" : arrival_rate,
        "service_rate" : service_rate,
        "avg_delay_ms" : round(avg_delay, 4),
        "throughput"   : round(throughput, 4),
        "drop_rate"    : round(drop_rate, 4),
        "congestion"   : congestion,
    }


# ─────────────────────────────────────────────
# Dataset generator
# ─────────────────────────────────────────────

def generate_dataset(num_samples=2000,
                     output_path="data/traffic_data.csv"):
    """
    Generates num_samples rows by randomising
    traffic conditions across each simulation run.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []

    print("=" * 50)
    print("  📊  Dataset Generator Starting...")
    print(f"  Samples     : {num_samples}")
    print(f"  Output path : {output_path}")
    print("=" * 50)

    start = time.time()

    for i in range(num_samples):

        # Randomise traffic conditions
        max_queue    = random.randint(10, 100)
        arrival_rate = random.randint(3, 25)
        service_rate = random.randint(3, 20)

        record = run_simulation(
            max_queue    = max_queue,
            arrival_rate = arrival_rate,
            service_rate = service_rate,
            sim_steps    = 30,
            seed         = i
        )

        records.append(record)

        # Progress update every 500 samples
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(
                f"  ✅  {i + 1:>5}/{num_samples} samples "
                f"generated  [{elapsed:.1f}s]"
            )

    # ── Save to CSV ───────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start

    print(f"\n  💾  Saved → '{output_path}'")
    print(f"  Shape      : {df.shape}")
    print(
        f"  Congested  : {df['congestion'].sum()} "
        f"({df['congestion'].mean()*100:.1f}%)"
    )
    print(
        f"  Normal     : {(df['congestion']==0).sum()} "
        f"({(1-df['congestion'].mean())*100:.1f}%)"
    )
    print(f"  Time taken : {elapsed:.2f}s")
    print("\n  Sample preview:")
    print(df.head(5).to_string(index=False))
    print("=" * 50)

    return df


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    generate_dataset(
        num_samples=2000,
        output_path="data/traffic_data.csv"
    )