"""
red.py
Random Early Detection (RED) Algorithm
----------------------------------------
Improvement over FIFO:
  - Monitors average queue size continuously
  - Drops packets EARLY and RANDOMLY before
    the queue becomes full
  - Three zones:
      Below min_thresh  → accept all packets
      min to max thresh → probabilistic drop
      Above max_thresh  → drop all packets
"""

import random
import time

from packet import Packet
from metrics import Metrics


# ─────────────────────────────────────────────
# RED Queue
# ─────────────────────────────────────────────

class REDQueue:

    def __init__(self,
                 max_size=50,
                 min_thresh=10,
                 max_thresh=30,
                 max_prob=0.5,
                 weight=0.002):
        """
        max_size   : Physical queue capacity
        min_thresh : Below this → accept all
        max_thresh : Above this → drop all
        max_prob   : Peak drop probability
                     in the linear zone
        weight     : EWMA smoothing factor
                     for average queue size
        """

        self.queue      = []
        self.max_size   = max_size
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.max_prob   = max_prob
        self.weight     = weight

        self.avg_queue  = 0.0     # exponential moving average
        self.metrics    = Metrics()

    # -----------------------------------------
    # Update EWMA of queue length
    # -----------------------------------------

    def _update_avg(self):
        """
        Exponential Weighted Moving Average:
        avg = (1 - w) * avg + w * current_len
        Smooths out short bursts so RED does
        not over-react to momentary spikes.
        """
        self.avg_queue = (
            (1 - self.weight) * self.avg_queue
            + self.weight * len(self.queue)
        )

    # -----------------------------------------
    # Calculate drop probability
    # -----------------------------------------

    def _drop_probability(self):
        """
        Zone 1 — below min_thresh  : p = 0
        Zone 2 — between thresholds: p rises
                 linearly from 0 to max_prob
        Zone 3 — above max_thresh  : p = 1
        """

        if self.avg_queue < self.min_thresh:
            return 0.0

        elif self.avg_queue <= self.max_thresh:
            # Linear ramp up
            slope = self.max_prob / (
                self.max_thresh - self.min_thresh
            )
            return slope * (
                self.avg_queue - self.min_thresh
            )

        else:
            return 1.0

    # -----------------------------------------
    # Enqueue — probabilistic early drop
    # -----------------------------------------

    def enqueue(self, packet):

        self.metrics.total_packets += 1

        # Update smoothed average queue size
        self._update_avg()

        # ── Hard drop — queue physically full ─
        if len(self.queue) >= self.max_size:

            packet.dropped = True
            self.metrics.packet_loss += 1

            print(
                f"  ❌  HARD DROP  | Packet {packet.packet_id:>4} "
                f"| Queue full ({len(self.queue)}/{self.max_size})"
            )
            return False

        # ── Probabilistic early drop ──────────
        drop_prob = self._drop_probability()

        if random.random() < drop_prob:

            packet.dropped = True
            self.metrics.packet_loss += 1

            print(
                f"  ⚠️   EARLY DROP | Packet {packet.packet_id:>4} "
                f"| Avg Queue: {self.avg_queue:.2f} "
                f"| Drop Prob: {drop_prob:.2f}"
            )
            return False

        # ── Accept packet ─────────────────────
        self.queue.append(packet)

        print(
            f"  ✅  IN         | Packet {packet.packet_id:>4} "
            f"| Queue: {len(self.queue)}/{self.max_size} "
            f"| Avg: {self.avg_queue:.2f} "
            f"| P(drop): {drop_prob:.2f}"
        )
        return True

    # -----------------------------------------
    # Dequeue — remove front packet
    # -----------------------------------------

    def dequeue(self):

        if not self.queue:
            return None

        packet = self.queue.pop(0)
        packet.mark_departure()

        delay = packet.get_delay()

        self.metrics.delivered_packets += 1
        self.metrics.record(
            queue_len=len(self.queue),
            delay=delay
        )

        print(
            f"  📤  OUT        | Packet {packet.packet_id:>4} "
            f"| Delay: {delay:.4f} ms "
            f"| Queue: {len(self.queue)}/{self.max_size}"
        )
        return packet

    # -----------------------------------------
    # Run simulation
    # -----------------------------------------

    def simulate(self,
                 simulation_steps=20,
                 arrival_rate=6,
                 service_rate=4):

        print("\n" + "=" * 50)
        print("   🚦  RED Simulation Starting...")
        print(f"   Max Queue   : {self.max_size}")
        print(f"   Min Thresh  : {self.min_thresh}")
        print(f"   Max Thresh  : {self.max_thresh}")
        print(f"   Max Prob    : {self.max_prob}")
        print(f"   Steps       : {simulation_steps}")
        print(f"   Arrival     : up to {arrival_rate} pkts/step")
        print(f"   Service     : {service_rate} pkts/step")
        print("=" * 50)

        packet_id = 1

        for step in range(simulation_steps):

            print(
                f"\n┌── Step {step + 1:>2}/{simulation_steps} "
                f"{'─' * 33}"
            )

            # ── Arrivals ──────────────────────
            arrivals = random.randint(1, arrival_rate)
            print(f"│  Arriving: {arrivals} packet(s)")

            for _ in range(arrivals):
                pkt = Packet(packet_id)
                self.enqueue(pkt)
                packet_id += 1

            # ── Service ───────────────────────
            served = min(service_rate, len(self.queue))
            print(f"│  Serving : {served} packet(s)")

            for _ in range(served):
                self.dequeue()

            # ── Throughput ────────────────────
            self.metrics.record_throughput(served)

            print(
                f"└── Queue: {len(self.queue)} "
                f"| Avg Queue: {self.avg_queue:.2f} "
                f"| Dropped so far: {self.metrics.packet_loss}"
            )

            time.sleep(0.05)

        print("\n🎯  RED Simulation Complete!")
        return self.metrics.summary("RED")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    red = REDQueue(
        max_size=50,
        min_thresh=10,
        max_thresh=30,
        max_prob=0.5,
        weight=0.002
    )

    red.simulate(
        simulation_steps=20,
        arrival_rate=6,
        service_rate=4
    )