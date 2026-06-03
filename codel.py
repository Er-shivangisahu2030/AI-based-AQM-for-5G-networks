"""
codel.py
CoDel (Controlled Delay) Algorithm
-------------------------------------
Improvement over RED:
  - Does NOT watch queue SIZE
  - Watches how long packets WAIT (sojourn time)
  - If delay stays above target for too long,
    it enters a dropping state and drops packets
    at an increasing rate until delay recovers
  - Self-tuning: no manual threshold tweaking
  - Specifically designed to fight bufferbloat
    in low-latency networks like 5G
"""

import random
import math
import time

from packet import Packet
from metrics import Metrics


# ─────────────────────────────────────────────
# CoDel Queue
# ─────────────────────────────────────────────

class CoDelQueue:

    def __init__(self,
                 max_size=200,
                 target_delay=5.0,
                 interval=100.0):
        """
        max_size    : Large buffer (CoDel manages
                      delay, not size)
        target_delay: Acceptable sojourn time (ms)
                      5ms is ideal for 5G
        interval    : Observation window (ms).
                      If delay stays above target
                      for this long, start dropping.
        """

        self.queue        = []
        self.max_size     = max_size
        self.target_delay = target_delay
        self.interval     = interval

        # ── CoDel internal state ──────────────
        self.drop_state      = False  # are we dropping?
        self.drop_next_ms    = 0.0    # next scheduled drop time
        self.drop_count      = 0      # consecutive drops so far
        self.first_above_time = None  # when delay first exceeded target

        # simulation clock in ms
        self.now_ms   = 0.0
        self.step_ms  = 10.0          # each step = 10 ms

        self.metrics  = Metrics()

    # -----------------------------------------
    # Control law — next drop time
    # -----------------------------------------

    def _control_law(self, t):
        """
        Schedule next drop at:
            t + interval / sqrt(drop_count)
        More drops have occurred → shorter gap
        → drops accelerate until delay recovers.
        """
        return t + self.interval / math.sqrt(self.drop_count)

    # -----------------------------------------
    # Enqueue — CoDel uses large buffers
    # -----------------------------------------

    def enqueue(self, packet):

        self.metrics.total_packets += 1

        # Store simulation arrival time (ms)
        packet.sim_arrival_ms = self.now_ms

        if len(self.queue) >= self.max_size:

            packet.dropped = True
            self.metrics.packet_loss += 1

            print(
                f"  ❌  HARD DROP  | Packet {packet.packet_id:>4} "
                f"| Buffer full ({len(self.queue)}/{self.max_size})"
            )
            return False

        self.queue.append(packet)

        print(
            f"  ✅  IN         | Packet {packet.packet_id:>4} "
            f"| Queue: {len(self.queue)}/{self.max_size}"
        )
        return True

    # -----------------------------------------
    # Dequeue — delay-aware dropping
    # -----------------------------------------

    def dequeue(self):

        if not self.queue:
            self.drop_state       = False
            self.first_above_time = None
            return None

        packet   = self.queue[0]
        sojourn  = self.now_ms - packet.sim_arrival_ms

        # ── Check if delay is acceptable ──────
        if sojourn < self.target_delay:
            # Delay is fine — reset tracking
            self.first_above_time = None
            ok_to_drop = False
        else:
            # Delay is too high
            if self.first_above_time is None:
                # Start the interval timer
                self.first_above_time = (
                    self.now_ms + self.interval
                )
                ok_to_drop = False
            else:
                # Have we been above target long enough?
                ok_to_drop = (
                    self.now_ms >= self.first_above_time
                )

        # ── Dropping state logic ──────────────
        if self.drop_state:

            while (self.queue
                   and self.now_ms >= self.drop_next_ms):

                # Drop front packet
                drop_pkt = self.queue.pop(0)
                drop_pkt.dropped = True
                self.metrics.packet_loss += 1
                self.drop_count += 1
                self.drop_next_ms = self._control_law(
                    self.drop_next_ms
                )

                print(
                    f"  ⚠️   CODEL DROP | Packet {drop_pkt.packet_id:>4} "
                    f"| Sojourn: {sojourn:.2f} ms "
                    f"| Drop #{self.drop_count}"
                )

                if not ok_to_drop:
                    self.drop_state = False
                    break

            if not self.queue:
                return None

        elif ok_to_drop:

            # Enter dropping state
            drop_pkt = self.queue.pop(0)
            drop_pkt.dropped = True
            self.metrics.packet_loss += 1

            self.drop_state = True
            self.drop_count = max(
                1,
                self.drop_count
                - int(math.sqrt(self.drop_count))
            )
            self.drop_next_ms = self._control_law(
                self.now_ms
            )

            print(
                f"  ⚠️   ENTER DROP | Packet {drop_pkt.packet_id:>4} "
                f"| Sojourn: {sojourn:.2f} ms "
                f"| Entering drop state"
            )

            if not self.queue:
                return None

        # ── Normal dequeue ────────────────────
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
            f"| Sojourn: {sojourn:.2f} ms "
            f"| Queue: {len(self.queue)}"
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
        print("   🚦  CoDel Simulation Starting...")
        print(f"   Max Buffer   : {self.max_size}")
        print(f"   Target Delay : {self.target_delay} ms")
        print(f"   Interval     : {self.interval} ms")
        print(f"   Steps        : {simulation_steps}")
        print(f"   Arrival      : up to {arrival_rate} pkts/step")
        print(f"   Service      : {service_rate} pkts/step")
        print("=" * 50)

        packet_id = 1

        for step in range(simulation_steps):

            # Advance simulation clock
            self.now_ms += self.step_ms

            print(
                f"\n┌── Step {step + 1:>2}/{simulation_steps} "
                f"[t={self.now_ms:.0f}ms] "
                f"{'─' * 22}"
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

            drop_state_label = (
                "🔴 DROPPING" if self.drop_state
                else "🟢 Normal"
            )

            print(
                f"└── Queue: {len(self.queue)} "
                f"| State: {drop_state_label} "
                f"| Dropped: {self.metrics.packet_loss}"
            )

            time.sleep(0.05)

        # Drain remaining packets
        print("\n⏳  Draining remaining queue...")
        while self.queue:
            self.now_ms += self.step_ms
            self.dequeue()

        print("\n🎯  CoDel Simulation Complete!")
        return self.metrics.summary("CoDel")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    codel = CoDelQueue(
        max_size=200,
        target_delay=5.0,
        interval=100.0
    )

    codel.simulate(
        simulation_steps=20,
        arrival_rate=6,
        service_rate=4
    )