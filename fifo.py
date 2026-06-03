"""
fifo.py
FIFO Queue Simulation for AQM 5G Project
-----------------------------------------
Baseline algorithm — tail-drop when full.
No congestion awareness; packets dropped
only when the queue is completely full.
"""
 
import random
import time
 
from packet import Packet
from metrics import Metrics
 
 
# ─────────────────────────────────────────────
# FIFO Queue
# ─────────────────────────────────────────────
 
class FIFOQueue:
 
    def __init__(self, max_size=20):
        """
        max_size : Maximum number of packets
                   the queue can hold at once.
        """
        self.queue    = []
        self.max_size = max_size
        self.metrics  = Metrics()
 
    # -----------------------------------------
    # Enqueue — add packet to queue
    # -----------------------------------------
 
    def enqueue(self, packet):
 
        self.metrics.total_packets += 1
 
        # ── Queue full → tail drop ────────────
        if len(self.queue) >= self.max_size:
 
            packet.dropped = True
            self.metrics.packet_loss += 1
 
            print(
                f"  ❌  DROP  | Packet {packet.packet_id:>4} "
                f"| Queue full ({len(self.queue)}/{self.max_size})"
            )
            return False
 
        # ── Accept packet ─────────────────────
        self.queue.append(packet)
 
        print(
            f"  ✅  IN    | Packet {packet.packet_id:>4} "
            f"| Queue: {len(self.queue)}/{self.max_size}"
        )
        return True
 
    # -----------------------------------------
    # Dequeue — remove & process front packet
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
            f"  📤  OUT   | Packet {packet.packet_id:>4} "
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
        """
        simulation_steps : Number of time steps
        arrival_rate     : Max packets per step
        service_rate     : Packets served per step
        """
 
        print("\n" + "=" * 45)
        print("   🚦  FIFO Simulation Starting...")
        print(f"   Max Queue : {self.max_size}")
        print(f"   Steps     : {simulation_steps}")
        print(f"   Arrival   : up to {arrival_rate} pkts/step")
        print(f"   Service   : {service_rate} pkts/step")
        print("=" * 45)
 
        packet_id = 1
 
        for step in range(simulation_steps):
 
            print(f"\n┌── Step {step + 1:>2}/{simulation_steps} "
                  f"{'─' * 28}")
 
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
                f"└── Queue now: {len(self.queue)} | "
                f"Total dropped: {self.metrics.packet_loss}"
            )
 
            time.sleep(0.05)   # small pause for readability
 
        # ── Final summary ─────────────────────
        print("\n🎯  FIFO Simulation Complete!")
        return self.metrics.summary("FIFO")
 
 
# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
 
    fifo = FIFOQueue(max_size=15)
 
    fifo.simulate(
        simulation_steps=20,
        arrival_rate=6,
        service_rate=4
    )