"""
packet.py
Packet structure for AQM 5G Simulation
"""

import time


class Packet:

    def __init__(self, packet_id, size=1500):

        self.packet_id    = packet_id
        self.size         = size          # bytes
        self.arrival_time = time.time()   # wall-clock arrival
        self.departure_time = None
        self.dropped      = False

    # --------------------------------------------------
    # Mark departure time
    # --------------------------------------------------

    def mark_departure(self):
        self.departure_time = time.time()

    # --------------------------------------------------
    # Sojourn delay in milliseconds
    # --------------------------------------------------

    def get_delay(self):
        if self.departure_time is None:
            return 0.0
        return (self.departure_time - self.arrival_time) * 1000

    # --------------------------------------------------
    # String representation
    # --------------------------------------------------

    def __repr__(self):
        return (
            f"Packet(id={self.packet_id}, "
            f"size={self.size}B, "
            f"dropped={self.dropped})"
        )