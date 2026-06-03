"""
metrics.py
Metrics collector for AQM 5G Simulation
"""


class Metrics:

    def __init__(self):

        self.total_packets     = 0
        self.delivered_packets = 0
        self.packet_loss       = 0

        self.queue_lengths     = []
        self.delays            = []
        self.throughput_values = []

    # --------------------------------------------------
    # Record per-step queue length and delay
    # --------------------------------------------------

    def record(self, queue_len, delay=None):

        self.queue_lengths.append(queue_len)

        if delay is not None and delay > 0:
            self.delays.append(delay)

    # --------------------------------------------------
    # Record throughput for a step
    # --------------------------------------------------

    def record_throughput(self, tp):
        self.throughput_values.append(tp)

    # --------------------------------------------------
    # Computed averages
    # --------------------------------------------------

    def avg_delay(self):
        if not self.delays:
            return 0.0
        return sum(self.delays) / len(self.delays)

    def avg_queue(self):
        if not self.queue_lengths:
            return 0.0
        return sum(self.queue_lengths) / len(self.queue_lengths)

    def avg_throughput(self):
        if not self.throughput_values:
            return 0.0
        return sum(self.throughput_values) / len(self.throughput_values)

    def loss_rate(self):
        if self.total_packets == 0:
            return 0.0
        return (self.packet_loss / self.total_packets) * 100

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    def summary(self, algo_name="Algorithm"):

        print(f"\n{'=' * 45}")
        print(f"  {algo_name} — Performance Summary")
        print(f"{'=' * 45}")
        print(f"  Total Packets     : {self.total_packets}")
        print(f"  Delivered         : {self.delivered_packets}")
        print(f"  Dropped           : {self.packet_loss}")
        print(f"  Packet Loss Rate  : {self.loss_rate():.2f}%")
        print(f"  Avg Queue Length  : {self.avg_queue():.2f}")
        print(f"  Avg Delay (ms)    : {self.avg_delay():.4f}")
        print(f"  Avg Throughput    : {self.avg_throughput():.2f} pkts/step")
        print(f"{'=' * 45}\n")

        return {
            "algorithm"      : algo_name,
            "total"          : self.total_packets,
            "delivered"      : self.delivered_packets,
            "dropped"        : self.packet_loss,
            "loss_rate"      : round(self.loss_rate(), 2),
            "avg_queue"      : round(self.avg_queue(), 2),
            "avg_delay_ms"   : round(self.avg_delay(), 4),
            "avg_throughput" : round(self.avg_throughput(), 2),
        }