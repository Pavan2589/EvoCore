"""
Memory Controller + Main Memory model.

Equations
---------
Peak bandwidth with k active channels (bytes/ns == GB/s):

    B_peak = k * 12.8

Bus utilisation for demanded bandwidth D:

    rho = D / B_peak

Loaded access latency (queueing delay grows as the bus saturates):

    L(rho) = L_0 + Q * rho / (1 - rho)          (L_0 = 75 ns, Q = 30 ns)

Queue occupancy by Little's law, requests/ns * latency:

    N_q = (D / 64) * L ;    queue_util = N_q / (32 * k)
"""

from __future__ import annotations

from config import default_config as K


class MemoryController:
    def __init__(self, channels: int):
        self.channels = channels

    @property
    def peak_bandwidth(self) -> float:
        return self.channels * K.CHANNEL_BANDWIDTH_GBPS

    def utilization(self, demand_gbps: float) -> float:
        return demand_gbps / self.peak_bandwidth

    @staticmethod
    def latency_ns(rho: float) -> float:
        rho = min(max(rho, 0.0), K.MAX_BUS_UTILIZATION)
        return K.DRAM_UNLOADED_LATENCY_NS + K.QUEUE_DELAY_SCALE_NS * rho / (1.0 - rho)

    def queue_utilization(self, demand_gbps: float, latency_ns: float) -> float:
        outstanding = demand_gbps / K.CACHE_LINE_BYTES * latency_ns
        return min(1.0, outstanding / (K.QUEUE_ENTRIES_PER_CHANNEL * self.channels))
