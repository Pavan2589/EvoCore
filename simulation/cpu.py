"""
CPU Execution Units model.

Equations
---------
Amdahl speed-up of n cores for a workload with parallel fraction p:

    S(n) = 1 / ((1 - p) + p / n)

Per-thread execution time per instruction (ns), excluding DRAM stalls:

    t_exec = [CPI_exec + A_llc * (1 - m) * H(C) * e_hit] / f

    A_llc   LLC accesses per instruction
    m       LLC demand miss rate
    H(C)    LLC hit latency in core cycles, grows with capacity C
    e_hit   fraction of hit latency the out-of-order core cannot hide
    f       core frequency (GHz = cycles per ns)
"""

from __future__ import annotations

import math

from config import default_config as K
from simulation.models import WorkloadCharacteristics


def amdahl_speedup(cores: int, parallel_fraction: float) -> float:
    p = min(max(parallel_fraction, 0.0), 1.0)
    return 1.0 / ((1.0 - p) + p / max(cores, 1))


def parallel_fraction_from_activity(cores: int, activity: float, default: float = 0.9) -> float:
    """Invert Amdahl's law from the observed fraction of busy cores.

    activity = S(n)/n  =>  p = (1 - 1/S) / (1 - 1/n).  Not observable for n == 1.
    """
    if cores <= 1 or activity <= 0:
        return default
    s = activity * cores
    p = (1.0 - 1.0 / s) / (1.0 - 1.0 / cores)
    return min(max(p, 0.0), 1.0)


def llc_hit_cycles(cache_kb: float) -> float:
    return K.LLC_HIT_BASE_CYCLES + K.LLC_HIT_CYCLES_PER_DOUBLING * math.log2(max(cache_kb, 256) / 256)


class CPU:
    """Multi-core CPU with configurable core count and frequency."""

    def __init__(self, cores: int, freq_ghz: float):
        self.cores = cores
        self.freq_ghz = freq_ghz

    def speedup(self, w: WorkloadCharacteristics) -> float:
        return amdahl_speedup(self.cores, w.parallel_fraction)

    def exec_time_per_instr(self, w: WorkloadCharacteristics, demand_miss: float,
                            cache_kb: float) -> float:
        hit_cycles = llc_hit_cycles(cache_kb)
        cycles = w.cpi_exec + w.llc_api * (1.0 - demand_miss) * hit_cycles * K.LLC_HIT_EXPOSURE
        return cycles / self.freq_ghz
