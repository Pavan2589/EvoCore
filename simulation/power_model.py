"""
Power model.

    V(f)     = 0.6 + 0.15 f                         (volts, f in GHz)
    P_core   = P_static + C * f * V(f)^2 * (a_idle + (1 - a_idle) * u)
    P_cache  = 1.0 W/MB * capacity
    P_mem    = k * (1.5 + 2.5 * rho)
    P_pf     = 0.4 W if the prefetcher is on

The power budget is what prevents "max everything" from being optimal and
forces the decision engine to *trade* resources between CPU and memory.
"""

from __future__ import annotations

from config import default_config as K
from simulation.models import ArchConfig


def core_power(freq_ghz: float, utilization: float = 1.0) -> float:
    v = K.VOLTAGE_BASE_V + K.VOLTAGE_SLOPE_V_PER_GHZ * freq_ghz
    dyn = K.CORE_DYN_COEFF * freq_ghz * v * v
    act = K.CORE_IDLE_ACTIVITY + (1.0 - K.CORE_IDLE_ACTIVITY) * min(max(utilization, 0.0), 1.0)
    return K.CORE_STATIC_W + dyn * act


def total_power(cfg: ArchConfig, cpu_util: float = 1.0, bw_util: float = 1.0) -> float:
    p = cfg.cores * core_power(cfg.freq_ghz, cpu_util)
    p += K.CACHE_W_PER_MB * cfg.cache_kb / 1024.0
    p += cfg.mem_channels * (K.CHANNEL_STATIC_W + K.CHANNEL_DYN_W * min(max(bw_util, 0.0), 1.0))
    if cfg.prefetch:
        p += K.PREFETCHER_W
    return p


def peak_power(cfg: ArchConfig) -> float:
    """Worst-case power used to check a configuration against the budget."""
    return total_power(cfg, 1.0, 1.0)
