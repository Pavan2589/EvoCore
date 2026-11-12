"""
Shared last-level cache (LLC) model.

The cache keeps *architectural state*: ``resident_kb`` is the amount of the
current working set that is actually held in the cache.  It survives
reconfiguration (growing the cache adds empty, cold capacity; shrinking it
evicts data) and is partially invalidated when the workload changes phase.

Equations
---------
Coverage of the working set W by resident data R:

    cov = min(1, R / W)

Demand miss rate (power-law / "sqrt(2) rule" miss curve):

    m = m_min + (m_base - m_min) * (1 - cov^0.5)

Replacement policy (FIFO vs LRU), with temporal locality L_t:

    m_FIFO = m * (1 + 0.15 * (L_t - 0.4))     (LRU better for high reuse,
                                               FIFO cheaper for scans)

Prefetcher (stream detector), with stream fraction s:

    coverage c   = 0.8 * s
    accuracy  a  = 0.5 + 0.4 * s
    pollution    m <- m * (1 + 0.06 * (1 - s))
    exposed      m_exp = m * (1 - c)
    extra traffic       m * c * (1/a - 1)

Warm-up after N_fill KB of lines are brought in (each demand miss installs
its 128 B sector pair, i.e. 2 lines), target T = min(C, W):

    R <- min(T, R + N_fill)
"""

from __future__ import annotations

from dataclasses import dataclass

from config import default_config as K
from simulation.models import WorkloadCharacteristics


@dataclass
class CacheBehaviour:
    demand_miss: float
    exposed_miss: float
    hit_rate: float
    prefetch_coverage: float
    traffic_lines_per_access: float   # lines moved from DRAM per LLC access
    accuracy: float


def prefetch_accuracy(stream_fraction: float) -> float:
    return 0.5 + 0.4 * stream_fraction


def behaviour(resident_kb: float, w: WorkloadCharacteristics, policy: str,
              prefetch: bool) -> CacheBehaviour:
    cov = min(1.0, resident_kb / w.working_set_kb) if w.working_set_kb > 0 else 1.0
    m = w.min_miss_rate + (w.base_miss_rate - w.min_miss_rate) * (1.0 - cov ** K.MISS_CURVE_EXPONENT)
    if policy == "FIFO":
        m *= 1.0 + K.FIFO_LOCALITY_PENALTY * (w.temporal_locality - K.FIFO_LOCALITY_PIVOT)
    c = 0.0
    acc = prefetch_accuracy(w.prefetchability)
    extra = 0.0
    if prefetch:
        m *= 1.0 + K.PREFETCH_POLLUTION * (1.0 - w.prefetchability)
        c = K.PREFETCH_MAX_COVERAGE * w.prefetchability
        extra = m * c * (1.0 / acc - 1.0)
    m = min(max(m, 0.0), 1.0)
    exposed = m * (1.0 - c)
    return CacheBehaviour(
        demand_miss=m,
        exposed_miss=exposed,
        hit_rate=1.0 - exposed,
        prefetch_coverage=c,
        traffic_lines_per_access=(m + extra) * (1.0 + K.WRITEBACK_FACTOR * w.write_fraction),
        accuracy=acc,
    )


class Cache:
    """Stateful LLC: capacity/policy are configuration, resident data is state."""

    def __init__(self, capacity_kb: float, policy: str = "LRU"):
        self.capacity_kb = float(capacity_kb)
        self.policy = policy
        self.resident_kb = 0.0          # cold start
        self.dirty_fraction = 0.0

    def behaviour(self, w: WorkloadCharacteristics, prefetch: bool,
                  capacity_kb: float | None = None) -> CacheBehaviour:
        cap = self.capacity_kb if capacity_kb is None else capacity_kb
        return behaviour(min(self.resident_kb, cap), w, self.policy, prefetch)

    def fill(self, fetched_kb: float, w: WorkloadCharacteristics) -> None:
        target = min(self.capacity_kb, w.working_set_kb)
        if target <= 0 or fetched_kb <= 0:
            return
        if self.resident_kb < target:
            # every demand-missed line is installed until the target is reached
            self.resident_kb = min(target, self.resident_kb + fetched_kb)
        self.dirty_fraction = w.write_fraction

    def on_phase_change(self, new_w: WorkloadCharacteristics) -> None:
        """New working set: only a small overlap of the old contents stays useful."""
        self.resident_kb = min(self.resident_kb * new_w.ws_overlap, new_w.working_set_kb)

    def resize(self, new_capacity_kb: float) -> float:
        """Change capacity.  Returns KB of dirty data that must be written back."""
        flushed = 0.0
        if new_capacity_kb < self.resident_kb:
            evicted = self.resident_kb - new_capacity_kb
            flushed = evicted * self.dirty_fraction
            self.resident_kb = float(new_capacity_kb)
        self.capacity_kb = float(new_capacity_kb)
        return flushed

    def occupancy(self) -> float:
        return min(1.0, self.resident_kb / self.capacity_kb) if self.capacity_kb else 0.0
