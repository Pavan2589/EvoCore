"""
Dynamic Reconfiguration Controller.

Applies configuration changes chosen by the Decision Engine *while the
workload keeps running*.  A reconfiguration is not instantaneous:

    duration  = T_ctrl + max_i(latency_i) + T_flush
    T_flush   = dirty_bytes_evicted / (channels * 12.8 GB/s)   (cache shrink)
    factor    = min_i(factor_i)            throughput retained during transition
    lost      = sum over transition segments of
                    cycles * (1 - X_transitional * factor / X_new)
                i.e. the extra time compared with an ideal *instantaneous* switch to
                the new configuration (measured by the engine each step; the
                decision engine estimates it with its predictor)
    energy    = sum_i(E_i) + |dCores| * E_migr + dirty_MB * E_flush

Components are switched in parallel by independent hardware (PLL, power
gates, cache way-mask, DRAM controller), hence the max().  During the
transition the machine runs on the *transitional* configuration (only the
resources common to old and new configuration) at the reduced factor.  At
completion the new configuration becomes architectural state; cache contents,
retired instructions and workload position are preserved (no restart).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import default_config as K
from simulation.models import ArchConfig, ReconfigRecord

FIELD_TO_COMPONENT = {
    "cores": "cores",
    "freq_ghz": "freq",
    "cache_kb": "cache",
    "mem_channels": "channels",
    "prefetch": "prefetch",
    "policy": "policy",
}


@dataclass
class CostEstimate:
    duration: float
    factor: float
    lost_cycles: float
    energy_uj: float
    flush_kb: float


@dataclass
class Segment:
    config: ArchConfig
    cycles: float
    factor: float
    reconfiguring: bool
    target: Optional[ArchConfig] = None    # configuration being switched to


class ReconfigurationController:
    def __init__(self):
        self.active: Optional[ReconfigRecord] = None
        self.history: list[ReconfigRecord] = []

    @property
    def busy(self) -> bool:
        return self.active is not None

    # ------------------------------------------------------------------
    @staticmethod
    def estimate(old: ArchConfig, new: ArchConfig, resident_kb: float = 0.0,
                 dirty_fraction: float = 0.0) -> CostEstimate:
        changed = old.diff(new)
        if not changed:
            return CostEstimate(0.0, 1.0, 0.0, 0.0, 0.0)
        comps = [K.RECONFIG_COSTS[FIELD_TO_COMPONENT[f]] for f in changed]
        latency = max(c["latency"] for c in comps)
        factor = min(c["factor"] for c in comps)
        energy = sum(c["energy_uj"] for c in comps)
        energy += abs(new.cores - old.cores) * K.CORE_MIGRATION_UJ_PER_CORE
        flush_kb = 0.0
        if new.cache_kb < old.cache_kb and resident_kb > new.cache_kb:
            flush_kb = (resident_kb - new.cache_kb) * dirty_fraction
            bw = min(old.mem_channels, new.mem_channels) * K.CHANNEL_BANDWIDTH_GBPS  # bytes/ns
            latency += flush_kb * 1024.0 / bw
            energy += flush_kb / 1024.0 * K.FLUSH_ENERGY_UJ_PER_MB
        duration = K.RECONFIG_CONTROLLER_CYCLES + latency
        return CostEstimate(duration, factor, duration * (1.0 - factor), energy, flush_kb)

    def begin(self, cycle: float, old: ArchConfig, new: ArchConfig, resident_kb: float,
              dirty_fraction: float, predicted_gain: float, reason: str) -> ReconfigRecord:
        if self.busy:
            raise RuntimeError("reconfiguration already in progress")
        est = self.estimate(old, new, resident_kb, dirty_fraction)
        rec = ReconfigRecord(
            id=len(self.history) + 1,
            start_cycle=cycle,
            end_cycle=cycle + est.duration,
            old=old,
            new=new,
            changed=old.diff(new),
            duration=est.duration,
            throughput_factor=est.factor,
            energy_uj=est.energy_uj,
            predicted_gain=predicted_gain,
            reason=reason,
        )
        self.active = rec
        self.history.append(rec)
        return rec

    def segments(self, t0: float, t1: float, current: ArchConfig) -> list[Segment]:
        """Split [t0, t1) into execution segments around an active transition."""
        rec = self.active
        if rec is None:
            return [Segment(current, t1 - t0, 1.0, False)]
        if rec.end_cycle <= t0:                    # finished before this step began
            return [Segment(rec.new, t1 - t0, 1.0, False)]
        trans = ArchConfig.transitional(rec.old, rec.new)
        segs = []
        r0, r1 = max(t0, rec.start_cycle), min(t1, rec.end_cycle)
        if r0 > t0:
            segs.append(Segment(current, r0 - t0, 1.0, False))
        if r1 > r0:
            segs.append(Segment(trans, r1 - r0, rec.throughput_factor, True, rec.new))
        if t1 > r1:
            segs.append(Segment(rec.new, t1 - r1, 1.0, False))
        return segs

    def account_loss(self, lost_cycles: float) -> None:
        if self.active is not None:
            self.active.lost_cycles += lost_cycles

    def complete_if_due(self, cycle: float) -> Optional[ReconfigRecord]:
        rec = self.active
        if rec is not None and cycle >= rec.end_cycle - 1e-9:
            rec.completed = True
            self.active = None
            return rec
        return None

    # summary helpers ---------------------------------------------------
    def total_lost_cycles(self) -> float:
        return sum(r.lost_cycles for r in self.history)

    def total_duration(self) -> float:
        return sum(r.duration for r in self.history)

    def total_energy(self) -> float:
        return sum(r.energy_uj for r in self.history)
