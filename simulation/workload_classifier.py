"""
Workload Characterization Engine.

Threshold rules (applied to one monitoring sample):

    Compute-bound : cpu >= cpu_high  and  bw <= bw_low  and  miss <= miss_low
    Memory-bound  : cpu <= cpu_low   and miss >= miss_high
                    and (bw >= bw_high or stall >= stall_high)
    Balanced      : otherwise

Two anti-oscillation mechanisms:

1. *Threshold hysteresis* - the thresholds of the class the system is
   currently in are relaxed by ``hysteresis`` so that a sample sitting near a
   boundary does not flip the class.
2. *Consecutive-sample validation* - a new class is only accepted after
   ``confirm_samples`` consecutive samples agree.

Samples taken while a reconfiguration was in progress measure a transitional
machine and are ignored.
"""

from __future__ import annotations

from config import default_config as K
from simulation.models import Classification, Sample

COMPUTE, MEMORY, BALANCED, UNKNOWN = "Compute-bound", "Memory-bound", "Balanced", "Unknown"
CLASSES = (COMPUTE, MEMORY, BALANCED)


class WorkloadClassifier:
    def __init__(self, confirm_samples: int = K.DEFAULT_CONFIRM_SAMPLES,
                 thresholds: dict | None = None, hysteresis: bool = True):
        self.t = dict(K.CLASSIFIER_THRESHOLDS if thresholds is None else thresholds)
        self.confirm_samples = max(1, int(confirm_samples))
        self.hysteresis = hysteresis
        self.stable = UNKNOWN
        self._candidate = None
        self._streak = 0
        self.history: list[tuple[float, str, str]] = []   # (cycle, raw, stable)

    def raw_class(self, s: Sample) -> tuple[str, str]:
        t = self.t
        h = t["hysteresis"] if self.hysteresis else 0.0
        hc = h if self.stable == COMPUTE else 0.0
        hm = h if self.stable == MEMORY else 0.0
        if (s.cpu_util >= t["cpu_high"] - hc and s.mem_bw_util <= t["bw_low"] + hc
                and s.miss_rate <= t["miss_low"] + hc):
            return COMPUTE, (f"CPU {s.cpu_util:.0%} high, memory BW {s.mem_bw_util:.0%} low, "
                             f"LLC miss {s.miss_rate:.0%} low")
        if (s.cpu_util <= t["cpu_low"] + hm and s.miss_rate >= t["miss_high"] - hm
                and (s.mem_bw_util >= t["bw_high"] - hm or s.mem_stall_frac >= t["stall_high"] - hm)):
            return MEMORY, (f"CPU {s.cpu_util:.0%} low, memory BW {s.mem_bw_util:.0%}, "
                            f"LLC miss {s.miss_rate:.0%}, DRAM stalls {s.mem_stall_frac:.0%}")
        return BALANCED, (f"CPU {s.cpu_util:.0%}, memory BW {s.mem_bw_util:.0%}, "
                          f"LLC miss {s.miss_rate:.0%} (moderate)")

    def update(self, s: Sample) -> Classification:
        if s.reconfiguring:
            return Classification(raw=self.stable, stable=self.stable, changed=False,
                                  streak=self._streak, reason="sample discarded (reconfiguration in progress)")
        raw, reason = self.raw_class(s)
        changed = False
        if raw == self.stable:
            self._candidate, self._streak = None, 0
        else:
            if raw == self._candidate:
                self._streak += 1
            else:
                self._candidate, self._streak = raw, 1
            if self._streak >= self.confirm_samples:
                self.stable = raw
                self._candidate, self._streak = None, 0
                changed = True
        self.history.append((s.cycle, raw, self.stable))
        return Classification(raw=raw, stable=self.stable, changed=changed,
                              streak=self._streak, reason=reason)
