"""
Runtime Monitoring Unit.

Models a hardware performance-monitoring unit (PMU).  Every engine step it
accumulates counters; every ``interval`` steps it closes a sample window and
emits a :class:`Sample` of averaged metrics.

* Utilisation-type metrics (CPU util, bandwidth, latency, ...) are averaged
  weighted by elapsed cycles.
* Per-instruction metrics (CPI stack, miss rates, bytes/instruction) are
  averaged weighted by retired instructions.

Each sample costs ``overhead_cycles`` of execution time (sampling interrupt +
counter reads); the engine charges this as monitoring overhead.
"""

from __future__ import annotations

from typing import Optional

from config import default_config as K
from simulation.models import ArchConfig, Sample

_TIME_WEIGHTED = ("cpu_util", "core_activity", "mem_bw_util", "mem_latency_ns",
                  "queue_util", "resource_util", "mem_stall_frac")
_INSTR_WEIGHTED = ("hit_rate", "miss_rate", "exposed_miss_rate", "t_exec_ns", "t_mem_ns",
                   "llc_api", "bytes_per_instr", "stream_fraction")


class RuntimeMonitor:
    def __init__(self, interval: int, overhead_cycles: float = K.MONITOR_OVERHEAD_CYCLES):
        if interval < 1:
            raise ValueError("sampling interval must be >= 1 step")
        self.interval = interval
        self.overhead_cycles = overhead_cycles
        self.samples: list[Sample] = []
        self._reset_window()

    def _reset_window(self) -> None:
        self._steps = 0
        self._cycles = 0.0
        self._instr = 0.0
        self._tw = {k: 0.0 for k in _TIME_WEIGHTED}
        self._iw = {k: 0.0 for k in _INSTR_WEIGHTED}
        self._reconfiguring = False
        self._core_cycles = 0.0   # sum of f * cycles, for IPC

    def observe(self, counters: dict, cycles: float, instructions: float, freq_ghz: float,
                reconfiguring: bool, cycle_now: float, config: ArchConfig) -> Optional[Sample]:
        """Feed one step of counters.  Returns a Sample when a window closes."""
        self._steps += 1
        self._cycles += cycles
        self._instr += instructions
        self._core_cycles += cycles * freq_ghz
        for k in _TIME_WEIGHTED:
            self._tw[k] += counters[k] * cycles
        for k in _INSTR_WEIGHTED:
            self._iw[k] += counters[k] * instructions
        self._reconfiguring |= reconfiguring
        if self._steps < self.interval:
            return None
        return self._close(cycle_now, config)

    def _close(self, cycle_now: float, config: ArchConfig) -> Sample:
        c = max(self._cycles, 1e-9)
        n = max(self._instr, 1e-9)
        tw = {k: v / c for k, v in self._tw.items()}
        iw = {k: v / n for k, v in self._iw.items()}
        s = Sample(
            index=len(self.samples),
            cycle=cycle_now,
            steps=self._steps,
            instructions=self._instr,
            throughput=self._instr / c,
            ipc=self._instr / max(self._core_cycles, 1e-9),
            cpu_util=tw["cpu_util"],
            core_activity=tw["core_activity"],
            mem_bw_util=tw["mem_bw_util"],
            hit_rate=iw["hit_rate"],
            miss_rate=iw["miss_rate"],
            exposed_miss_rate=iw["exposed_miss_rate"],
            mem_latency_ns=tw["mem_latency_ns"],
            queue_util=tw["queue_util"],
            resource_util=tw["resource_util"],
            mem_stall_frac=tw["mem_stall_frac"],
            t_exec_ns=iw["t_exec_ns"],
            t_mem_ns=iw["t_mem_ns"],
            llc_api=iw["llc_api"],
            bytes_per_instr=iw["bytes_per_instr"],
            stream_fraction=iw["stream_fraction"],
            reconfiguring=self._reconfiguring,
            config=config,
        )
        self.samples.append(s)
        self._reset_window()
        return s
