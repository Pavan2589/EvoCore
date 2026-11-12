"""
Discrete-time simulation engine.

One engine step simulates ``step_cycles`` reference cycles.  Per step:

  1. Workload Generator   -> characteristics of the current phase
  2. Reconfiguration Ctrl -> execution segments (normal / transitional)
  3. Performance model    -> CPU <-> cache <-> memory operating point per segment
  4. Retire instructions, warm the cache, advance the workload (phase changes)
  5. Complete any reconfiguration that finished in this step
  6. Runtime Monitor      -> closes a sample every ``sampling_interval`` steps
  7. Classifier -> Decision Engine -> Reconfiguration Controller (adaptive only)

Monitoring and decision overheads are charged as cycles stolen from the next
step.  The static architecture runs exactly the same loop with the adaptive
path (6-7) disabled, so both execute identical workloads.
"""

from __future__ import annotations

from typing import Optional

from config import default_config as K
from simulation.cache import Cache
from simulation.decision_engine import DecisionEngine
from simulation.models import (ArchConfig, Event, SimulationParams, StepRecord, fmt_cache)
from simulation.performance_model import PerformanceModel
from simulation.reconfiguration_controller import ReconfigurationController
from simulation.runtime_monitor import RuntimeMonitor
from simulation.workload_classifier import UNKNOWN, WorkloadClassifier
from simulation.workload_generator import PhaseSpec, WorkloadGenerator

_FIELD_LABEL = {
    "cores": ("CPU core allocation", lambda v: f"{v}"),
    "freq_ghz": ("CPU frequency", lambda v: f"{v:.1f} GHz"),
    "cache_kb": ("LLC capacity", fmt_cache),
    "mem_channels": ("Memory channels", lambda v: f"{v}"),
    "prefetch": ("Memory prefetcher", lambda v: "enabled" if v else "disabled"),
    "policy": ("Cache replacement policy", lambda v: v),
}


class SimulationEngine:
    def __init__(self, phases: list[PhaseSpec], params: SimulationParams | None = None,
                 adaptive: bool = True, name: str | None = None):
        self.params = params or SimulationParams()
        self.adaptive = adaptive
        self.name = name or ("Adaptive" if adaptive else "Static")
        p = self.params

        self.workload = WorkloadGenerator(phases, noise=p.noise, seed=p.seed)
        self.config: ArchConfig = p.initial_config
        self.cache = Cache(self.config.cache_kb, self.config.policy)
        self.model = PerformanceModel()
        self.monitor = RuntimeMonitor(p.sampling_interval)
        self.classifier = WorkloadClassifier(p.confirm_samples, hysteresis=p.hysteresis)
        self.decider = DecisionEngine(p)
        self.controller = ReconfigurationController()

        self.cycle = 0.0
        self.step_index = 0
        self.finished = False
        self.history: list[StepRecord] = []
        self.events: list[Event] = []
        self.phase_changes: list[tuple[float, int, str]] = [(0.0, 0, self.workload.current_phase.label)]

        self.total_energy_uj = 0.0
        self.monitor_overhead = 0.0
        self.decision_overhead = 0.0
        self._pending_monitor = 0.0
        self._pending_decision = 0.0
        self._last_activity = "normal"

        self.log(0, "init", f"{self.name} architecture initialised: {self.config.describe()}")
        self.log(0, "phase", f"Workload started in phase 1: {self.workload.current_phase.label} "
                             f"({self.workload.current_phase.instructions / 1e6:,.2f} M instructions)")

    # ------------------------------------------------------------------
    def log(self, cycle: float, category: str, message: str) -> None:
        self.events.append(Event(cycle, category, message))

    @property
    def reconfiguring(self) -> bool:
        return self.controller.busy

    # ------------------------------------------------------------------
    def step(self) -> bool:
        """Advance one step.  Returns False once the workload has completed."""
        if self.finished:
            return False
        p = self.params
        w = self.workload.characteristics()
        t0 = self.cycle
        step_cycles = float(p.step_cycles)

        overhead_now = self._pending_monitor + self._pending_decision
        mon_now, dec_now = self._pending_monitor, self._pending_decision
        self._pending_monitor = self._pending_decision = 0.0
        avail = max(0.0, 1.0 - overhead_now / step_cycles)

        segments = self.controller.segments(t0, t0 + step_cycles, self.config)
        instr = 0.0
        lost = 0.0
        acc = {k: 0.0 for k in ("cpu_util", "core_activity", "mem_bw_util", "mem_bw_gbps",
                                "mem_latency_ns", "queue_util", "mem_stall_frac", "power_w")}
        iacc = {k: 0.0 for k in ("hit_rate", "miss_rate", "exposed_miss_rate", "t_exec_ns",
                                 "t_mem_ns", "llc_api", "bytes_per_instr", "stream_fraction")}
        core_cycles = 0.0
        fetched_kb = 0.0
        any_reconf = False
        for seg in segments:
            perf = self.model.evaluate(seg.config, w, self.cache)
            factor = seg.factor * avail
            seg_instr = perf.throughput * seg.cycles * factor
            instr += seg_instr
            if seg.reconfiguring:
                any_reconf = True
                # Overhead vs an ideal instantaneous switch: work deficit relative to
                # what the new configuration would retire in the same time.
                x_new = self.model.evaluate(seg.target, w, self.cache).throughput
                seg_lost = seg.cycles * max(0.0, 1.0 - perf.throughput * seg.factor / x_new)
                lost += seg_lost
                self.controller.account_loss(seg_lost)
            wgt = seg.cycles
            for k in acc:
                acc[k] += getattr(perf, k) * wgt
            # utilisation metrics scale with retained throughput during stalls
            acc["cpu_util"] += perf.cpu_util * wgt * (factor - 1.0)
            acc["mem_bw_util"] += perf.mem_bw_util * wgt * (factor - 1.0)
            acc["mem_bw_gbps"] += perf.mem_bw_gbps * wgt * (factor - 1.0)
            for k, attr in (("hit_rate", "hit_rate"), ("miss_rate", "demand_miss_rate"),
                            ("exposed_miss_rate", "exposed_miss_rate"), ("t_exec_ns", "t_exec_ns"),
                            ("t_mem_ns", "t_mem_ns"), ("llc_api", "llc_api"),
                            ("bytes_per_instr", "bytes_per_instr"), ("stream_fraction", "stream_fraction")):
                iacc[k] += getattr(perf, attr) * max(seg_instr, 1e-12)
            core_cycles += seg.cycles * seg.config.freq_ghz
            fetched_kb += (seg_instr * perf.llc_api * perf.demand_miss_rate * K.CACHE_LINE_BYTES
                           * K.ADJACENT_LINE_FILL / 1024.0)

        # Final (partial) step: only run as long as needed to retire the remaining work.
        frac = 1.0
        remaining = self.workload.remaining
        if instr >= remaining and instr > 0:
            frac = remaining / instr
            instr = remaining
            fetched_kb *= frac
            lost *= frac
        cycles = step_cycles * frac
        t1 = t0 + cycles

        m = {k: v / step_cycles for k, v in acc.items()}
        tot_i = max(instr / frac if frac > 0 else instr, 1e-12)
        im = {k: v / tot_i for k, v in iacc.items()}
        ipc = instr / max(core_cycles * frac, 1e-12)

        self.cache.fill(fetched_kb, w)
        energy = m["power_w"] * cycles / 1000.0          # W * ns = nJ -> uJ
        self.total_energy_uj += energy
        self.monitor_overhead += mon_now * frac
        self.decision_overhead += dec_now * frac

        cache_util = self.cache.occupancy()
        resource_util = (m["cpu_util"] + m["mem_bw_util"] + cache_util) / 3.0
        phase_idx = self.workload.phase_index
        phase_kind = self.workload.current_phase.kind

        crossed = self.workload.retire(instr, t1)
        self.cycle = t1
        if crossed:
            new_phase = self.workload.current_phase
            self.cache.on_phase_change(self.workload.base_characteristics())
            self.phase_changes.append((t1, self.workload.phase_index, new_phase.label))
            self.log(t1, "phase", f"Workload changed to phase {self.workload.phase_index + 1}: "
                                  f"{new_phase.label}")

        done = self.controller.complete_if_due(t1)
        if done is not None:
            flushed = self.cache.resize(done.new.cache_kb)
            self.cache.policy = done.new.policy
            self.config = done.new
            self.decider.on_reconfig_complete()
            self.log(t1, "reconfig",
                     f"Reconfiguration #{done.id} complete in {done.duration:,.0f} cycles "
                     f"(lost {done.lost_cycles:,.0f} cycles, {done.energy_uj:.1f} uJ"
                     f"{f', flushed {flushed:.0f} KB dirty data' if flushed > 0 else ''}). "
                     f"Now: {self.config.describe()}")

        activity = "reconfiguring" if any_reconf else "normal"
        record = StepRecord(
            step=self.step_index, cycle_start=t0, cycle_end=t1,
            phase_index=phase_idx, phase_kind=phase_kind, instructions=instr, ipc=ipc,
            cpu_util=m["cpu_util"], mem_bw_util=m["mem_bw_util"], mem_bw_gbps=m["mem_bw_gbps"],
            hit_rate=im["hit_rate"], miss_rate=im["miss_rate"], mem_latency_ns=m["mem_latency_ns"],
            queue_util=m["queue_util"], resource_util=resource_util, cache_util=cache_util,
            mem_stall_frac=m["mem_stall_frac"], power_w=m["power_w"], energy_uj=energy,
            cores=self.config.cores, freq_ghz=self.config.freq_ghz, cache_kb=self.config.cache_kb,
            mem_channels=self.config.mem_channels, prefetch=self.config.prefetch,
            policy=self.config.policy, reconfiguring=any_reconf, reconfig_lost_cycles=lost,
            monitor_overhead_cycles=mon_now * frac, decision_overhead_cycles=dec_now * frac,
            classification=self.classifier.stable if self.adaptive else "-",
            activity=activity, resident_kb=self.cache.resident_kb,
        )
        if any_reconf:
            # show the transitional resources actually in use
            seg = next(s for s in segments if s.reconfiguring)
            record.cores, record.freq_ghz = seg.config.cores, seg.config.freq_ghz
            record.cache_kb, record.mem_channels = seg.config.cache_kb, seg.config.mem_channels
        self.history.append(record)
        self.step_index += 1

        if self.workload.finished:
            self.finished = True
            self.log(t1, "done", f"Workload complete: {self.workload.total_instructions / 1e6:,.2f} M "
                                 f"instructions in {t1:,.0f} cycles")
            return False

        if self.adaptive:
            self._adapt(m, im, cycles, instr, core_cycles * frac, any_reconf, record)
        if self.step_index >= K.MAX_STEPS:
            self.finished = True
            self.log(t1, "done", "Stopped: maximum step count reached")
        return not self.finished

    # ------------------------------------------------------------------
    def _adapt(self, m: dict, im: dict, cycles: float, instr: float, core_cycles: float,
               reconf: bool, record: StepRecord) -> None:
        counters = {
            "cpu_util": m["cpu_util"], "core_activity": m["core_activity"],
            "mem_bw_util": m["mem_bw_util"], "mem_latency_ns": m["mem_latency_ns"],
            "queue_util": m["queue_util"], "resource_util": record.resource_util,
            "mem_stall_frac": m["mem_stall_frac"],
            **{k: im[k] for k in ("hit_rate", "miss_rate", "exposed_miss_rate", "t_exec_ns",
                                  "t_mem_ns", "llc_api", "bytes_per_instr", "stream_fraction")},
        }
        freq = core_cycles / cycles if cycles > 0 else self.config.freq_ghz
        sample = self.monitor.observe(counters, cycles, instr, freq, reconf, self.cycle, self.config)
        if sample is None:
            return
        self._pending_monitor += self.monitor.overhead_cycles + K.CLASSIFY_OVERHEAD_CYCLES
        if record.activity == "normal":
            record.activity = "monitoring"
        if sample.index < self.params.warmup_samples:
            if sample.index == self.params.warmup_samples - 1:
                self.log(self.cycle, "monitor", f"Monitoring warm-up complete after "
                                                f"{self.params.warmup_samples} samples; classification enabled")
            return

        prev = self.classifier.stable
        cls = self.classifier.update(sample)
        if cls.changed:
            self.decider.on_class_change(prev, self.step_index)
            self.log(self.cycle, "classify", f"Workload classified as {cls.stable} ({cls.reason})")

        decision = self.decider.evaluate(sample, cls, self.config, self.step_index, self.controller,
                                         self.cache.resident_kb, self.cache.dirty_fraction)
        if decision is None:
            return
        self._pending_decision += self.decider.last_overhead_cycles
        if record.activity in ("normal", "monitoring"):
            record.activity = "deciding"
        if decision.action == "hold":
            self.log(self.cycle, "hold", f"Decision engine holds configuration: {decision.reason}")
            return
        rec = self.controller.begin(self.cycle, self.config, decision.target, self.cache.resident_kb,
                                    self.cache.dirty_fraction, decision.predicted_gain, decision.reason)
        self.log(self.cycle, "decision", f"Decision: reconfigure - {decision.reason}")
        for f in rec.changed:
            label, fmt = _FIELD_LABEL[f]
            old, new = getattr(rec.old, f), getattr(rec.new, f)
            if isinstance(old, bool):
                change = f"{label} {fmt(new)}"
            elif isinstance(old, (int, float)):
                change = f"{label} {'increased' if new > old else 'decreased'} from {fmt(old)} to {fmt(new)}"
            else:
                change = f"{label} changed from {fmt(old)} to {fmt(new)}"
            self.log(self.cycle, "reconfig", f"{change} (reconfiguration #{rec.id}, "
                                             f"est. {rec.duration:,.0f} cycles)")

    # ------------------------------------------------------------------
    def run(self, max_steps: Optional[int] = None) -> "SimulationEngine":
        n = 0
        while not self.finished and (max_steps is None or n < max_steps):
            self.step()
            n += 1
        return self

    # summary -----------------------------------------------------------
    def summary(self) -> dict:
        h = self.history
        total_cycles = self.cycle
        instr = sum(r.instructions for r in h)
        tw = lambda attr: (sum(getattr(r, attr) * (r.cycle_end - r.cycle_start) for r in h)
                           / max(total_cycles, 1e-9))
        reconf_lost = self.controller.total_lost_cycles()
        overhead = reconf_lost + self.monitor_overhead + self.decision_overhead
        accesses = sum(r.instructions for r in h) or 1.0
        hit = sum(r.hit_rate * r.instructions for r in h) / accesses
        return {
            "name": self.name,
            "finished": self.finished,
            "total_cycles": total_cycles,
            "instructions": instr,
            "avg_ipc": tw("ipc"),
            "throughput": instr / max(total_cycles, 1e-9),
            "avg_cpu_util": tw("cpu_util"),
            "avg_mem_bw_util": tw("mem_bw_util"),
            "avg_hit_rate": hit,
            "avg_mem_latency_ns": tw("mem_latency_ns"),
            "avg_resource_util": tw("resource_util"),
            "avg_power_w": tw("power_w"),
            "energy_uj": self.total_energy_uj + self.controller.total_energy(),
            "reconfig_count": len(self.controller.history),
            "reconfig_cycles": self.controller.total_duration(),
            "reconfig_lost_cycles": reconf_lost,
            "reconfig_energy_uj": self.controller.total_energy(),
            "monitor_overhead_cycles": self.monitor_overhead,
            "decision_overhead_cycles": self.decision_overhead,
            "adaptation_overhead_cycles": overhead,
            "adaptation_overhead_pct": overhead / max(total_cycles, 1e-9) * 100.0,
            "normal_execution_cycles": total_cycles - overhead,
            "decisions": len(self.decider.decisions),
            "holds": sum(1 for d in self.decider.decisions if d.action == "hold"),
        }
