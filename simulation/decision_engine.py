"""
Autonomous Decision Engine.

When it runs
    * the validated workload class changed (or a change is still pending
      because it happened during a reconfiguration), or
    * IPC drifted by more than DRIFT_REEVALUATE while the samples still agree
      with the current class (behaviour changed inside the same class), or
    * closed-loop refinement: REFINE_AFTER_SAMPLES clean samples after a
      reconfiguration it re-checks the choice once using counters measured on
      the *new* configuration (corrects prediction error; once per class).
  It never runs on samples polluted by an in-flight reconfiguration and never
  starts a new reconfiguration while one is in progress.

What it decides
    1. Policy: the class selects a constrained search space (see
       POLICY_SPACE): e.g. Memory-bound => 3-4 channels, >= 2 MB cache,
       prefetch on, frequency <= 2.5 GHz.
    2. Optimisation: every candidate within the power budget is scored with
       the counter-based PerformancePredictor.  Among candidates within
       EFFICIENCY_TOLERANCE (3 %) of the best predicted throughput, the one
       with the lowest peak power is the target (near-optimal performance at
       minimum power).  If the current configuration is itself in that set it
       is kept.
    3. Cost/benefit gate (cost-aware mode):
           gain      = X_target / X_current - 1
           H         = expected remaining residency of the phase (steps),
                       learnt online as an EMA of observed class durations
           saved     = H * step_cycles * (1 - X_current / X_target)
           cost      = duration * (1 - factor * X_transitional / X_target)
                       (lost cycles vs. an instantaneous switch)
       Reconfigure only if  gain >= min_gain  and  saved >= margin * cost.

Naive mode (cost_aware=False) reconfigures whenever the best candidate
differs from the current configuration - used to show why the gate matters.
"""

from __future__ import annotations

from typing import Optional

from config import default_config as K
from simulation.models import ArchConfig, Classification, Decision, Sample, SimulationParams
from simulation.performance_predictor import PerformancePredictor
from simulation.reconfiguration_controller import ReconfigurationController
from simulation.workload_classifier import UNKNOWN


def policy_candidates(workload_class: str, power_budget_w: float,
                      space: dict | None = None) -> list[ArchConfig]:
    from simulation.power_model import peak_power
    sp = (space or K.POLICY_SPACE)[workload_class]
    out = []
    for n in sp["cores"]:
        for f in sp["freq"]:
            for c in sp["cache"]:
                for ch in sp["channels"]:
                    for pf in sp["prefetch"]:
                        for pol in sp["policy"]:
                            cfg = ArchConfig(n, f, c, ch, pf, pol)
                            if peak_power(cfg) <= power_budget_w:
                                out.append(cfg)
    return out


class DecisionEngine:
    def __init__(self, params: SimulationParams, predictor: PerformancePredictor | None = None):
        self.p = params
        self.predictor = predictor or PerformancePredictor()
        self.decisions: list[Decision] = []
        self.residency_ema = float(params.expected_phase_steps)
        self._class_start_step = 0
        self._pending = False
        self._baseline_ipc: Optional[float] = None
        self._verify_in: Optional[int] = None      # clean samples until refinement check
        self._refined = False                      # one refinement per class period
        self.last_overhead_cycles = 0.0

    # ------------------------------------------------------------------
    def on_class_change(self, old_class: str, step: int) -> None:
        if old_class != UNKNOWN:
            observed = step - self._class_start_step
            a = K.RESIDENCY_EMA_ALPHA
            self.residency_ema = (1 - a) * self.residency_ema + a * observed
        self._class_start_step = step
        self._refined = False

    def on_reconfig_complete(self) -> None:
        self._baseline_ipc = None   # re-baseline on the next clean sample
        if not self._refined:
            self._verify_in = K.REFINE_AFTER_SAMPLES

    def horizon_steps(self, step: int) -> float:
        # Memoryless phase model: expected remaining residency = mean residency.
        return max(self.residency_ema, float(self.p.sampling_interval))

    # ------------------------------------------------------------------
    def evaluate(self, sample: Sample, cls: Classification, current: ArchConfig, step: int,
                 controller: ReconfigurationController, resident_kb: float,
                 dirty_fraction: float) -> Optional[Decision]:
        """Return a Decision when the engine ran, None when it stayed idle."""
        self.last_overhead_cycles = 0.0
        if cls.changed:
            self._pending = True
        if sample.reconfiguring or controller.busy or cls.stable == UNKNOWN:
            return None

        trigger = None
        if self._verify_in is not None:
            self._verify_in -= 1
        if self._pending:
            trigger = "class-change"
        elif self._verify_in is not None and self._verify_in <= 0 and cls.raw == cls.stable:
            # closed-loop refinement with counters measured on the new configuration
            trigger = "refinement"
            self._refined = True

        elif self._baseline_ipc is None:
            self._baseline_ipc = sample.ipc
        elif (cls.raw == cls.stable
              and abs(sample.ipc / self._baseline_ipc - 1.0) > K.DRIFT_REEVALUATE):
            # drift only counts when the sample agrees with the current class;
            # otherwise a class change is in progress and hysteresis decides.
            trigger = "drift"
        if trigger is None:
            return None
        self._pending = False
        self._verify_in = None
        self._baseline_ipc = sample.ipc

        candidates = policy_candidates(cls.stable, self.p.power_budget_w)
        self.last_overhead_cycles = (K.DECISION_BASE_CYCLES
                                     + K.DECISION_CYCLES_PER_CANDIDATE * (len(candidates) + 1))
        cur_pred = self.predictor.predict(sample, current)
        horizon = self.horizon_steps(step)

        def mk(action, reason, target=None, gain=0.0, saved=0.0, cost=0.0):
            d = Decision(cycle=sample.cycle, workload_class=cls.stable, action=action, reason=reason,
                         current=current, target=target, predicted_gain=gain,
                         expected_saved_cycles=saved, expected_cost_cycles=cost,
                         candidates_evaluated=len(candidates), horizon_steps=horizon)
            self.decisions.append(d)
            return d

        if not candidates:
            return mk("hold", "no candidate configuration fits the power budget")

        preds = [self.predictor.predict(sample, c) for c in candidates]
        top = max(pr.throughput for pr in preds)
        # Energy-aware tie-break: near-optimal performance at the lowest power.
        near = [pr for pr in preds if pr.throughput >= (1.0 - K.EFFICIENCY_TOLERANCE) * top]
        best = min(near, key=lambda pr: (pr.peak_power_w, -pr.throughput))
        # Prefer staying put if the current config is itself near-optimal.
        if any(pr.config == current for pr in near) and current in [c for c in candidates]:
            best = next(pr for pr in near if pr.config == current)
        gain = best.throughput / cur_pred.throughput - 1.0
        if best.config == current:
            return mk("hold", f"current configuration is already optimal for {cls.stable}",
                      target=current, gain=0.0)

        cost = controller.estimate(current, best.config, resident_kb, dirty_fraction)
        # Expected lost cycles vs an instantaneous switch: during the transition the
        # machine runs on the transitional resources at the reduced factor.
        trans = self.predictor.predict(sample, ArchConfig.transitional(current, best.config))
        cost.lost_cycles = cost.duration * max(0.0, 1.0 - cost.factor * trans.throughput / best.throughput)
        saved = horizon * self.p.step_cycles * (1.0 - cur_pred.throughput / best.throughput) \
            if best.throughput > 0 else 0.0

        if not self.p.cost_aware:
            if gain <= 0:
                return mk("hold", "naive policy: no predicted gain", best.config, gain, saved, cost.lost_cycles)
            return mk("reconfigure", f"naive policy: chase best config for {cls.stable}",
                      best.config, gain, saved, cost.lost_cycles)

        if gain < self.p.min_gain:
            return mk("hold", f"predicted gain {gain:+.1%} below {self.p.min_gain:.0%} threshold",
                      best.config, gain, saved, cost.lost_cycles)
        if saved < self.p.benefit_margin * cost.lost_cycles:
            return mk("hold", (f"benefit {saved:,.0f} cycles over ~{horizon:.0f}-step horizon does not "
                               f"justify cost {cost.lost_cycles:,.0f} cycles x{self.p.benefit_margin:g}"),
                      best.config, gain, saved, cost.lost_cycles)
        return mk("reconfigure",
                  (f"{cls.stable} ({trigger}): predicted {gain:+.1%}, saves ~{saved:,.0f} cycles "
                   f"vs cost {cost.lost_cycles:,.0f}"),
                  best.config, gain, saved, cost.lost_cycles)
