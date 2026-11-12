"""
Workload Generator.

A workload is a sequence of phases.  Each phase has a *kind* (compute,
memory, mixed) and an amount of work expressed in instructions.  Phase
boundaries are defined by work (instructions retired), not by time, so a
faster architecture genuinely finishes each phase sooner; this is what makes
static-vs-adaptive comparisons meaningful.

Optional noise perturbs the phase characteristics step by step with a seeded
RNG, so two architectures given the same seed see an identical workload.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from config.default_config import WORKLOAD_PROFILES
from simulation.models import WorkloadCharacteristics

PHASE_KINDS = tuple(WORKLOAD_PROFILES.keys())


def make_characteristics(kind: str, **overrides) -> WorkloadCharacteristics:
    if kind not in WORKLOAD_PROFILES:
        raise ValueError(f"Unknown workload kind {kind!r}; expected one of {PHASE_KINDS}")
    p = dict(WORKLOAD_PROFILES[kind])
    p.update(overrides)
    return WorkloadCharacteristics(kind=kind, **p)


@dataclass(frozen=True)
class PhaseSpec:
    kind: str
    instructions: float            # work in the phase (instructions)
    name: Optional[str] = None

    @property
    def label(self) -> str:
        return self.name or WORKLOAD_PROFILES[self.kind]["label"]


class WorkloadGenerator:
    """Produces the workload characteristics for each simulation step."""

    # Parameters that noise is applied to (all stay in physical ranges).
    _NOISY = ("mem_ratio", "cpi_exec", "working_set_kb", "prefetchability")

    def __init__(self, phases: list[PhaseSpec], noise: float = 0.0, seed: int = 7):
        if not phases:
            raise ValueError("A workload needs at least one phase")
        for ph in phases:
            if ph.instructions <= 0:
                raise ValueError("Phase instruction counts must be positive")
        self.phases = list(phases)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)
        self._base = [make_characteristics(p.kind) for p in self.phases]
        self.boundaries = np.cumsum([p.instructions for p in self.phases])
        self.total_instructions = float(self.boundaries[-1])
        self.retired = 0.0
        self.phase_index = 0
        self.phase_start_cycles: list[float] = [0.0]

    # ------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.retired >= self.total_instructions - 1e-6

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_instructions - self.retired)

    @property
    def current_phase(self) -> PhaseSpec:
        return self.phases[self.phase_index]

    def phase_progress(self) -> float:
        start = 0.0 if self.phase_index == 0 else self.boundaries[self.phase_index - 1]
        size = self.phases[self.phase_index].instructions
        return min(1.0, (self.retired - start) / size)

    def characteristics(self) -> WorkloadCharacteristics:
        """Characteristics for the next step (noise applied if enabled)."""
        base = self._base[self.phase_index]
        if self.noise <= 0:
            return base
        changes = {}
        for name in self._NOISY:
            v = getattr(base, name) * (1.0 + self._rng.normal(0.0, self.noise))
            if name in ("mem_ratio", "prefetchability"):
                v = float(np.clip(v, 0.01, 0.99))
            else:
                v = max(v, 1e-3)
            changes[name] = v
        return replace(base, **changes)

    def base_characteristics(self) -> WorkloadCharacteristics:
        return self._base[self.phase_index]

    def instructions_left_in_phase(self) -> float:
        return float(self.boundaries[self.phase_index] - self.retired)

    def retire(self, instructions: float, cycle_now: float) -> bool:
        """Account retired work.  Returns True if a phase boundary was crossed."""
        self.retired = min(self.total_instructions, self.retired + instructions)
        crossed = False
        while (self.phase_index < len(self.phases) - 1
               and self.retired >= self.boundaries[self.phase_index] - 1e-6):
            self.phase_index += 1
            self.phase_start_cycles.append(cycle_now)
            crossed = True
        return crossed
