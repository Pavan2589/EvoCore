"""
Conventional static architecture (baseline).

It is the *same* simulator with the adaptive path disabled: no runtime
monitoring overhead, no classification, no decisions, no reconfiguration.
Running it through the same engine guarantees both systems execute an
identical workload (same phases, same noise sequence via the shared seed)
and are measured with identical equations.
"""

from __future__ import annotations

from dataclasses import replace

from simulation.engine import SimulationEngine
from simulation.models import ArchConfig, SimulationParams
from simulation.workload_generator import PhaseSpec


def create_static_engine(phases: list[PhaseSpec], params: SimulationParams,
                         static_config: ArchConfig | None = None) -> SimulationEngine:
    p = replace(params, initial_config=static_config or params.initial_config)
    return SimulationEngine(phases, p, adaptive=False, name="Static")


def create_adaptive_engine(phases: list[PhaseSpec], params: SimulationParams,
                           name: str = "Adaptive") -> SimulationEngine:
    return SimulationEngine(phases, params, adaptive=True, name=name)


def naive_params(params: SimulationParams) -> SimulationParams:
    """Parameters of a 'blind' adaptive system: samples every step, accepts the
    first sample of a new class, no threshold hysteresis and no cost/benefit
    check - it chases every apparent change in behaviour."""
    return replace(params, cost_aware=False, hysteresis=False, confirm_samples=1,
                   sampling_interval=1, warmup_samples=max(1, params.warmup_samples * params.sampling_interval))
