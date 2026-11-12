"""
Static-vs-adaptive comparison and adaptation-overhead analysis.

    Performance improvement (%) = (T_static - T_adaptive) / T_static * 100
    Total effective time         = normal execution time + adaptation overhead
    Adaptation overhead          = reconfiguration lost cycles
                                   + monitoring cycles + decision cycles
    Gross benefit                = net benefit + adaptation overhead
                                   (what adaptation would have saved if free)
"""

from __future__ import annotations

from dataclasses import replace

from simulation.baseline_architecture import (create_adaptive_engine, create_static_engine,
                                              naive_params)
from simulation.engine import SimulationEngine
from simulation.models import ArchConfig, SimulationParams
from simulation.workload_generator import PhaseSpec


def improvement_pct(static_cycles: float, adaptive_cycles: float) -> float:
    return (static_cycles - adaptive_cycles) / static_cycles * 100.0 if static_cycles > 0 else 0.0


def compare(static: SimulationEngine, adaptive: SimulationEngine) -> dict:
    s, a = static.summary(), adaptive.summary()
    if abs(s["instructions"] - a["instructions"]) > 1e-3 * max(s["instructions"], 1.0):
        raise ValueError("static and adaptive runs did not execute the same amount of work")
    net = s["total_cycles"] - a["total_cycles"]
    return {
        "static": s,
        "adaptive": a,
        "improvement_pct": improvement_pct(s["total_cycles"], a["total_cycles"]),
        "net_benefit_cycles": net,
        "gross_benefit_cycles": net + a["adaptation_overhead_cycles"],
        "speedup": s["total_cycles"] / a["total_cycles"] if a["total_cycles"] else 0.0,
        "ipc_improvement_pct": (a["avg_ipc"] / s["avg_ipc"] - 1.0) * 100.0 if s["avg_ipc"] else 0.0,
        "resource_util_improvement_pct": (a["avg_resource_util"] - s["avg_resource_util"]) * 100.0,
        "energy_change_pct": (a["energy_uj"] / s["energy_uj"] - 1.0) * 100.0 if s["energy_uj"] else 0.0,
        "edp_change_pct": ((a["energy_uj"] * a["total_cycles"]) / (s["energy_uj"] * s["total_cycles"]) - 1.0)
        * 100.0 if s["energy_uj"] else 0.0,
    }


def run_comparison(phases: list[PhaseSpec], params: SimulationParams,
                   static_config: ArchConfig | None = None, naive: bool = False) -> dict:
    p = naive_params(params) if naive else params
    adaptive = create_adaptive_engine(phases, p, "Naive adaptive" if naive else "Adaptive").run()
    static = create_static_engine(phases, params, static_config).run()
    out = compare(static, adaptive)
    out["engines"] = (static, adaptive)
    return out


def phase_length_sweep(params: SimulationParams, total_baseline_steps: float = 240,
                       segment_counts=(2, 4, 8, 12, 16, 24, 40, 60, 80)) -> list[dict]:
    """Alternating compute/memory workload of fixed total work, cut into more and
    more (shorter) phases.  Shows where adaptation stops paying off."""
    from simulation.scenarios import phase
    rows = []
    for n in segment_counts:
        steps = total_baseline_steps / n
        phases = [phase("compute" if i % 2 == 0 else "memory", steps) for i in range(n)]
        for label, naive in (("Cost-aware adaptive", False), ("Naive adaptive", True)):
            r = run_comparison(phases, params, naive=naive)
            a = r["adaptive"]
            rows.append({
                "phases": n,
                "phase_length_steps": steps,
                "policy": label,
                "improvement_pct": r["improvement_pct"],
                "reconfigurations": a["reconfig_count"],
                "overhead_pct": a["adaptation_overhead_pct"],
            })
    return rows


def short_workload_sweep(params: SimulationParams,
                         lengths=(4, 8, 12, 20, 30, 50, 80, 120, 200)) -> list[dict]:
    """Single compute phase of increasing length (in baseline steps)."""
    from simulation.scenarios import phase
    rows = []
    for L in lengths:
        r = run_comparison([phase("compute", L)], params)
        rows.append({"length_steps": L, "improvement_pct": r["improvement_pct"],
                     "reconfigurations": r["adaptive"]["reconfig_count"],
                     "overhead_pct": r["adaptive"]["adaptation_overhead_pct"]})
    return rows
