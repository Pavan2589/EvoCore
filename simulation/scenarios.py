"""
Predefined simulation scenarios.

Phase sizes are given in instructions.  They were chosen so that on the
default static architecture each long phase takes roughly 100 engine steps
(about 1 M reference cycles), mirroring the "0-100 / 100-200 / 200-300"
example of the project brief - but because phases are defined by *work*,
the adaptive architecture finishes them sooner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simulation.workload_generator import PhaseSpec

M = 1e6

# Work that takes ~100 steps on the static baseline, per phase kind.
STEPS_100 = {"compute": 22.0 * M, "memory": 2.4 * M, "mixed": 8.0 * M}


def phase(kind: str, steps_on_baseline: float) -> PhaseSpec:
    return PhaseSpec(kind, STEPS_100[kind] * steps_on_baseline / 100.0)


@dataclass
class Scenario:
    key: str
    title: str
    purpose: str
    phases: list[PhaseSpec]
    param_overrides: dict = field(default_factory=dict)


def _rapid(pairs: int = 12, steps: float = 6) -> list[PhaseSpec]:
    out = []
    for _ in range(pairs):
        out += [phase("compute", steps), phase("memory", steps)]
    return out


SCENARIOS: dict[str, Scenario] = {
    s.key: s for s in [
        Scenario("static_compute", "1. Static compute-intensive",
                 "Workload stays compute-bound for its whole run. Establishes the baseline: "
                 "the architecture should adapt once and then stay put.",
                 [phase("compute", 300)]),
        Scenario("static_memory", "2. Static memory-intensive",
                 "Workload stays memory-bound. Evaluates the memory-side optimisations "
                 "(channels, LLC capacity, prefetching, frequency reduction).",
                 [phase("memory", 300)]),
        Scenario("dynamic", "3. Dynamic workload transition",
                 "Main demonstration: compute -> memory -> mixed. The architecture must detect "
                 "each transition and reconfigure.",
                 [phase("compute", 100), phase("memory", 100), phase("mixed", 100)]),
        Scenario("adaptive_vs_static", "4. Adaptive vs static (long, recurring phases)",
                 "Same workload on both architectures; phases recur so the adaptive system has to "
                 "re-adapt to behaviour it has seen before.",
                 [phase("compute", 100), phase("memory", 100), phase("mixed", 100),
                  phase("compute", 100), phase("memory", 100)]),
        Scenario("rapid_switching", "5. Rapid phase switching (adaptation not worth it)",
                 "Compute and memory phases alternate every ~6 baseline steps - faster than the "
                 "monitor can confirm a class. A cost-aware engine should mostly hold; compare "
                 "against the naive engine on the Overhead tab.",
                 _rapid()),
        Scenario("short", "6. Very short workload",
                 "The program ends after ~15 steps; there is barely time to monitor, let alone "
                 "amortise a reconfiguration.",
                 [phase("compute", 8), phase("memory", 7)]),
    ]
}

CUSTOM_KEY = "custom"
