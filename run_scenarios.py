"""
Headless runner: executes every predefined scenario on the static, cost-aware
adaptive and naive adaptive architectures and prints a comparison table.

    python run_scenarios.py            # summary table
    python run_scenarios.py --events   # also print the adaptive event log per scenario
    python run_scenarios.py --sweeps   # also run the overhead sweeps
"""

from __future__ import annotations

import argparse

from simulation.analysis import phase_length_sweep, run_comparison, short_workload_sweep
from simulation.models import ArchConfig, SimulationParams
from simulation.scenarios import SCENARIOS
from config.default_config import DEFAULT_STATIC_CONFIG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--sweeps", action="store_true")
    args = ap.parse_args()

    params = SimulationParams(initial_config=ArchConfig(**DEFAULT_STATIC_CONFIG))
    hdr = (f"{'Scenario':44s} {'Static cyc':>11s} {'Adapt cyc':>11s} {'Impr %':>7s} {'IPC s/a':>11s} "
           f"{'Reconf':>6s} {'Ovh %':>6s} {'Energy %':>8s} | {'Naive impr %':>12s} {'Naive rec':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for key, sc in SCENARIOS.items():
        r = run_comparison(sc.phases, params)
        n = run_comparison(sc.phases, params, naive=True)
        s, a = r["static"], r["adaptive"]
        print(f"{sc.title:44s} {s['total_cycles']:>11,.0f} {a['total_cycles']:>11,.0f} "
              f"{r['improvement_pct']:>+7.1f} {s['avg_ipc']:>5.2f}/{a['avg_ipc']:<5.2f} "
              f"{a['reconfig_count']:>6d} {a['adaptation_overhead_pct']:>6.2f} {r['energy_change_pct']:>+8.1f} | "
              f"{n['improvement_pct']:>+12.1f} {n['adaptive']['reconfig_count']:>9d}")
        if args.events:
            for e in r["engines"][1].events:
                print(f"    {e.cycle:>12,.0f}  [{e.category}] {e.message}")
            print()

    if args.sweeps:
        print("\nPhase-length sweep (alternating compute/memory, fixed total work)")
        rows = phase_length_sweep(params)
        for i in range(0, len(rows), 2):
            c, nv = rows[i], rows[i + 1]
            print(f"  {c['phases']:3d} phases of {c['phase_length_steps']:6.1f} steps: "
                  f"cost-aware {c['improvement_pct']:+6.1f}% ({c['reconfigurations']} reconf)   "
                  f"naive {nv['improvement_pct']:+6.1f}% ({nv['reconfigurations']} reconf)")
        print("\nShort-workload sweep (single compute phase)")
        for row in short_workload_sweep(params):
            print(f"  {row['length_steps']:4d} steps: {row['improvement_pct']:+6.1f}% "
                  f"({row['reconfigurations']} reconf, overhead {row['overhead_pct']:.1f}%)")


if __name__ == "__main__":
    main()
