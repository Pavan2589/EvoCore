"""
Counter-based performance predictor used by the Decision Engine.

The decision engine must NOT see the ground-truth workload.  This predictor
estimates the throughput of a candidate configuration using only quantities a
real PMU exposes (CPI stack, LLC miss rate, accesses/instruction, bytes per
instruction, busy-core ratio, stream-detector hit ratio) plus known hardware
constants (DRAM latency curve, channel bandwidth, LLC hit latency).

Estimation steps for a candidate configuration c1 from a sample at c0:

    p      = Amdahl-inverse(busy-core ratio)            parallel fraction
    MLP    = A_llc * m_exp0 * L0 / t_mem0              memory-level parallelism
    m1     = m0 * (C1 / C0)^-0.5                       sqrt(2) miss-curve rule
    c1     = 0.8 * stream_fraction  (if prefetch on)    prefetch coverage
    t_exec1= (core_cycles0 + A_llc (1-m1) H(C1) e) / f1
    X1     = fixed point of  S(n1) / (t_exec1 + A_llc m_exp1 L(rho) / MLP)

The same solver as the simulator is used for the memory feedback loop (it is
a property of the hardware, not the workload), but the workload parameters
are *estimates*, so predictions are approximate - exactly like real systems.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import default_config as K
from simulation.cache import prefetch_accuracy
from simulation.cpu import amdahl_speedup, llc_hit_cycles, parallel_fraction_from_activity
from simulation.models import ArchConfig, Sample
from simulation.performance_model import solve_operating_point
from simulation.power_model import peak_power


@dataclass
class Prediction:
    config: ArchConfig
    throughput: float
    mem_bw_util: float
    peak_power_w: float


class PerformancePredictor:
    def predict(self, s: Sample, cand: ArchConfig) -> Prediction:
        cur = s.config
        api = max(s.llc_api, 1e-9)
        sf = s.stream_fraction

        p = parallel_fraction_from_activity(cur.cores, s.core_activity)
        speedup = amdahl_speedup(cand.cores, p)

        # Prefetch coverage/accuracy in the sampled and candidate configs.
        acc = prefetch_accuracy(sf)
        c0 = K.PREFETCH_MAX_COVERAGE * sf if cur.prefetch else 0.0
        c1 = K.PREFETCH_MAX_COVERAGE * sf if cand.prefetch else 0.0

        # Miss-rate scaling with capacity (sqrt(2) rule), bounded.
        m0 = max(s.miss_rate, 1e-6)
        m1 = m0 * (cand.cache_kb / cur.cache_kb) ** (-K.MISS_CURVE_EXPONENT)
        if cand.cache_kb > cur.cache_kb:
            m1 = max(m1, min(m0, 0.03), m0 * 0.25)   # cannot beat the compulsory floor
        m1 = min(m1, 1.0)
        exp0 = m0 * (1.0 - c0)
        exp1 = m1 * (1.0 - c1)

        # Memory-level parallelism inferred from the stall component.
        if s.t_mem_ns > 1e-6 and exp0 > 1e-6:
            mlp = api * exp0 * s.mem_latency_ns / s.t_mem_ns
        else:
            mlp = 4.0
        mlp = max(mlp, 0.5)

        # Execution component: strip and re-add the LLC-hit part, rescale frequency.
        hit0 = api * (1.0 - m0) * llc_hit_cycles(cur.cache_kb) * K.LLC_HIT_EXPOSURE
        core_cycles = max(s.t_exec_ns * cur.freq_ghz - hit0, 0.05)
        hit1 = api * (1.0 - m1) * llc_hit_cycles(cand.cache_kb) * K.LLC_HIT_EXPOSURE
        t_exec1 = (core_cycles + hit1) / cand.freq_ghz

        # DRAM traffic per instruction, scaled from the measured value.
        traffic0 = m0 * (1.0 + c0 * (1.0 / acc - 1.0))
        traffic1 = m1 * (1.0 + c1 * (1.0 / acc - 1.0))
        bpi1 = s.bytes_per_instr * traffic1 / traffic0 if traffic0 > 0 else 0.0

        peak_bw = cand.mem_channels * K.CHANNEL_BANDWIDTH_GBPS
        x, rho, _, _ = solve_operating_point(speedup, t_exec1, api * exp1 / mlp, bpi1, peak_bw)
        return Prediction(config=cand, throughput=x, mem_bw_util=rho, peak_power_w=peak_power(cand))
