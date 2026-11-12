"""
Performance model: couples CPU, cache and memory into one consistent result.

The three subsystems interact through a feedback loop:

    throughput X  ->  DRAM bandwidth demand D = X * bytes_per_instr
    D             ->  bus utilisation rho and loaded latency L(rho)
    L             ->  per-thread stall time t_mem
    t_mem         ->  throughput X = S(n) / (t_exec + t_mem)

The model finds the self-consistent operating point X* of this loop:

    X* = S(n) / (t_exec + A_llc * m_exp * L(X* * b / B_peak) / MLP)

The right-hand side decreases monotonically in X, so a unique fixed point
exists and is found by bisection.  If even the bandwidth ceiling cannot be
reached the workload is *bandwidth-bound* and X = 0.97 * B_peak / b.

Derived metrics
    IPC (system)        = X / f                    instructions per core cycle
    IPC per core        = X / (f * n)
    CPU utilisation     = (S/n) * t_exec / (t_exec + t_mem)
    Memory-stall frac.  = t_mem / (t_exec + t_mem)
"""

from __future__ import annotations

from config import default_config as K
from simulation import cache as cache_mod
from simulation.cpu import CPU, llc_hit_cycles
from simulation.memory_controller import MemoryController
from simulation.models import ArchConfig, PerfResult, WorkloadCharacteristics
from simulation.power_model import total_power


def solve_operating_point(speedup: float, t_exec: float, stall_coeff: float,
                          bytes_per_instr: float, peak_bw: float) -> tuple[float, float, float, bool]:
    """Return (throughput, rho, latency_ns, bandwidth_limited).

    stall_coeff = A_llc * m_exp / MLP so that t_mem = stall_coeff * L.
    """
    def rhs(x: float) -> float:
        rho = x * bytes_per_instr / peak_bw
        lat = MemoryController.latency_ns(rho)
        return speedup / (t_exec + stall_coeff * lat)

    if bytes_per_instr <= 0:
        x = speedup / (t_exec + stall_coeff * K.DRAM_UNLOADED_LATENCY_NS)
        return x, 0.0, K.DRAM_UNLOADED_LATENCY_NS, False

    x_max = K.MAX_BUS_UTILIZATION * peak_bw / bytes_per_instr
    if rhs(x_max) >= x_max:
        return x_max, K.MAX_BUS_UTILIZATION, MemoryController.latency_ns(K.MAX_BUS_UTILIZATION), True
    lo, hi = 0.0, x_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rhs(mid) > mid:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    rho = x * bytes_per_instr / peak_bw
    return x, rho, MemoryController.latency_ns(rho), False


class PerformanceModel:
    """Evaluates one configuration against one workload phase and cache state."""

    def evaluate(self, cfg: ArchConfig, w: WorkloadCharacteristics,
                 cache: cache_mod.Cache) -> PerfResult:
        cpu = CPU(cfg.cores, cfg.freq_ghz)
        mc = MemoryController(cfg.mem_channels)
        cb = cache.behaviour(w, cfg.prefetch, capacity_kb=cfg.cache_kb)

        s = cpu.speedup(w)
        t_exec = cpu.exec_time_per_instr(w, cb.demand_miss, cfg.cache_kb)
        stall_coeff = w.llc_api * cb.exposed_miss / w.mlp
        bpi = w.llc_api * cb.traffic_lines_per_access * K.CACHE_LINE_BYTES

        x, rho, lat, bw_limited = solve_operating_point(s, t_exec, stall_coeff, bpi, mc.peak_bandwidth)
        t_mem = stall_coeff * lat
        if bw_limited:
            # The bus, not latency, limits progress: the extra time is stall time.
            t_total = s / x
            t_mem = t_total - t_exec
        t_total = t_exec + t_mem
        activity = s / cfg.cores
        cpu_util = activity * t_exec / t_total
        demand = x * bpi
        res = PerfResult(
            throughput=x,
            ipc=x / cfg.freq_ghz,
            ipc_per_core=x / (cfg.freq_ghz * cfg.cores),
            cpu_util=cpu_util,
            core_activity=activity,
            mem_stall_frac=t_mem / t_total,
            t_exec_ns=t_exec,
            t_mem_ns=t_mem,
            llc_api=w.llc_api,
            demand_miss_rate=cb.demand_miss,
            exposed_miss_rate=cb.exposed_miss,
            hit_rate=cb.hit_rate,
            prefetch_coverage=cb.prefetch_coverage,
            stream_fraction=w.prefetchability,
            bytes_per_instr=bpi,
            mem_bw_gbps=demand,
            mem_bw_util=rho,
            mem_latency_ns=lat,
            queue_util=mc.queue_utilization(demand, lat),
            llc_hit_cycles=llc_hit_cycles(cfg.cache_kb),
            bandwidth_limited=bw_limited,
        )
        res.power_w = total_power(cfg, cpu_util, rho)
        return res
