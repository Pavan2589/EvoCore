"""Plain data types shared by all simulation components."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Optional


@dataclass(frozen=True)
class ArchConfig:
    """A complete architectural configuration (the reconfigurable state)."""

    cores: int = 4
    freq_ghz: float = 3.0
    cache_kb: int = 1024
    mem_channels: int = 2
    prefetch: bool = False
    policy: str = "LRU"

    def with_(self, **kw) -> "ArchConfig":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return (f"{self.cores}C @ {self.freq_ghz:.1f} GHz | "
                f"{fmt_cache(self.cache_kb)} {self.policy} | "
                f"{self.mem_channels} ch | PF {'on' if self.prefetch else 'off'}")

    def diff(self, other: "ArchConfig") -> list[str]:
        """Names of fields that differ between two configurations."""
        return [k for k, v in self.as_dict().items() if getattr(other, k) != v]

    @staticmethod
    def transitional(a: "ArchConfig", b: "ArchConfig") -> "ArchConfig":
        """Conservative configuration in effect while moving from a to b.

        While resources are being switched only the resources common to both
        configurations are usable (e.g. a core being woken up is not yet
        available, a core being parked is already draining).
        """
        return ArchConfig(
            cores=min(a.cores, b.cores),
            freq_ghz=min(a.freq_ghz, b.freq_ghz),
            cache_kb=min(a.cache_kb, b.cache_kb),
            mem_channels=min(a.mem_channels, b.mem_channels),
            prefetch=a.prefetch and b.prefetch,
            policy=a.policy,
        )


def fmt_cache(kb: float) -> str:
    return f"{kb / 1024:g} MB" if kb >= 1024 else f"{kb:g} KB"


@dataclass(frozen=True)
class WorkloadCharacteristics:
    """Ground-truth behaviour of one workload phase.

    Only the workload generator and the performance model see these values;
    the monitoring/decision side must infer behaviour from counters.
    """

    kind: str
    label: str
    mem_ratio: float
    l1_hit_rate: float
    working_set_kb: float
    cpi_exec: float
    parallel_fraction: float
    mlp: float
    prefetchability: float
    temporal_locality: float
    base_miss_rate: float
    min_miss_rate: float
    write_fraction: float
    ws_overlap: float

    @property
    def llc_api(self) -> float:
        """LLC accesses per instruction (memory refs that miss the private L1)."""
        return self.mem_ratio * (1.0 - self.l1_hit_rate)


@dataclass
class PerfResult:
    """Output of the performance model for one interval under one config."""

    throughput: float           # instructions per ns (all cores)
    ipc: float                  # instructions per core-clock cycle (system)
    ipc_per_core: float
    cpu_util: float             # fraction of allocated core time actively executing
    core_activity: float        # S(n)/n: fraction of cores with runnable work
    mem_stall_frac: float       # fraction of thread time stalled on DRAM
    t_exec_ns: float            # per-thread ns per instruction, execution part
    t_mem_ns: float             # per-thread ns per instruction, DRAM stall part
    llc_api: float
    demand_miss_rate: float     # LLC miss rate before prefetch coverage
    exposed_miss_rate: float    # misses still visible after prefetching
    hit_rate: float             # LLC hit rate incl. prefetch hits
    prefetch_coverage: float
    stream_fraction: float      # observable share of sequential misses
    bytes_per_instr: float
    mem_bw_gbps: float
    mem_bw_util: float
    mem_latency_ns: float
    queue_util: float
    llc_hit_cycles: float
    bandwidth_limited: bool
    power_w: float = 0.0


@dataclass
class StepRecord:
    """Everything recorded for one engine step (one row of the history)."""

    step: int
    cycle_start: float
    cycle_end: float
    phase_index: int
    phase_kind: str
    instructions: float
    ipc: float
    cpu_util: float
    mem_bw_util: float
    mem_bw_gbps: float
    hit_rate: float
    miss_rate: float
    mem_latency_ns: float
    queue_util: float
    resource_util: float
    cache_util: float
    mem_stall_frac: float
    power_w: float
    energy_uj: float
    cores: int
    freq_ghz: float
    cache_kb: int
    mem_channels: int
    prefetch: bool
    policy: str
    reconfiguring: bool
    reconfig_lost_cycles: float
    monitor_overhead_cycles: float
    decision_overhead_cycles: float
    classification: str
    activity: str               # normal | monitoring | deciding | reconfiguring
    resident_kb: float


@dataclass
class Sample:
    """Aggregated counters for one monitoring interval."""

    index: int
    cycle: float
    steps: int
    instructions: float
    throughput: float
    ipc: float
    cpu_util: float
    core_activity: float
    mem_bw_util: float
    hit_rate: float
    miss_rate: float            # demand LLC miss rate
    exposed_miss_rate: float
    mem_latency_ns: float
    queue_util: float
    resource_util: float
    mem_stall_frac: float
    t_exec_ns: float
    t_mem_ns: float
    llc_api: float
    bytes_per_instr: float
    stream_fraction: float
    reconfiguring: bool
    config: ArchConfig


@dataclass
class Classification:
    raw: str                    # class suggested by this sample alone
    stable: str                 # validated class (after hysteresis)
    changed: bool               # stable class changed with this sample
    streak: int                 # consecutive samples agreeing with raw
    reason: str


@dataclass
class Decision:
    cycle: float
    workload_class: str
    action: str                 # "reconfigure" | "hold"
    reason: str
    current: ArchConfig
    target: Optional[ArchConfig]
    predicted_gain: float       # fractional throughput improvement
    expected_saved_cycles: float
    expected_cost_cycles: float
    candidates_evaluated: int
    horizon_steps: float


@dataclass
class ReconfigRecord:
    id: int
    start_cycle: float
    end_cycle: float
    old: ArchConfig
    new: ArchConfig
    changed: list
    duration: float
    throughput_factor: float
    energy_uj: float
    predicted_gain: float
    reason: str
    lost_cycles: float = 0.0     # measured work-equivalent cycles lost
    completed: bool = False


@dataclass
class Event:
    cycle: float
    category: str               # init|phase|classify|decision|reconfig|hold|done
    message: str


@dataclass
class SimulationParams:
    """User-facing knobs of one simulation run."""

    step_cycles: int = 10_000
    sampling_interval: int = 5
    confirm_samples: int = 2
    warmup_samples: int = 2
    min_gain: float = 0.05
    benefit_margin: float = 1.25
    expected_phase_steps: float = 50
    power_budget_w: float = 45.0
    cost_aware: bool = True           # False = naive "always chase the class" policy
    hysteresis: bool = True
    noise: float = 0.0                # relative std-dev of workload parameter noise
    seed: int = 7
    initial_config: ArchConfig = field(default_factory=ArchConfig)
