"""
Default configuration for the Self-Evolving Architecture simulator.

Every architectural constant used by the simulation lives here so that the
assumptions are explicit, documented in one place and easy to change.

Time base
---------
All simulation time is expressed in *reference cycles* of a fixed 1 GHz
reference clock, i.e. 1 reference cycle == 1 ns.  The CPU frequency of the
simulated cores is variable (1-4 GHz), so core-clock cycles are converted to
reference cycles through the frequency.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Resource ranges available to the Decision Engine
# --------------------------------------------------------------------------
CORE_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8]
FREQ_OPTIONS_GHZ = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
CACHE_OPTIONS_KB = [256, 512, 1024, 2048, 4096]
CHANNEL_OPTIONS = [1, 2, 3, 4]
REPLACEMENT_POLICIES = ("LRU", "FIFO")

# --------------------------------------------------------------------------
# Simulation engine
# --------------------------------------------------------------------------
REFERENCE_CLOCK_GHZ = 1.0          # 1 reference cycle = 1 ns
DEFAULT_STEP_CYCLES = 10_000       # reference cycles simulated per engine step
DEFAULT_SAMPLING_INTERVAL = 5      # engine steps per monitoring sample
MAX_STEPS = 20_000                 # safety stop

# --------------------------------------------------------------------------
# CPU model
# --------------------------------------------------------------------------
LLC_HIT_BASE_CYCLES = 10           # LLC hit latency (core cycles) at 256 KB
LLC_HIT_CYCLES_PER_DOUBLING = 3    # bigger caches are slower (wire delay)
LLC_HIT_EXPOSURE = 0.35            # fraction of LLC hit latency not hidden by OoO

# --------------------------------------------------------------------------
# Cache model
# --------------------------------------------------------------------------
CACHE_LINE_BYTES = 64
MISS_CURVE_EXPONENT = 0.5          # "sqrt(2) rule": miss ~ (capacity)^-0.5
FIFO_LOCALITY_PENALTY = 0.15       # FIFO miss multiplier slope vs temporal locality
FIFO_LOCALITY_PIVOT = 0.4          # locality at which LRU and FIFO are equivalent
PREFETCH_MAX_COVERAGE = 0.80       # fraction of stream misses a prefetcher can cover
PREFETCH_POLLUTION = 0.06          # extra misses caused by useless prefetches
WRITEBACK_FACTOR = 0.5             # dirty-line writeback traffic per write fraction
ADJACENT_LINE_FILL = 2.0           # lines installed per miss (128 B sector pair)

# --------------------------------------------------------------------------
# Memory controller / DRAM
# --------------------------------------------------------------------------
CHANNEL_BANDWIDTH_GBPS = 12.8      # DDR4-1600 x 64 bit per channel (bytes/ns)
DRAM_UNLOADED_LATENCY_NS = 75.0    # row access + controller + interconnect
QUEUE_DELAY_SCALE_NS = 30.0        # M/M/1-style queueing delay scale
MAX_BUS_UTILIZATION = 0.97         # bandwidth saturation ceiling
QUEUE_ENTRIES_PER_CHANNEL = 32

# --------------------------------------------------------------------------
# Power model (Watts)
# --------------------------------------------------------------------------
CORE_STATIC_W = 0.8                # leakage per powered core
CORE_DYN_COEFF = 1.6               # P_dyn = C * f * V(f)^2
VOLTAGE_BASE_V = 0.6               # V(f) = V0 + k*f
VOLTAGE_SLOPE_V_PER_GHZ = 0.15
CORE_IDLE_ACTIVITY = 0.35          # dynamic power fraction burnt when stalled
CACHE_W_PER_MB = 1.0
CHANNEL_STATIC_W = 1.5
CHANNEL_DYN_W = 2.5                # at 100 % bandwidth utilisation
PREFETCHER_W = 0.4
DEFAULT_POWER_BUDGET_W = 45.0

# --------------------------------------------------------------------------
# Reconfiguration cost model
#   latency  : reference cycles the change takes
#   factor   : fraction of normal throughput retained while it is in progress
#   energy_uj: fixed energy overhead of the change (micro-joules)
# --------------------------------------------------------------------------
RECONFIG_COSTS = {
    "freq":     {"latency": 15_000, "factor": 0.50, "energy_uj": 6.0},   # PLL relock + voltage ramp
    "cores":    {"latency": 40_000, "factor": 0.60, "energy_uj": 12.0},  # power-gate wake + thread migration + context warm-up
    "cache":    {"latency": 5_000,  "factor": 0.70, "energy_uj": 4.0},   # way enable/disable (+ dirty flush)
    "channels": {"latency": 50_000, "factor": 0.50, "energy_uj": 15.0},  # self-refresh exit + address-interleave remap
    "prefetch": {"latency": 500,    "factor": 0.95, "energy_uj": 0.2},
    "policy":   {"latency": 2_000,  "factor": 0.90, "energy_uj": 0.5},
}
RECONFIG_CONTROLLER_CYCLES = 2_000         # fixed controller sequencing overhead
CORE_MIGRATION_UJ_PER_CORE = 3.0           # extra energy per core switched
FLUSH_ENERGY_UJ_PER_MB = 20.0              # energy to write back 1 MB of dirty data

# --------------------------------------------------------------------------
# Monitoring / decision overhead (reference cycles stolen from execution)
# --------------------------------------------------------------------------
MONITOR_OVERHEAD_CYCLES = 150              # PMU read + sample interrupt, per sample
CLASSIFY_OVERHEAD_CYCLES = 50              # per classification
DECISION_BASE_CYCLES = 400                 # per optimisation run
DECISION_CYCLES_PER_CANDIDATE = 4          # per candidate configuration evaluated

# --------------------------------------------------------------------------
# Workload characterisation thresholds (with hysteresis bands)
# --------------------------------------------------------------------------
CLASSIFIER_THRESHOLDS = {
    "cpu_high": 0.60,       # CPU util above this suggests compute-bound
    "cpu_low": 0.40,        # CPU util below this suggests memory-bound
    "bw_low": 0.35,         # memory BW util below this suggests compute-bound
    "bw_high": 0.45,        # memory BW util above this suggests memory-bound
    "miss_high": 0.30,      # LLC miss rate above this suggests memory-bound
    "miss_low": 0.15,       # LLC miss rate below this is required for compute-bound
    "stall_high": 0.55,     # memory-stall fraction above this suggests memory-bound
    "hysteresis": 0.05,     # thresholds relax by this much for the current class
}
DEFAULT_CONFIRM_SAMPLES = 2                # consecutive samples to accept a new class
DEFAULT_WARMUP_SAMPLES = 2                 # initial samples ignored (cold caches, start-up)

# --------------------------------------------------------------------------
# Decision engine
# --------------------------------------------------------------------------
DEFAULT_MIN_GAIN = 0.05                    # 5 % predicted improvement required
DEFAULT_BENEFIT_MARGIN = 1.25              # saved cycles must exceed cost x margin
DEFAULT_EXPECTED_PHASE_STEPS = 50          # prior for how long a phase lasts
RESIDENCY_EMA_ALPHA = 0.5                  # learning rate of observed residency
DRIFT_REEVALUATE = 0.15
REFINE_AFTER_SAMPLES = 2                   # clean samples after a reconfig before refinement
EFFICIENCY_TOLERANCE = 0.03                # among configs within 3 % of best, pick lowest power                    # re-evaluate if IPC drifts > 15 % within a class

# Policy-constrained search spaces for each workload class.
POLICY_SPACE = {
    "Compute-bound": {
        "cores": [4, 5, 6, 7, 8],
        "freq": [2.5, 3.0, 3.5, 4.0],
        "cache": [256, 512, 1024],
        "channels": [1, 2],
        "prefetch": [False],
        "policy": ["LRU"],
    },
    "Memory-bound": {
        "cores": [2, 3, 4, 5, 6, 7, 8],
        "freq": [1.0, 1.5, 2.0, 2.5],
        "cache": [2048, 4096],
        "channels": [3, 4],
        "prefetch": [True],
        "policy": ["FIFO"],
    },
    "Balanced": {
        "cores": [4, 5, 6],
        "freq": [2.0, 2.5, 3.0],
        "cache": [1024, 2048, 4096],
        "channels": [2, 3],
        "prefetch": [True],
        "policy": ["LRU"],
    },
}

# Static (conventional) baseline: a reasonable "one size fits all" design point.
DEFAULT_STATIC_CONFIG = {
    "cores": 4,
    "freq_ghz": 3.0,
    "cache_kb": 1024,
    "mem_channels": 2,
    "prefetch": False,
    "policy": "LRU",
}

# --------------------------------------------------------------------------
# Workload phase profiles (ground truth used only by the workload generator)
# --------------------------------------------------------------------------
WORKLOAD_PROFILES = {
    "compute": {
        "label": "Compute-intensive",
        "mem_ratio": 0.20,          # memory references per instruction
        "l1_hit_rate": 0.97,
        "working_set_kb": 384,
        "cpi_exec": 0.45,           # core CPI without LLC/memory stalls
        "parallel_fraction": 0.97,
        "mlp": 2.0,                 # memory-level parallelism (overlapped misses)
        "prefetchability": 0.20,    # fraction of misses following a stream
        "temporal_locality": 0.80,
        "base_miss_rate": 0.50,     # LLC miss rate with (almost) no capacity
        "min_miss_rate": 0.02,      # compulsory floor when the WS fits
        "write_fraction": 0.20,
        "ws_overlap": 0.10,         # share of old cache contents useful on entry
    },
    "memory": {
        "label": "Memory-intensive",
        "mem_ratio": 0.45,
        "l1_hit_rate": 0.75,
        "working_set_kb": 32768,
        "cpi_exec": 0.60,
        "parallel_fraction": 0.90,
        "mlp": 10.0,
        "prefetchability": 0.75,
        "temporal_locality": 0.20,
        "base_miss_rate": 0.95,
        "min_miss_rate": 0.05,
        "write_fraction": 0.30,
        "ws_overlap": 0.10,
    },
    "mixed": {
        "label": "Mixed",
        "mem_ratio": 0.22,
        "l1_hit_rate": 0.92,
        "working_set_kb": 6144,
        "cpi_exec": 0.55,
        "parallel_fraction": 0.85,
        "mlp": 4.0,
        "prefetchability": 0.50,
        "temporal_locality": 0.50,
        "base_miss_rate": 0.60,
        "min_miss_rate": 0.03,
        "write_fraction": 0.25,
        "ws_overlap": 0.10,
    },
}
