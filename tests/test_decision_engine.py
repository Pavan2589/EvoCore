"""Decision engine: policies, power budget, cost/benefit gate."""

from dataclasses import replace

from simulation.decision_engine import DecisionEngine, policy_candidates
from simulation.models import ArchConfig, Classification
from simulation.power_model import peak_power
from simulation.reconfiguration_controller import ReconfigurationController
from simulation.workload_classifier import BALANCED, COMPUTE, MEMORY
from tests.conftest import STATIC

# Realistic counter values measured on the static config (taken from simulation runs)
COMPUTE_COUNTERS = dict(cpu_util=0.86, core_activity=0.98, mem_bw_util=0.02, miss_rate=0.03,
                        exposed_miss_rate=0.03, mem_latency_ns=75, mem_stall_frac=0.03, t_exec_ns=0.17,
                        t_mem_ns=0.005, llc_api=0.006, bytes_per_instr=0.015, stream_fraction=0.2, ipc=7.3)
MEMORY_COUNTERS = dict(cpu_util=0.14, core_activity=0.92, mem_bw_util=0.60, miss_rate=0.79,
                       exposed_miss_rate=0.79, mem_latency_ns=120, mem_stall_frac=0.81, t_exec_ns=0.24,
                       t_mem_ns=1.0, llc_api=0.1125, bytes_per_instr=6.5, stream_fraction=0.75, ipc=0.8)


def changed(cls):
    return Classification(raw=cls, stable=cls, changed=True, streak=0, reason="test")


def run(params, sample, cls, current=STATIC, ctrl=None):
    eng = DecisionEngine(params)
    return eng, eng.evaluate(sample, changed(cls), current, 100, ctrl or ReconfigurationController(), 512, 0.2)


def test_candidates_respect_policy_and_power_budget():
    for cls in (COMPUTE, MEMORY, BALANCED):
        cands = policy_candidates(cls, 45)
        assert cands and all(peak_power(c) <= 45 for c in cands)
    assert all(c.mem_channels >= 3 and c.prefetch and c.freq_ghz <= 2.5 for c in policy_candidates(MEMORY, 45))
    assert all(c.freq_ghz >= 2.5 and not c.prefetch and c.mem_channels <= 2 for c in policy_candidates(COMPUTE, 45))
    assert len(policy_candidates(COMPUTE, 20)) < len(policy_candidates(COMPUTE, 45))


def test_memory_bound_targets_memory_resources(params, sample_factory):
    _, d = run(params, sample_factory(**MEMORY_COUNTERS), MEMORY)
    assert d.action == "reconfigure"
    t = d.target
    assert t.mem_channels > STATIC.mem_channels and t.cache_kb > STATIC.cache_kb and t.prefetch
    assert t.freq_ghz < STATIC.freq_ghz            # does not waste power on frequency
    assert d.predicted_gain > 0.5


def test_compute_bound_targets_cpu_resources(params, sample_factory):
    _, d = run(params, sample_factory(**COMPUTE_COUNTERS), COMPUTE)
    assert d.action == "reconfigure"
    assert d.target.cores > STATIC.cores and not d.target.prefetch
    assert d.target.mem_channels <= STATIC.mem_channels


def test_already_optimal_configuration_is_held(params):
    """A system that already starts in the configuration the engine would pick for
    a steady workload must evaluate, hold, and never reconfigure."""
    from simulation.engine import SimulationEngine
    from simulation.scenarios import SCENARIOS
    for key in ("static_memory", "static_compute"):
        first = SimulationEngine(SCENARIOS[key].phases, params).run()
        assert 1 <= len(first.controller.history) <= 2      # adapt (+ one refinement)
        tuned = replace(params, initial_config=first.config)
        e = SimulationEngine(SCENARIOS[key].phases, tuned).run()
        assert e.controller.history == [], key
        assert e.decider.decisions and all(d.action == "hold" for d in e.decider.decisions)


def test_cost_exceeding_benefit_is_rejected(params, sample_factory):
    p = replace(params, expected_phase_steps=1, sampling_interval=1, benefit_margin=5.0)
    _, d = run(p, sample_factory(**COMPUTE_COUNTERS), COMPUTE)
    assert d.action == "hold" and "does not justify" in d.reason
    assert d.expected_saved_cycles < 5.0 * d.expected_cost_cycles


def test_gain_below_threshold_is_rejected(params, sample_factory):
    p = replace(params, min_gain=5.0)               # demand +500 %
    _, d = run(p, sample_factory(**COMPUTE_COUNTERS), COMPUTE)
    assert d.action == "hold" and "below" in d.reason


def test_naive_mode_skips_cost_check(params, sample_factory):
    p = replace(params, expected_phase_steps=1, sampling_interval=1, benefit_margin=5.0, cost_aware=False)
    _, d = run(p, sample_factory(**COMPUTE_COUNTERS), COMPUTE)
    assert d.action == "reconfigure"


def test_no_decision_while_reconfiguring(params, sample_factory):
    ctrl = ReconfigurationController()
    ctrl.begin(0, STATIC, STATIC.with_(cores=8), 0, 0, 0.1, "test")
    _, d = run(params, sample_factory(**COMPUTE_COUNTERS), COMPUTE, ctrl=ctrl)
    assert d is None
    _, d = run(params, sample_factory(reconfiguring=True, **COMPUTE_COUNTERS), COMPUTE)
    assert d is None


def test_engine_does_not_rerun_without_trigger(params, sample_factory):
    eng = DecisionEngine(params)
    ctrl = ReconfigurationController()
    s = sample_factory(**COMPUTE_COUNTERS)
    assert eng.evaluate(s, changed(COMPUTE), STATIC, 10, ctrl, 0, 0) is not None
    same = Classification(raw=COMPUTE, stable=COMPUTE, changed=False, streak=0, reason="")
    assert eng.evaluate(s, same, STATIC, 15, ctrl, 0, 0) is None      # baseline
    assert eng.evaluate(s, same, STATIC, 20, ctrl, 0, 0) is None      # no drift -> idle


def test_residency_learning_shortens_horizon(params):
    eng = DecisionEngine(params)
    h0 = eng.horizon_steps(0)
    for step in range(10, 100, 10):
        eng.on_class_change(COMPUTE, step)
    assert eng.horizon_steps(100) < h0


def test_predictor_is_reasonably_accurate(params):
    """Predicted throughput of a new config vs. what the simulator then measures."""
    from simulation.engine import SimulationEngine
    from simulation.scenarios import SCENARIOS
    e = SimulationEngine(SCENARIOS["static_memory"].phases, params).run()
    rec = e.controller.history[0]
    d = next(d for d in e.decider.decisions if d.action == "reconfigure")
    before = [s for s in e.monitor.samples if s.cycle <= rec.start_cycle][-1]
    after = [s for s in e.monitor.samples if s.cycle > rec.end_cycle + 200_000][0]
    actual_gain = after.throughput / before.throughput - 1
    assert abs(d.predicted_gain - actual_gain) / (1 + actual_gain) < 0.35
