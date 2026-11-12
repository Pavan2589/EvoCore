"""Reconfiguration mechanics, overhead accounting and static-vs-adaptive comparison."""

import pytest

from config import default_config as K
from simulation.analysis import compare, improvement_pct, run_comparison
from simulation.cache import Cache
from simulation.engine import SimulationEngine
from simulation.models import ArchConfig
from simulation.reconfiguration_controller import ReconfigurationController
from simulation.scenarios import SCENARIOS, phase
from tests.conftest import STATIC


def test_no_change_costs_nothing():
    est = ReconfigurationController.estimate(STATIC, STATIC)
    assert est.duration == 0 and est.lost_cycles == 0 and est.energy_uj == 0


def test_cost_model_parallel_components_and_flush():
    new = STATIC.with_(freq_ghz=2.0, mem_channels=4)
    est = ReconfigurationController.estimate(STATIC, new)
    c = K.RECONFIG_COSTS
    assert est.duration == K.RECONFIG_CONTROLLER_CYCLES + max(c["freq"]["latency"], c["channels"]["latency"])
    assert est.factor == min(c["freq"]["factor"], c["channels"]["factor"])
    assert est.lost_cycles == pytest.approx(est.duration * (1 - est.factor))   # controller's resource-only estimate
    shrink = ReconfigurationController.estimate(STATIC.with_(cache_kb=4096), STATIC.with_(cache_kb=256),
                                                resident_kb=4096, dirty_fraction=0.3)
    plain = ReconfigurationController.estimate(STATIC.with_(cache_kb=4096), STATIC.with_(cache_kb=8192 // 2 // 2))
    assert shrink.flush_kb == pytest.approx((4096 - 256) * 0.3)
    assert shrink.duration > plain.duration


def test_segments_split_step_around_transition():
    ctrl = ReconfigurationController()
    new = STATIC.with_(cores=8, freq_ghz=2.5)
    rec = ctrl.begin(5_000, STATIC, new, 0, 0, 0.2, "t")
    segs = ctrl.segments(0, 10_000, STATIC)
    assert [s.reconfiguring for s in segs] == [False, True]
    assert segs[1].config == ArchConfig.transitional(STATIC, new)
    assert segs[1].factor == rec.throughput_factor
    assert sum(s.cycles for s in segs) == pytest.approx(10_000)
    t_end = rec.end_cycle
    segs = ctrl.segments(t_end - 1_000, t_end + 4_000, STATIC)
    assert [s.reconfiguring for s in segs] == [True, False] and segs[1].config == new
    assert ctrl.complete_if_due(t_end) is rec and not ctrl.busy


def test_cache_state_survives_resizing():
    c = Cache(1024)
    c.resident_kb = 900
    assert c.resize(4096) == 0 and c.resident_kb == 900       # grow: data kept, new space cold
    c.dirty_fraction = 0.5
    assert c.resize(512) == pytest.approx(388 * 0.5)          # shrink: dirty evictions written back
    assert c.resident_kb == 512


def test_reconfiguration_changes_actual_parameters_and_keeps_state(params):
    e = SimulationEngine(SCENARIOS["dynamic"].phases, params)
    retired_before = None
    while not e.finished:
        busy_before = e.controller.busy
        retired_before = e.workload.retired
        e.step()
        assert e.workload.retired >= retired_before          # never restarts
        if busy_before and not e.controller.busy and e.controller.history[-1].completed:
            rec = e.controller.history[-1]
            assert e.config == rec.new                       # parameters actually changed
            assert e.cache.capacity_kb == rec.new.cache_kb
            assert e.cache.policy == rec.new.policy
    assert len(e.controller.history) >= 3
    for rec in e.controller.history:
        assert rec.completed and rec.end_cycle > rec.start_cycle
        assert rec.duration == pytest.approx(rec.end_cycle - rec.start_cycle)
        # lost work vs an instantaneous switch: positive, bounded by the transition window
        assert 0 < rec.lost_cycles <= rec.duration
        assert rec.old != rec.new and rec.energy_uj > 0
    # recorded history reflects the resources in use
    assert {r.cores for r in e.history} != {STATIC.cores}


def test_each_transition_triggers_matching_reconfiguration(params):
    e = SimulationEngine(SCENARIOS["dynamic"].phases, params).run()
    recs = e.controller.history
    phase_starts = [p[0] for p in e.phase_changes]
    for i, start in enumerate(phase_starts):
        end = phase_starts[i + 1] if i + 1 < len(phase_starts) else e.cycle
        assert any(start <= r.start_cycle < end for r in recs), f"no reconfiguration in phase {i + 1}"
    compute_cfg = next(r.new for r in recs if r.start_cycle < phase_starts[1])
    memory_cfg = next(r.new for r in recs if phase_starts[1] <= r.start_cycle < phase_starts[2])
    assert compute_cfg.cores >= 6 and not compute_cfg.prefetch
    assert memory_cfg.mem_channels >= 3 and memory_cfg.prefetch and memory_cfg.freq_ghz < compute_cfg.freq_ghz + 0.01


def test_overhead_accounting_is_consistent(params):
    r = run_comparison(SCENARIOS["dynamic"].phases, params)
    s, a = r["static"], r["adaptive"]
    assert a["adaptation_overhead_cycles"] == pytest.approx(
        a["reconfig_lost_cycles"] + a["monitor_overhead_cycles"] + a["decision_overhead_cycles"])
    assert a["normal_execution_cycles"] + a["adaptation_overhead_cycles"] == pytest.approx(a["total_cycles"])
    assert a["adaptation_overhead_cycles"] > 0 and s["adaptation_overhead_cycles"] == 0
    assert r["improvement_pct"] == pytest.approx(
        (s["total_cycles"] - a["total_cycles"]) / s["total_cycles"] * 100)
    assert r["gross_benefit_cycles"] == pytest.approx(r["net_benefit_cycles"] + a["adaptation_overhead_cycles"])


def test_static_and_adaptive_execute_identical_work(params):
    for key in ("dynamic", "rapid_switching"):
        r = run_comparison(SCENARIOS[key].phases, params)
        assert r["static"]["instructions"] == pytest.approx(r["adaptive"]["instructions"])
        static, adaptive = r["engines"]
        assert [p.kind for p in static.workload.phases] == [p.kind for p in adaptive.workload.phases]
        assert static.controller.history == []


def test_compare_rejects_different_workloads(params):
    a = SimulationEngine([phase("compute", 10)], params).run()
    s = SimulationEngine([phase("compute", 20)], params, adaptive=False).run()
    with pytest.raises(ValueError):
        compare(s, a)


def test_improvement_formula():
    assert improvement_pct(200, 150) == 25
    assert improvement_pct(100, 110) == pytest.approx(-10)


def test_adaptive_beats_static_on_long_phases(params):
    for key in ("static_compute", "static_memory", "dynamic", "adaptive_vs_static"):
        assert run_comparison(SCENARIOS[key].phases, params)["improvement_pct"] > 10, key


def test_unchanged_workload_converges_and_stays_put(params):
    """Steady workload: adapt (plus at most one closed-loop refinement), then never again."""
    for key in ("static_compute", "static_memory"):
        r = run_comparison(SCENARIOS[key].phases, params)
        adaptive = r["engines"][1]
        recs = adaptive.controller.history
        assert 1 <= len(recs) <= 2, key
        assert all(rec.start_cycle < 0.4 * adaptive.cycle for rec in recs), key


def test_frequent_switching_no_thrashing(params):
    smart = run_comparison(SCENARIOS["rapid_switching"].phases, params)
    naive = run_comparison(SCENARIOS["rapid_switching"].phases, params, naive=True)
    assert smart["adaptive"]["reconfig_count"] <= 3
    assert naive["adaptive"]["reconfig_count"] > 5 * smart["adaptive"]["reconfig_count"]
    assert naive["improvement_pct"] < 0                       # blind adaptation hurts
    assert smart["improvement_pct"] > naive["improvement_pct"] + 30


def test_short_workload_adaptation_not_beneficial(params):
    r = run_comparison(SCENARIOS["short"].phases, params)
    assert r["improvement_pct"] <= 0.5                        # no meaningful gain possible
    assert r["adaptive"]["reconfig_count"] == 0               # and it does not try


def test_adaptation_cost_can_exceed_benefit(params):
    # A workload that ends shortly after the first decision: the reconfiguration
    # cannot be amortised and the adaptive system is slower than static.
    r = run_comparison([phase("compute", 20)], params)
    assert r["adaptive"]["reconfig_count"] == 1 and r["improvement_pct"] < 0


def test_overhead_matches_instantaneous_counterfactual(params, monkeypatch):
    """Reported reconfiguration overhead ~= extra time vs an ideal instant switch."""
    from simulation import reconfiguration_controller as rc
    real = run_comparison(SCENARIOS["static_compute"].phases, params)["adaptive"]
    orig = rc.ReconfigurationController.estimate

    def instant(*a, **k):
        e = orig(*a, **k)
        e.duration, e.lost_cycles = 0.0, 0.0
        return e
    monkeypatch.setattr(rc.ReconfigurationController, "estimate", staticmethod(instant))
    ideal = run_comparison(SCENARIOS["static_compute"].phases, params)["adaptive"]
    assert ideal["reconfig_lost_cycles"] == 0
    extra = real["total_cycles"] - ideal["total_cycles"]
    assert extra == pytest.approx(real["reconfig_lost_cycles"], rel=0.10)
