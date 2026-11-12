"""Workload classification, hysteresis and oscillation resistance."""

from simulation.engine import SimulationEngine
from simulation.scenarios import SCENARIOS
from simulation.workload_classifier import BALANCED, COMPUTE, MEMORY, UNKNOWN, WorkloadClassifier

COMPUTE_S = dict(cpu_util=0.85, mem_bw_util=0.03, miss_rate=0.03, mem_stall_frac=0.03)
MEMORY_S = dict(cpu_util=0.15, mem_bw_util=0.80, miss_rate=0.75, mem_stall_frac=0.80)
MIXED_S = dict(cpu_util=0.50, mem_bw_util=0.20, miss_rate=0.30, mem_stall_frac=0.30)


def test_threshold_rules(sample_factory):
    c = WorkloadClassifier()
    assert c.raw_class(sample_factory(**COMPUTE_S))[0] == COMPUTE
    assert c.raw_class(sample_factory(**MEMORY_S))[0] == MEMORY
    assert c.raw_class(sample_factory(**MIXED_S))[0] == BALANCED


def test_high_misses_alone_are_not_memory_bound(sample_factory):
    # low CPU and many misses, but no bandwidth pressure and few stalls
    c = WorkloadClassifier()
    s = sample_factory(cpu_util=0.38, mem_bw_util=0.10, miss_rate=0.40, mem_stall_frac=0.15)
    assert c.raw_class(s)[0] == BALANCED


def test_consecutive_sample_validation(sample_factory):
    c = WorkloadClassifier(confirm_samples=3)
    assert c.update(sample_factory(**MEMORY_S)).stable == UNKNOWN
    assert c.update(sample_factory(**MEMORY_S)).stable == UNKNOWN
    r = c.update(sample_factory(**MEMORY_S))
    assert r.stable == MEMORY and r.changed


def test_interrupted_streak_resets(sample_factory):
    c = WorkloadClassifier(confirm_samples=2)
    c.update(sample_factory(**COMPUTE_S))
    c.update(sample_factory(**MEMORY_S))          # breaks the streak
    assert c.update(sample_factory(**COMPUTE_S)).stable == UNKNOWN
    assert c.update(sample_factory(**COMPUTE_S)).stable == COMPUTE


def test_hysteresis_keeps_class_near_boundary(sample_factory):
    border = dict(cpu_util=0.57, mem_bw_util=0.05, miss_rate=0.05, mem_stall_frac=0.05)  # just under cpu_high
    with_h = WorkloadClassifier(confirm_samples=1, hysteresis=True)
    with_h.update(sample_factory(**COMPUTE_S))
    assert with_h.update(sample_factory(**border)).stable == COMPUTE
    without = WorkloadClassifier(confirm_samples=1, hysteresis=False)
    without.update(sample_factory(**COMPUTE_S))
    assert without.update(sample_factory(**border)).stable == BALANCED


def test_samples_during_reconfiguration_are_ignored(sample_factory):
    c = WorkloadClassifier(confirm_samples=1)
    r = c.update(sample_factory(reconfiguring=True, **MEMORY_S))
    assert r.stable == UNKNOWN and not r.changed


def test_steady_workloads_classified_once_without_oscillation(params):
    for key, expected in (("static_compute", COMPUTE), ("static_memory", MEMORY)):
        e = SimulationEngine(SCENARIOS[key].phases, params).run()
        changes = [ev for ev in e.events if ev.category == "classify"]
        assert len(changes) == 1, [c.message for c in changes]
        assert e.classifier.stable == expected


def test_dynamic_workload_classes_follow_phases(params):
    e = SimulationEngine(SCENARIOS["dynamic"].phases, params).run()
    stable_sequence = [ev.message.split(" as ")[1].split(" (")[0] for ev in e.events if ev.category == "classify"]
    assert stable_sequence == [COMPUTE, MEMORY, BALANCED]
