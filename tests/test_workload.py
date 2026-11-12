"""Workload generation, phase transitions and runtime monitoring."""

import pytest

from simulation.engine import SimulationEngine
from simulation.runtime_monitor import RuntimeMonitor
from simulation.scenarios import SCENARIOS
from simulation.workload_generator import PhaseSpec, WorkloadGenerator, make_characteristics
from tests.conftest import STATIC


def test_profiles_have_expected_character():
    c, m, x = (make_characteristics(k) for k in ("compute", "memory", "mixed"))
    assert c.llc_api < x.llc_api < m.llc_api            # memory traffic ordering
    assert c.working_set_kb < x.working_set_kb < m.working_set_kb
    assert c.cpi_exec < m.cpi_exec


def test_phase_boundaries_follow_retired_work():
    wg = WorkloadGenerator([PhaseSpec("compute", 100), PhaseSpec("memory", 50), PhaseSpec("mixed", 25)])
    assert wg.total_instructions == 175
    assert not wg.retire(60, 1.0) and wg.phase_index == 0
    assert wg.retire(60, 2.0) and wg.phase_index == 1      # crossed 100
    assert wg.retire(40, 3.0) and wg.phase_index == 2      # crossed 150
    wg.retire(100, 4.0)
    assert wg.finished and wg.remaining == 0
    assert wg.phase_start_cycles == [0.0, 2.0, 3.0]


def test_one_step_can_cross_several_phases():
    wg = WorkloadGenerator([PhaseSpec("compute", 10), PhaseSpec("memory", 10), PhaseSpec("mixed", 10)])
    assert wg.retire(25, 1.0)
    assert wg.phase_index == 2


def test_invalid_workloads_rejected():
    with pytest.raises(ValueError):
        WorkloadGenerator([])
    with pytest.raises(ValueError):
        WorkloadGenerator([PhaseSpec("compute", 0)])
    with pytest.raises(ValueError):
        make_characteristics("gpu")


def test_noise_is_optional_and_seeded():
    phases = [PhaseSpec("memory", 1e6)]
    assert WorkloadGenerator(phases).characteristics() == make_characteristics("memory")
    a = [WorkloadGenerator(phases, noise=0.1, seed=3).characteristics() for _ in range(2)]
    assert a[0] == a[1]                                       # same seed, same sequence
    b = WorkloadGenerator(phases, noise=0.1, seed=4).characteristics()
    assert b != a[0]


def test_phase_transitions_happen_during_execution(params):
    sc = SCENARIOS["dynamic"]
    e = SimulationEngine(sc.phases, params).run()
    assert e.finished
    assert [p[1] for p in e.phase_changes] == [0, 1, 2]
    starts = [p[0] for p in e.phase_changes]
    assert starts == sorted(starts) and starts[1] > 0
    kinds = {r.phase_kind for r in e.history}
    assert kinds == {"compute", "memory", "mixed"}
    assert sum(r.instructions for r in e.history) == pytest.approx(e.workload.total_instructions)


def test_monitor_aggregates_with_correct_weights():
    mon = RuntimeMonitor(interval=2, overhead_cycles=100)
    base = dict(cpu_util=0.0, core_activity=1.0, mem_bw_util=0.0, mem_latency_ns=80, queue_util=0,
                resource_util=0, mem_stall_frac=0, hit_rate=0.0, miss_rate=0.0, exposed_miss_rate=0,
                t_exec_ns=0.1, t_mem_ns=0, llc_api=0.01, bytes_per_instr=0.1, stream_fraction=0.2)
    assert mon.observe({**base, "cpu_util": 1.0, "hit_rate": 1.0}, 1000, 300, 2.0, False, 1000, STATIC) is None
    s = mon.observe({**base, "cpu_util": 0.0, "hit_rate": 0.0}, 3000, 100, 2.0, False, 4000, STATIC)
    assert s is not None and s.steps == 2
    assert s.cpu_util == pytest.approx(0.25)        # time-weighted: 1000 of 4000 cycles busy
    assert s.hit_rate == pytest.approx(0.75)        # instruction-weighted: 300 of 400
    assert s.ipc == pytest.approx(400 / (4000 * 2.0))
    assert mon.observe(base, 1000, 1, 2.0, False, 5000, STATIC) is None   # window reset


def test_monitoring_overhead_only_on_adaptive(params):
    phases = SCENARIOS["static_compute"].phases
    a = SimulationEngine(phases, params, adaptive=True).run()
    s = SimulationEngine(phases, params, adaptive=False).run()
    assert a.summary()["monitor_overhead_cycles"] > 0
    assert s.summary()["monitor_overhead_cycles"] == 0
    assert len(a.monitor.samples) == pytest.approx(a.step_index / params.sampling_interval, abs=1)
    assert len(s.monitor.samples) == 0
