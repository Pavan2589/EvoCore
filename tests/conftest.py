import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.models import ArchConfig, Sample, SimulationParams  # noqa: E402

STATIC = ArchConfig(4, 3.0, 1024, 2, False, "LRU")


@pytest.fixture
def params():
    return SimulationParams(initial_config=STATIC)


def make_sample(cfg=STATIC, **kw):
    base = dict(index=10, cycle=1e5, steps=5, instructions=1e6, throughput=10.0, ipc=3.0, cpu_util=0.5,
                core_activity=0.9, mem_bw_util=0.2, hit_rate=0.8, miss_rate=0.2, exposed_miss_rate=0.2,
                mem_latency_ns=80.0, queue_util=0.1, resource_util=0.4, mem_stall_frac=0.3, t_exec_ns=0.3,
                t_mem_ns=0.1, llc_api=0.02, bytes_per_instr=0.3, stream_fraction=0.3, reconfiguring=False,
                config=cfg)
    base.update(kw)
    return Sample(**base)


@pytest.fixture
def sample_factory():
    return make_sample
