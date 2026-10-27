"""vdra_core: framework-free implementation of VDRA's dispersion proxy and
bounded rollout allocation, shared by the treetune (SPO), verl (TreePO) and
OpenRLHF (TreeRL) integrations and by the frozen-policy diagnostics."""

from .config import VDRAConfig
from .allocation import (
    QueueItem,
    allocate_queue,
    incoming_weight,
    objective,
    solve_bounded,
    upper_bounds,
)
from .proxy import (
    Pilot,
    ProxyResult,
    build_proxy,
    constant_proxy,
    dispersion_coefficient,
    redundancy_prune,
    tail_correct,
    tanh_tv,
)
from .queues import AsyncQueueManager, partition_into_queues
from .jsonl import JsonlLogger

__all__ = [
    "VDRAConfig", "QueueItem", "allocate_queue", "incoming_weight", "objective",
    "solve_bounded", "upper_bounds", "Pilot", "ProxyResult", "build_proxy",
    "constant_proxy", "dispersion_coefficient", "redundancy_prune", "tail_correct",
    "tanh_tv", "AsyncQueueManager", "partition_into_queues", "JsonlLogger",
]
__version__ = "1.0.0"
