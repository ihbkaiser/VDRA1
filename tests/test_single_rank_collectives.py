import sys
from types import ModuleType

from modal_app.single_rank_collectives import install_single_rank_collective_identity


def _fake_distributed(world_size):
    calls = []
    dist = ModuleType("torch.distributed")
    dist.is_initialized = lambda: True
    dist.get_world_size = lambda group=None: world_size
    for name in ("_broadcast_coalesced", "broadcast", "all_reduce", "barrier"):
        setattr(dist, name, lambda *args, _name=name, **kwargs: calls.append(_name) or _name)
    return dist, calls


def _install_fake_torch(monkeypatch, dist):
    torch = ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)


def test_fsdp_coalesced_broadcast_is_skipped_for_world_size_one(monkeypatch):
    dist, calls = _fake_distributed(world_size=1)
    _install_fake_torch(monkeypatch, dist)

    install_single_rank_collective_identity()

    assert dist._broadcast_coalesced(object(), [object()], 1024, 0) is None
    assert calls == []


def test_fsdp_coalesced_broadcast_is_preserved_for_multiple_ranks(monkeypatch):
    dist, calls = _fake_distributed(world_size=2)
    _install_fake_torch(monkeypatch, dist)

    install_single_rank_collective_identity()

    assert dist._broadcast_coalesced(object(), [object()], 1024, 0) == "_broadcast_coalesced"
    assert calls == ["_broadcast_coalesced"]
