"""Skip NCCL collectives that are exact identities for a one-rank group.

This is a narrowly scoped B200 compatibility workaround: a world-size-one
DeepSpeed initialization currently blocks in NCCL broadcast. The wrappers keep
Torch's original behavior for any group containing more than one rank and for
asynchronous operations.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable


def install_single_rank_collective_identity() -> None:
    import torch.distributed as dist

    def is_single_rank(args: tuple[Any, ...], kwargs: dict[str, Any], group_index: int) -> bool:
        if not dist.is_initialized() or kwargs.get("async_op", False):
            return False
        group = kwargs.get("group")
        if group is None and len(args) > group_index:
            group = args[group_index]
        try:
            return dist.get_world_size(group=group) == 1
        except (RuntimeError, ValueError):
            return False

    def skip_identity(name: str, group_index: int) -> None:
        original: Callable[..., Any] = getattr(dist, name)

        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if is_single_rank(args, kwargs, group_index):
                return None
            return original(*args, **kwargs)

        setattr(dist, name, wrapper)

    # DeepSpeed initializes with broadcast and performs gradient all-reduce.
    # With one participant both operations return their input unchanged.
    skip_identity("broadcast", group_index=2)
    skip_identity("all_reduce", group_index=2)
    skip_identity("barrier", group_index=0)
