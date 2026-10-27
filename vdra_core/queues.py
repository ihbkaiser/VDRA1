"""Asynchronous cross-prompt allocation queues (Supp. Sec. E, Alg. 1 lines 16-30).

Used by the treetune integration, where every reasoning tree is built by its
own asyncio task and nodes from different prompts and depths become ready at
different times.  A node whose proxy is ready calls `await manager.submit(...)`
and receives n_v when its queue is dispatched.

Placement rule (paper): insert into the non-full queue with the fewest
remaining slots, ties broken by queue index.  A queue is dispatched when it
reaches capacity or `timeout_s` after its first insertion.  The timeout
guarantees progress, so no deadlock is possible even when only one tree is
still running.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .allocation import QueueItem, allocate_queue


@dataclass
class _Queue:
    qid: int
    items: List[QueueItem] = field(default_factory=list)
    futures: List[asyncio.Future] = field(default_factory=list)
    timer: Optional[asyncio.Task] = None
    opened_at: float = 0.0
    generation: int = 0


class AsyncQueueManager:
    def __init__(
        self,
        *,
        n_queues: int = 32,
        capacity: int = 16,
        timeout_s: float = 1.0,
        n_min: int = 1,
        on_dispatch: Optional[Callable[[int, List[QueueItem], List[int], int, float], None]] = None,
    ):
        self.capacity = int(capacity)
        self.timeout_s = float(timeout_s)
        self.n_min = int(n_min)
        self.on_dispatch = on_dispatch
        self._queues = [_Queue(qid=i) for i in range(int(n_queues))]
        self._lock: Optional[asyncio.Lock] = None
        self._space: Optional[asyncio.Condition] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.dispatch_count = 0

    def _ensure_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            # A new asyncio.run(...) per iteration: rebuild loop-bound objects.
            self._loop = loop
            self._lock = asyncio.Lock()
            self._space = asyncio.Condition(self._lock)
            for q in self._queues:
                q.items.clear()
                q.futures.clear()
                q.timer = None

    def _pick_queue(self) -> Optional[_Queue]:
        open_q = [q for q in self._queues if len(q.items) < self.capacity]
        if not open_q:
            return None
        # fewest remaining slots == most items; ties -> lowest index
        return min(open_q, key=lambda q: (self.capacity - len(q.items), q.qid))

    async def submit(self, item: QueueItem) -> int:
        self._ensure_loop()
        assert self._lock is not None and self._space is not None
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._space:
            q = self._pick_queue()
            while q is None:
                await self._space.wait()
                q = self._pick_queue()
            q.items.append(item)
            q.futures.append(fut)
            if len(q.items) == 1:
                q.opened_at = time.time()
                q.timer = asyncio.create_task(self._timeout(q, q.generation))
            if len(q.items) >= self.capacity:
                self._dispatch_locked(q)
        return await fut

    async def _timeout(self, q: _Queue, generation: int) -> None:
        try:
            await asyncio.sleep(self.timeout_s)
        except asyncio.CancelledError:
            return
        assert self._space is not None
        async with self._space:
            # Only fire for the batch that started this timer.
            if q.items and q.generation == generation:
                self._dispatch_locked(q, from_timer=True)

    def _dispatch_locked(self, q: _Queue, from_timer: bool = False) -> None:
        items, futs = list(q.items), list(q.futures)
        q.generation += 1
        q.items.clear()
        q.futures.clear()
        if q.timer is not None and not from_timer:
            q.timer.cancel()
        q.timer = None
        budget = int(sum(it.nominal for it in items))
        alloc = allocate_queue(items, n_min=self.n_min, budget=budget)
        waited = time.time() - q.opened_at
        self.dispatch_count += 1
        if self.on_dispatch is not None:
            try:
                self.on_dispatch(q.qid, items, alloc, budget, waited)
            except Exception:  # logging must never break training
                pass
        for f, n in zip(futs, alloc):
            if not f.done():
                f.set_result(int(n))
        assert self._space is not None
        self._space.notify_all()

    async def flush(self) -> None:
        """Dispatch every non-empty queue immediately (Alg. 1 lines 28-30)."""
        self._ensure_loop()
        assert self._space is not None
        async with self._space:
            for q in self._queues:
                if q.items:
                    self._dispatch_locked(q)


def partition_into_queues(items: List[QueueItem], capacity: int) -> List[List[QueueItem]]:
    """Synchronous helper for batch settings (TreePO/TreeRL/diagnostics):
    consecutive chunks of at most `capacity` items."""
    return [items[i:i + capacity] for i in range(0, len(items), capacity)]
