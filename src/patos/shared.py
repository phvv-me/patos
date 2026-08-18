from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class _Slot[R]:
    """One keyed resource with its refcount and idle bookkeeping."""

    __slots__ = ("built", "holders", "idle_since", "lock", "resource")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.built = False
        self.resource: R | None = None
        self.holders = 0
        self.idle_since = 0.0


class Shared[K, R]:
    """A keyed, single-flight facade over expensive, closable resources.

    Concurrent first callers for one key coalesce into a single construction,
    later callers share the built resource, and a handle is only reachable
    inside the `acquire` context so no raw resource escapes its refcount.
    When the last holder exits, the resource either closes immediately or
    lingers for `idle_seconds` so a follow-up caller reuses it, the pattern
    an ssh-connection-per-host or a database handle wants.

    Thread-safe by construction and free of module-global state, one
    `Shared` instance owns one resource family.
    """

    def __init__(
        self,
        build: Callable[[K], R],
        *,
        close: Callable[[R], None] | None = None,
        idle_seconds: float = 0.0,
    ) -> None:
        """build: constructs the resource for a key, called once per key at a time.

        close: releases a resource, called under the slot lock when evicted.
        idle_seconds: how long an unheld resource lingers before closing,
            zero closing it the moment its last holder exits.
        """
        self.build = build
        self.close = close
        self.idle_seconds = idle_seconds
        self.slots: dict[K, _Slot[R]] = {}
        self.registry_lock = threading.Lock()

    @contextmanager
    def acquire(self, key: K) -> Iterator[R]:
        """Hold the shared resource for `key` for the duration of the block.

        key: the resource family member wanted, a host alias, a path.
        """
        slot = self.__slot(key)
        with slot.lock:
            if not slot.built:
                slot.resource = self.build(key)
                slot.built = True
            slot.holders += 1
            resource = slot.resource
        try:
            yield resource  # type: ignore[misc]  # built is True so resource is R
        finally:
            with slot.lock:
                slot.holders -= 1
                slot.idle_since = time.monotonic()
                if slot.holders == 0 and self.idle_seconds == 0.0:
                    self.__evict(slot)

    def drain(self) -> None:
        """Close every unheld resource now, the scope-exit companion to `sweep`."""
        with self.registry_lock:
            slots = list(self.slots.values())
        for slot in slots:
            with slot.lock:
                if slot.built and slot.holders == 0:
                    self.__evict(slot)

    def sweep(self) -> int:
        """Close every unheld resource idle past the deadline, returning how many.

        The caller owns the cadence, a monitor tick or a scope exit, so the
        facade never needs a background thread of its own.
        """
        deadline = time.monotonic() - self.idle_seconds
        evicted = 0
        with self.registry_lock:
            slots = list(self.slots.values())
        for slot in slots:
            with slot.lock:
                if slot.built and slot.holders == 0 and slot.idle_since <= deadline:
                    self.__evict(slot)
                    evicted += 1
        return evicted

    def __evict(self, slot: _Slot[R]) -> None:
        resource = slot.resource
        slot.built = False
        slot.resource = None
        if self.close is not None and resource is not None:
            self.close(resource)

    def __slot(self, key: K) -> _Slot[R]:
        with self.registry_lock:
            return self.slots.setdefault(key, _Slot())
