import threading

from patos import Shared


def test_single_flight_coalesces_concurrent_first_builders() -> None:
    builds: list[str] = []
    gate = threading.Barrier(4)
    shared: Shared[str, str] = Shared(lambda key: (builds.append(key), f"conn-{key}")[1])
    seen: list[str] = []

    hold = threading.Barrier(4)

    def worker() -> None:
        gate.wait()
        with shared.acquire("gold") as conn:
            hold.wait()
            seen.append(conn)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert builds == ["gold"]
    assert seen == ["conn-gold"] * 4


def test_zero_idle_closes_on_last_release_and_rebuilds() -> None:
    closed: list[str] = []
    counter = iter(range(10))
    shared: Shared[str, str] = Shared(lambda key: f"{key}#{next(counter)}", close=closed.append)
    with shared.acquire("a") as first, shared.acquire("a") as second:
        assert first == second == "a#0"
        assert closed == []
    assert closed == ["a#0"]
    with shared.acquire("a") as rebuilt:
        assert rebuilt == "a#1"


def test_idle_linger_reuses_until_sweep_evicts() -> None:
    closed: list[str] = []
    counter = iter(range(10))
    shared: Shared[str, str] = Shared(
        lambda key: f"{key}#{next(counter)}", close=closed.append, idle_seconds=3600.0
    )
    with shared.acquire("a") as conn:
        assert conn == "a#0"
    with shared.acquire("a") as again:
        assert again == "a#0"
    assert closed == []
    assert shared.sweep() == 0
    shared.idle_seconds = 0.0
    assert shared.sweep() == 1
    assert closed == ["a#0"]


def test_drain_closes_unheld_but_never_held_resources() -> None:
    closed: list[str] = []
    shared: Shared[str, str] = Shared(
        lambda key: f"conn-{key}", close=closed.append, idle_seconds=60.0
    )
    with shared.acquire("busy"):
        with shared.acquire("idle"):
            pass
        shared.drain()
        assert closed == ["conn-idle"]
    shared.drain()
    assert sorted(closed) == ["conn-busy", "conn-idle"]


def test_keys_are_independent_slots() -> None:
    shared: Shared[str, str] = Shared(lambda key: f"conn-{key}")
    with shared.acquire("a") as a, shared.acquire("b") as b:
        assert (a, b) == ("conn-a", "conn-b")
