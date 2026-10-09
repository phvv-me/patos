import cupy as cp
import numpy as np
import pytest

from patos.cuda.graphs import Arena, Captured, Graphs, Unrecordable, branch, signal

_SIZES = (1, 513, 40_000_000, 100)


@pytest.fixture
def lane():
    """A stream that is current for the test."""
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        yield stream


@pytest.fixture
def store() -> Graphs[str]:
    return Graphs(cp)


def prefix_sums(source: cp.ndarray) -> cp.ndarray:
    """The running sum of twice the source plus one.

    source: an `int32` array with shape `[1000]`. Returns an `int32` array of the same shape.
    """
    return cp.cumsum(source * 2 + 1)


def expected(factor: int) -> np.ndarray:
    """What `prefix_sums` answers for `arange(1000) * factor`, an `int64` array of shape [1000]."""
    return np.cumsum(np.arange(1000) * factor * 2 + 1)


def record(store: Graphs[str], lane: cp.cuda.Stream, source: cp.ndarray) -> Captured:
    """Record `prefix_sums` over `source`, an `int32` array of shape `[1000]`, after one run."""
    prefix_sums(source)
    lane.synchronize()
    return store.capture("pass", lambda: prefix_sums(source), stream=lane)


def test_arena_hands_out_disjoint_aligned_memory() -> None:
    arena = Arena(cp)
    pointers = [arena.malloc(size) for size in _SIZES]
    sized = zip(pointers, _SIZES, strict=True)
    spans = sorted((pointer.ptr, pointer.ptr + size) for pointer, size in sized)
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:], strict=False))
    assert all(pointer.ptr % 512 == 0 and arena.holds(pointer.ptr) for pointer in pointers)


def test_arena_overlays_the_same_addresses_after_a_rewind() -> None:
    arena = Arena(cp)
    first = arena.malloc(1)
    held = arena.nbytes
    arena.rewind()
    assert arena.malloc(1).ptr == first.ptr
    assert arena.nbytes == held


@pytest.mark.parametrize("factor", [1, 3])
def test_a_recording_replays_the_pass_on_new_data(store, lane, factor: int) -> None:
    source = cp.arange(1000, dtype=cp.int32)
    captured = record(store, lane, source)
    source[:] = cp.arange(1000, dtype=cp.int32) * factor
    store.launch(captured, lane)
    lane.synchronize()
    assert np.array_equal(cp.asnumpy(captured.result), expected(factor))


def test_a_recording_owns_its_temporaries_and_the_pool_gives_it_none(store, lane) -> None:
    source = cp.arange(1000, dtype=cp.int32)
    prefix_sums(source)
    used = cp.get_default_memory_pool().used_bytes()
    captured = record(store, lane, source)
    assert cp.get_default_memory_pool().used_bytes() == used
    assert store.arena.holds(captured.result.data.ptr)


def test_a_recording_survives_the_pool_being_churned_between_replays(store, lane) -> None:
    captured = record(store, lane, cp.arange(1000, dtype=cp.int32))
    for _ in range(8):
        cp.empty(1 << 22, dtype=cp.uint8).fill(255)
    store.launch(captured, lane)
    lane.synchronize()
    assert np.array_equal(cp.asnumpy(captured.result), expected(1))


def test_a_pass_that_reads_the_device_is_refused(store, lane) -> None:
    value = cp.ones(4, dtype=cp.int32)
    with pytest.raises(Unrecordable):
        store.capture("read", lambda: int(value[0]), stream=lane)
    assert store.get("read") is None


def test_a_refused_capture_leaves_the_stream_able_to_record(store, lane) -> None:
    value = cp.ones(4, dtype=cp.int32)
    with pytest.raises(Unrecordable):
        store.capture("read", lambda: int(value[0]), stream=lane)
    assert store.capture("after", lambda: value + 1, stream=lane) is store.get("after")


def choose(source: cp.ndarray, *, flag: cp.ndarray) -> cp.ndarray:
    """The source doubled where the flag is set and raised by a hundred otherwise, plus one.

    source: an `int32` array with shape `[8]`. flag: an `int32` array with shape `[1]`.
    Returns an `int32` array with shape `[8]`.
    """
    out = cp.zeros(8, dtype=cp.int32)
    branch(
        flag,
        lambda: out.__setitem__(slice(None), source * 2),
        lambda: out.__setitem__(slice(None), source + 100),
    )
    return out + 1


def branched(flag: int) -> np.ndarray:
    """What `choose` answers for the flag, as an `int64` array with shape `[8]`."""
    return np.arange(8) * 2 + 1 if flag else np.arange(8) + 101


@pytest.mark.parametrize("flag", [0, 1])
def test_a_branch_runs_on_the_host_in_a_pass_run_as_usual(lane, flag: int) -> None:
    result = choose(cp.arange(8, dtype=cp.int32), flag=cp.full(1, flag, dtype=cp.int32))
    assert np.array_equal(cp.asnumpy(result), branched(flag))


@pytest.mark.parametrize("flag", [0, 1])
def test_a_recorded_branch_follows_the_flag_each_replay_finds(store, lane, flag: int) -> None:
    raised, source = cp.zeros(1, dtype=cp.int32), cp.arange(8, dtype=cp.int32)
    choose(source, flag=raised)
    captured = store.capture("branch", lambda: choose(source, flag=raised), stream=lane)
    raised.fill(flag)
    store.launch(captured, lane)
    lane.synchronize()
    assert np.array_equal(cp.asnumpy(captured.result), branched(flag))


def test_the_oldest_recording_is_dropped_past_the_limit(lane) -> None:
    store: Graphs[str] = Graphs(cp, limit=2)
    for key in "abc":
        store.capture(key, lambda: cp.zeros(4, dtype=cp.int32) + 1, stream=lane)
    assert [store.get(key) is not None for key in "abc"] == [False, True, True]


def signalled(seen: np.ndarray, event: cp.cuda.Event, stream: cp.cuda.Stream) -> cp.ndarray:
    """Write seven, copy it to the host words `seen`, signal `event`, then keep the device busy.

    Returns the seven, an `int32` array with shape `[1]`.
    """
    busy = cp.ones(1 << 22, dtype=cp.int32)
    value = cp.full(1, 7, dtype=cp.int32)
    runtime = cp.cuda.runtime
    runtime.memcpyAsync(
        seen.ctypes.data, value.data.ptr, 4, runtime.memcpyDeviceToHost, stream.ptr
    )
    signal(cp, event)
    for _ in range(400):
        cp.cumsum(busy, out=busy)
    return value


@pytest.fixture
def waits() -> tuple[np.ndarray, cp.cuda.Event, cp.cuda.Event]:
    """A pinned host word, the event a replay signals and the event after the replay.

    The word is an `int32` array with shape `[1]`. The signal event is also signalled once
    outside a recording, where a signal is the plain record.
    """
    memory = cp.cuda.alloc_pinned_memory(4)
    event, tail = (cp.cuda.Event(disable_timing=True) for _ in range(2))
    signal(cp, event)
    event.synchronize()
    return np.frombuffer(memory, dtype=np.int32, count=1), event, tail


def test_the_host_waits_for_a_signal_while_the_rest_of_a_replay_runs(store, lane, waits) -> None:
    seen, event, tail = waits
    captured = store.capture("signal", lambda: signalled(seen, event, lane), stream=lane)
    for _ in range(2):
        seen[0] = 0
        store.launch(captured, lane)
        tail.record(lane)
        event.synchronize()
        assert (seen[0], tail.done) == (7, False)
        lane.synchronize()
