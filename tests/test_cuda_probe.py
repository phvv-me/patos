"""The pair table's probe, one 16-byte load a slot, against the two-load probe it replaced."""

from typing import NamedTuple

import cupy as cp
import numpy as np
import pytest

from patos.cuda.primitives import EMPTY_KEY, PairTable, device_splitmix
from patos.cuda.typed import Matrix, Vector, device, i64, items, kernel, u64

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


class Probes(NamedTuple):
    """A table, what it should answer for the keys asked, and what each probe answered."""

    table: PairTable
    wanted: list[int]
    found: np.ndarray


@device
def probed(mask: u64, slots: Vector[u64], key: u64) -> u64:
    """`PairTable.get` as it was, two loads a probe, answering the empty key for none."""
    slot = device_splitmix(key) & mask
    while slots[slot * 2] != EMPTY_KEY:
        if slots[slot * 2] == key:
            return slots[slot * 2 + 1]
        slot = (slot + 1) & mask
    return slots[slot * 2]


@kernel
def getting(table: PairTable, keys: Vector[u64], found: Matrix[i64]) -> None:
    for item in items(keys.size):
        found[item, 0] = table.get(keys[item])
        found[item, 1] = probed(table.mask, table.slots, keys[item])


@pytest.fixture(params=[0, 1, 7, 200, 3000])
def probes(request: pytest.FixtureRequest) -> Probes:
    """Random pairs asked for by the keys they hold, strays, zero and the empty key.

    The payloads reach past `i64`, which the answer reads as negative.
    """
    rng, entries = np.random.default_rng(request.param), request.param
    keys = np.unique(rng.integers(0, EMPTY_KEY, entries, dtype=np.uint64))
    payloads = rng.integers(0, 2**64, len(keys), dtype=np.uint64)
    table = PairTable.build(list(zip(keys.tolist(), payloads.tolist(), strict=True)))
    strays = rng.integers(0, EMPTY_KEY, entries + 1, dtype=np.uint64)
    asked = np.concatenate([keys, strays, np.array([0, EMPTY_KEY], dtype=np.uint64)])
    found = cp.zeros((len(asked), 2), np.int64)
    getting[len(asked)](table, cp.asarray(asked), found)
    stored = dict(zip(keys.tolist(), payloads.astype(np.int64).tolist(), strict=True))
    return Probes(table, [stored.get(key, -1) for key in asked.tolist()], found.get())


def test_a_probe_on_one_load_answers_what_the_host_stored_and_the_two_load_probe_did(
    probes: Probes,
) -> None:
    assert probes.found[:, 0].tolist() == probes.wanted
    assert probes.found[:, 1].tolist() == probes.wanted


def test_the_pairs_sit_on_16_bytes_and_each_probe_is_one_vector_load(probes: Probes) -> None:
    program = getting.dispatcher.inspect_asm(next(iter(getting.dispatcher.overloads)))
    assert cp.asarray(probes.table.slots).data.ptr % 16 == 0
    assert "ld.global.nc.v2.u64" in program
