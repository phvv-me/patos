"""The pair table's probe, one 16-byte load a slot along cuco's probe sequence, against cuco's own
`find` over the same map, both built by cuco's insert on the device."""

import re
from typing import NamedTuple

import cupy as cp
import numpy as np
import pytest

from patos.cuda.primitives import EMPTY_KEY, PairTable, StaticMap
from patos.cuda.typed import Matrix, Vector, i64, items, kernel, u64

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


class Probes(NamedTuple):
    """A table, what it should answer for the keys asked, and what each lookup answered."""

    table: PairTable
    wanted: list[int]
    found: np.ndarray


@kernel
def looking_up(table: PairTable, held: StaticMap, keys: Vector[u64], found: Matrix[i64]) -> None:
    for item in items(keys.size):
        found[item, 0] = table.get(keys[item])
        found[item, 1] = held.find(keys[item])


@kernel
def getting(table: PairTable, keys: Vector[u64], found: Vector[i64]) -> None:
    for item in items(keys.size):
        found[item] = table.get(keys[item])


@pytest.fixture(params=[0, 1, 7, 200, 3000])
def probes(request: pytest.FixtureRequest) -> Probes:
    """Random pairs asked for by the keys they hold, strays, zero and the empty key.

    The payloads reach past `i64`, which the answer reads as negative.
    """
    rng, entries = np.random.default_rng(request.param), request.param
    keys = np.unique(rng.integers(0, EMPTY_KEY, entries, dtype=np.uint64))
    payloads = rng.integers(0, 2**64, len(keys), dtype=np.uint64)
    table = PairTable.build(dict(zip(keys.tolist(), payloads.tolist(), strict=True)))
    strays = rng.integers(0, EMPTY_KEY, entries + 1, dtype=np.uint64)
    asked = np.concatenate([keys, strays, np.array([0, EMPTY_KEY], dtype=np.uint64)])
    found = cp.zeros((len(asked), 2), np.int64)
    looking_up[len(asked)](table, StaticMap.of(table), cp.asarray(asked), found)
    stored = dict(zip(keys.tolist(), payloads.astype(np.int64).tolist(), strict=True))
    return Probes(table, [stored.get(key, -1) for key in asked.tolist()], found.get())


def test_the_probe_answers_what_was_stored_as_cuco_find_does(probes: Probes) -> None:
    assert probes.found[:, 0].tolist() == probes.wanted
    assert probes.found[:, 1].tolist() == probes.wanted


def test_the_pairs_sit_on_16_bytes_and_each_probe_is_one_vector_load(probes: Probes) -> None:
    """cuco's insert and probing iterator inline, and a probe loads its slot once, read-only."""
    getting[1](probes.table, cp.zeros(1, np.uint64), cp.zeros(1, np.int64))
    program = getting.dispatcher.inspect_sass(next(iter(getting.dispatcher.overloads)))
    assert cp.asarray(probes.table.slots).data.ptr % 16 == 0
    assert "LDG.E.128.CONSTANT" in program
    assert not re.search(r"\bCALL\.", program)
