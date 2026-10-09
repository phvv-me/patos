"""Binary search over a device array sorted ascending, `Range` of it at a time:
`search.lower_bound(values, search.Range(low, high), key)`."""

import operator
from collections.abc import Callable
from types import FunctionType
from typing import NamedTuple

from ..typed import device, i64, number


class Range(NamedTuple):
    """The indices `[low, high)` of the values a search looks in."""

    low: i64
    high: i64


def _search(name: str, before: Callable[..., bool], doc: str) -> FunctionType:
    """The search whose answer is the first index of the range holding no value `before` its key.

    name: the name the function answers to.
    before: whether a value in the range lies before the answer, given the value and the key.
    doc: what it finds.
    """

    def search[Key](values: number[int], within: Range, key: Key) -> i64:
        low, high = within
        while low < high:
            middle = low + ((high - low) >> 1)
            if before(values[middle], key):
                low = middle + 1
            else:
                high = middle
        return low

    search.__name__ = search.__qualname__ = name
    search.__doc__ = doc
    return device(search)


lower_bound = _search(
    "lower_bound", operator.lt,
    "The first index in `within` whose value is not below `key`, its end when none is.",
)  # fmt: skip
upper_bound = _search(
    "upper_bound", operator.le,
    "The first index in `within` whose value is above `key`, its end when none is.",
)  # fmt: skip
