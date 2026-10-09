"""Binary search over a device array sorted ascending, `Range` of it at a time:
`search.lower_bound(values, search.Range(low, high), key)`."""

from typing import NamedTuple

from ..typed import Vector, device, i64, number


class Range(NamedTuple):
    """The indices `[low, high)` of the values a search looks in."""

    low: i64
    high: i64


@device
def lower_bound[Key: float](values: Vector[number], within: Range, key: Key) -> i64:
    """The first index in `within` whose value is not below `key`, its end when none is."""
    low, high = within
    while low < high:
        middle = low + ((high - low) >> 1)
        if values[middle] < key:
            low = middle + 1
        else:
            high = middle
    return low


@device
def find[Key: float](values: Vector[number], within: Range, key: Key) -> i64:
    """The first index in `within` whose value is `key`, or -1 when none is."""
    found = lower_bound(values, within, key)
    return found if found < within.high and values[found] == key else -1


@device
def upper_bound[Key: float](values: Vector[number], within: Range, key: Key) -> i64:
    """The first index in `within` whose value is above `key`, its end when none is."""
    low, high = within
    while low < high:
        middle = low + ((high - low) >> 1)
        if values[middle] <= key:
            low = middle + 1
        else:
            high = middle
    return low
