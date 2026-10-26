"""`items` and `items_through`, the loops over the items a kernel's launch gave it."""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from ..scalars import i64

if TYPE_CHECKING:
    from ..scalars import Operand


def items(count: Operand) -> Iterator[i64]:
    """The items of the kernel below `count`, as `for item in items(count)`.

    An item is the thread, warp or block the kernel's `per` names. A kernel that strides visits
    its own item and every one a grid's worth of items after it, any other kernel its own if it
    is below `count`, and then `return` leaves it where `break` and `continue` have nothing to
    leave. The loop is the `while` loop written by hand, so `count` is evaluated at each pass:
    `live[0]` is read again each time, a local only once. Only a kernel loops over items; a device
    function takes its item as a parameter.
    """
    raise TypeError("`items` is the iterable of a `for` loop in a kernel")


def items_through(bound: Operand) -> Iterator[i64]:
    """The items of the kernel up to and including `bound`, otherwise as `items` loops them."""
    raise TypeError("`items_through` is the iterable of a `for` loop in a kernel")
