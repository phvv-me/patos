"""Scalar operations device code repeats."""

from ..typed import i32, ptx, u32


@ptx("min.u32 $result, $best, $value;", pure=True)
def min_nonnegative(best: u32, value: i32) -> i32:
    """The lower of `best` and `value`, -1 standing for none.

    -1 reads as the largest u32, so the unsigned minimum is the lower non-negative value. `best`
    is the running one, which the instruction reads unsigned.
    """
    ...
