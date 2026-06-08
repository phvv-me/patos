from __future__ import annotations

from enum import Flag


class StrFlag(Flag):
    """A ``Flag`` whose members carry a literal string, kept OR-combinable and iterable.

    Each member gets the next power-of-two value (so members compose with ``|`` and iterate)
    while the declared string lives on ``.string``. Handy for building command-line option
    sets where the enum is the vocabulary and ``.string`` is what you emit.

    ```python
    class Opt(StrFlag):
        ALL = "-a"
        BIG = "--big"

    [m.string for m in (Opt.ALL | Opt.BIG)]  # ['-a', '--big']
    ```
    """

    string: str

    def __new__(cls, string: str) -> StrFlag:
        member = object.__new__(cls)
        member._value_ = 1 << len(cls.__members__)
        member.string = string
        return member
