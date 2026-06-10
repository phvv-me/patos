from __future__ import annotations

from enum import Flag


class StrFlag(Flag):
    """A ``Flag`` whose members carry a literal string, kept OR-combinable and iterable.

    Each declared member gets the next power-of-two value (so members compose with ``|``
    and iterate) while its literal string lives on ``literal``. The ``string`` property
    works on every member, composites and the empty flag included, by joining the
    decomposed members' literals. Handy for building command-line option sets where the
    enum is the vocabulary and ``string`` is what you emit.

    ```python
    class Opt(StrFlag):
        ALL = "-a"
        BIG = "--big"

    Opt.ALL.string             # '-a'
    (Opt.ALL | Opt.BIG).string  # '-a --big'
    ```
    """

    literal: str

    def __new__(cls, string: str) -> StrFlag:
        member = object.__new__(cls)
        member._value_ = 1 << len(cls.__members__)
        member.literal = string
        return member

    @property
    def string(self) -> str:
        """The member's literal, or the decomposed members' literals joined with spaces.

        Composite and empty members are created by ``Flag`` internals that bypass the
        custom ``__new__``, so their string is derived here from the canonical members.
        """
        return " ".join(member.literal for member in self)
