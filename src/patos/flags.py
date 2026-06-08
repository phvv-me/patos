from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

FlagValue: TypeAlias = bool | int | str | Sequence[str] | None


def flags(
    *,
    joined: bool = False,
    separator: str | None = None,
    **options: FlagValue,
) -> tuple[str, ...]:
    """Turn keyword options into a CLI argv tuple, dropping the empty ones.

    Keys become long flags with underscores hyphenated (`mem_gb` -> `--mem-gb`).
    Rendering by value type:

    - `None`, `False`, `""`: dropped entirely.
    - `True`: a bare `--flag`.
    - `int` / `str`: `--flag value` (or `--flag=value` when `joined`).
    - sequence: by default one `--flag value` pair per item; with `separator`
      a single `--flag a,b,c` joined on `separator`. Empty sequences drop.

    joined: emit `--flag=value` single tokens instead of `--flag` then `value`.
    separator: join sequence values into one flag instead of repeating it.
    options: the flag name to value mapping.
    """
    argv: list[str] = []
    for key, value in options.items():
        flag = f"--{key.replace('_', '-')}"
        if value is None or value is False or value == "":
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, str | int):
            argv.extend(emit(flag, str(value), joined=joined))
        elif separator is not None:
            argv.extend(emit(flag, separator.join(value), joined=joined))
        else:
            for item in value:
                argv.extend(emit(flag, item, joined=joined))
    return tuple(argv)


def emit(flag: str, value: str, *, joined: bool) -> tuple[str, ...]:
    """Render one flag/value pair as `("--flag=value",)` or `("--flag", "value")`."""
    return (f"{flag}={value}",) if joined else (flag, value)
