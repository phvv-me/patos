from enum import StrEnum, auto
from typing import assert_type

from patos import type_dispatch, value_dispatch


class Command(StrEnum):
    select = auto()
    delete = auto()


@value_dispatch
def validate(left: int, right: str, *, kind: Command | None = None) -> bool:
    return False


@validate.register(Command.select)
@validate.register(Command.delete)
def validate_read(left: int, right: str) -> bool:
    return bool(left and right)


@validate.register()
def custom(left: int, right: str) -> bool:
    return bool(left or right)


class Marker:
    pass


@validate.register(Marker)
def validate_marker(left: int, right: str) -> bool:
    return bool(left or right)


@type_dispatch
def render(value: object) -> str:
    return repr(value)


@render.register()
def render_int(value: int) -> str:
    return str(value)


assert_type(validate(1, "x", kind=Command.select), bool)
assert_type(validate_read(1, "x"), bool)
assert_type(custom(1, "x"), bool)
assert_type(validate_marker(1, "x"), bool)
assert_type(render(1), str)
assert_type(render_int(1), str)
