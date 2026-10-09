"""What turns a device's refusal into the pass's own."""

from typing import Self

from .unrecordable import Unrecordable


class Refusal:
    """A scope that raises `Unrecordable` for any of `errors` the device raises inside it."""

    def __init__(self, *errors: type[Exception]) -> None:
        self.errors = errors

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _kind, error, _traceback) -> None:
        if isinstance(error, self.errors):
            raise Unrecordable(str(error)) from error
