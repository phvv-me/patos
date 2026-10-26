"""`Struct`: a record (`patos.cuda.front.record`) as the device sees it, one kernel argument of its
own device type, built by the host or by device code."""

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from ..front.declarations import Declared
from ..front.record import Record, Value
from ..scalars import ArrayOf, ConstantOf
from .arguments import Argument, Marshalled, argument, record_argument
from .decorators import bind_members
from .kernels import Kernel
from .records import register_constructor


class Struct(Record):
    """A frozen record of device values, declared like a model and passed to a kernel as one
    argument.

    A subclass annotates its fields as a `Record` does, and building one on the host validates
    them, a host array uploading. Numba sees a record as a C struct of its fields, read by
    attribute in device code, and device code builds one by calling its class, each scalar field
    converting to its declaration.

    Device functions defined in the class with an unannotated `self` are its device members: a
    method, an operator when the name is a dunder (`__getitem__`, `__len__`, `__contains__`), or
    an attribute under `@property`; the host class keeps none of them. A `kernel` defined so is
    launched from a record, `record.kernel[items](*arguments, stream=stream)`.
    """

    __slots__ = ("_argument",)
    # Declared for the checkers only: an annotation in the body would be a field.
    if TYPE_CHECKING:
        _argument: Argument | None

    @classmethod
    def _received(cls, kind: Declared, value: Value) -> Value:
        """`value` as `Record` receives it, a host array uploaded to the device."""
        held = super()._received(kind, value)
        return cp.asarray(held) if isinstance(held, np.ndarray) else held

    def __init__(self, *values: Value, **named: Value) -> None:
        super().__init__(*values, **named)
        object.__setattr__(self, "_argument", None)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        bind_members(cls)
        for name, member in list(vars(cls).items()):
            if isinstance(member, Kernel) and member.member:
                member.bind(cls, name)
        register_constructor(cls)

    def __kernel_argument__(self, stream: int) -> Argument:
        """What a kernel receives for this record, marshalled on its first launch.

        stream: the handle of that launch's stream, which waits once for the producer each array
            field's interface names.
        """
        held = self._argument
        if held is None:
            declarations = self.declarations()
            constants = tuple(
                (name, getattr(self, name))
                for name, kind in declarations.items()
                if isinstance(kind, ConstantOf)
            )
            fields = {
                name: self._marshalled(name, kind, getattr(self, name), stream)
                for name, kind in declarations.items()
                if not isinstance(kind, ConstantOf)
            }
            held = record_argument(type(self), fields, constants)
            object.__setattr__(self, "_argument", held)
        return held

    def _marshalled(self, name: str, kind: Declared, value: Marshalled, stream: int) -> Argument:
        """The argument field `name` marshals to; an array field must be contiguous."""
        marshalled = argument(value, stream)
        if isinstance(kind, ArrayOf) and marshalled[0].layout != "C":
            raise TypeError(f"{type(self).__name__}.{name} is strided, not contiguous")
        return marshalled
