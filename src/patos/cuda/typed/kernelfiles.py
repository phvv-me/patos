"""The cache numba-cuda keeps kernels in, placed in the folder of the sources they are made of
(`kernelfolder`)."""

from numba import types
from numba.cuda.dispatcher import CUDACache, CUDACacheImpl

from .kernelfolder import Locator
from .records import RecordType


class _Impl(CUDACacheImpl):
    _locator_classes = [Locator]

    def check_cachable(self, kernel) -> bool:
        """Whether another process can load `kernel`.

        It cannot when a record class it takes is made in a function, which no process can find
        again, or when it refers to a global device array, which a cache cannot hold.
        """
        held = kernel.library.referenced_objects.values()
        arrays = any(getattr(each, "__cuda_array_interface__", None) for each in held)
        return not (arrays or any(map(self._is_local, kernel.argument_types)))

    @staticmethod
    def _is_local(kind: types.Type) -> bool:
        """Whether `kind` is a record made in a function, or holds one."""
        if not isinstance(kind, RecordType):
            return False
        return "<locals>" in kind.cls.__qualname__ or any(
            _Impl._is_local(held) for _, held in kind.fields
        )


Cache = type("Cache", (CUDACache,), {"_impl_class": _Impl})
