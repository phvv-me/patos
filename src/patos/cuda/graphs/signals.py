"""A point of a recording that the host can wait for while the rest of it runs."""

from cuda.bindings import runtime as cudart
from cuda.core._utils import cuda_utils


def signal(arrays, event) -> None:
    """Record `event` on the current stream; in a recording, every replay signals it at this point.

    An event recorded into a graph the ordinary way cannot be waited for by the host after a
    replay. This one is an external event node, which can be, while the nodes after it still run.
    Outside a recording it is the plain record.

    arrays: the array module owning device memory, normally `cupy`.
    event: one of its events.
    """
    stream = arrays.cuda.get_current_stream()
    if not stream.is_capturing():
        event.record(stream)
        return
    cuda_utils.handle_return(
        cudart.cudaEventRecordWithFlags(event.ptr, stream.ptr, cudart.cudaEventRecordExternal)
    )
