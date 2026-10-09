"""One recorded pass."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core.graph import Graph


class Captured[Result]:
    """One recorded pass: the graph to replay and what the pass returned while it was recorded.

    result: the pass's return value, whose arrays are the arena memory every replay rewrites.
    """

    def __init__(self, graph: Graph, result: Result) -> None:
        self.graph = graph
        self.result = result
