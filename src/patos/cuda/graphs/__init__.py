"""Stream work captured once and replayed: a pass recorded as a graph, with its own memory.

A pass written as ordinary stream code (kernel launches, array module calls) is recorded by
running it once under capture, and the recording replays for the cost of one launch. Capture only
records, so the pass may not read the device back, and every address it uses is baked into the
recording: the temporaries it allocates come from an `Arena` the recording owns, never from the
array module's pool, which would hand the same addresses to other arrays once the pass returned.
A branch on a device value is `branch`: a read and an `if` in a pass run as usual, a conditional
node in a recording. `signal` records an event the host can wait for part way through a replay.
"""

from .arena import Arena
from .captured import Captured
from .recording import branch
from .signals import signal
from .store import Graphs
from .unrecordable import Unrecordable

__all__ = ["Arena", "Captured", "Graphs", "Unrecordable", "branch", "signal"]
