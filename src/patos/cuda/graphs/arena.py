"""Device memory a captured pass allocates its temporaries from."""

_ALIGNMENT = 512
_MIN_BLOCK_BYTES = 16 << 20


class Arena:
    """Device memory handed out bump by bump, which a captured pass allocates its temporaries from.

    Every capture rewinds it, so graphs captured one after another overlay the same addresses.
    That is safe because graphs replay in stream order, never together, and each writes what it
    reads. Memory is never returned inside a capture, and a pass that outgrows the blocks held
    adds one that fits it.
    """

    def __init__(self, arrays) -> None:
        """Hold no memory until the first capture asks for some.

        arrays: the array module owning device memory, normally `cupy`.
        """
        self.arrays = arrays
        self.blocks: list = []
        self.index = self.used = 0

    @property
    def nbytes(self) -> int:
        """The device memory the blocks hold."""
        return sum(block.size for block in self.blocks)

    def holds(self, address: int) -> bool:
        """Whether `address` lies in memory this arena owns."""
        return any(block.ptr <= address < block.ptr + block.size for block in self.blocks)

    def malloc(self, size: int):
        """A pointer to `size` bytes no other allocation of this capture overlaps."""
        size = -(-max(size, 1) // _ALIGNMENT) * _ALIGNMENT
        while self.index < len(self.blocks) and self.used + size > self.blocks[self.index].size:
            self.index, self.used = self.index + 1, 0
        if self.index == len(self.blocks):
            self.blocks.append(self.arrays.cuda.Memory(max(size, _MIN_BLOCK_BYTES)))
        pointer = self.arrays.cuda.MemoryPointer(self.blocks[self.index], self.used)
        self.used += size
        return pointer

    def rewind(self) -> None:
        """Start the next capture at the first block."""
        self.index = self.used = 0
