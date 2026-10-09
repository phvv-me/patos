"""Single-instruction bit and byte operations numba-cuda lacks, each one PTX instruction.

Call them through the module (`bits.permute(low, high, selector)`). `cuda.popc`, `clz`, `ffs`,
`brev` and `byte_perm` are numba-cuda's own.
"""

from ..typed import ptx, u32


@ptx("prmt.b32 $result, $low, $high, $selector;", pure=True)
def permute(low: u32, high: u32, selector: u32) -> u32:
    """The four bytes `selector` picks out of the eight of `high:low`.

    Nibble `i` of `selector` fills byte `i` of the result: its low three bits number a source byte
    (0 to 3 in `low`, 4 to 7 in `high`) and its top bit, when set, fills the byte with that
    byte's sign bit instead.
    """
    raise NotImplementedError


@ptx("shf.r.wrap.b32 $result, $low, $high, $shift;", pure=True)
def funnel(low: u32, high: u32, shift: u32) -> u32:
    """The low 32 bits of `high:low` shifted right by `shift` modulo 32."""
    raise NotImplementedError


@ptx("dp4a.u32.u32 $result, $a, $b, $c;", pure=True)
def dot4(a: u32, b: u32, c: u32) -> u32:
    """The sum of the four products of `a`'s and `b`'s bytes, read unsigned, plus `c`."""
    raise NotImplementedError


@ptx(
    "{\n.reg .u32 none;\nmov.u32 none, 0;\nvabsdiff4.u32.u32.u32 $result, $a, $b, none;\n}",
    pure=True,
)
def absdiff4(a: u32, b: u32) -> u32:
    """The absolute difference of each pair of bytes of `a` and `b`, read unsigned.

    One instruction from sm_86 to sm_100 and three on sm_121.
    """
    raise NotImplementedError
