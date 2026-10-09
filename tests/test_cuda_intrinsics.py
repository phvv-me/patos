"""What the PTX stubs may let the compiler move: a template is pure only when every instruction
of it is register arithmetic."""

import pytest

from patos.cuda.typed import intrinsics

# An instruction that writes the carry flag, or reads it, is ordered against the others.
_CARRIES = ["add.cc.u32", "sub.cc.u32", "mad.lo.cc.u32", "addc.u32", "subc.u32", "madc.hi.u32"]


@pytest.mark.parametrize("opcode", _CARRIES)
def test_an_instruction_that_writes_or_reads_the_carry_is_an_effect(opcode: str) -> None:
    """A pure `add.cc` the compiler drops leaves the `addc` after it reading a carry never made."""
    after = f"{{\n.reg .u32 t;\nmov.u32 t, $a;\n@p {opcode} $result, t;\n}}"

    assert not intrinsics.is_pure(f"{opcode} $result, $a;")
    assert not intrinsics.is_pure(after)


def test_a_directive_ending_at_its_line_hides_no_instruction_behind_it() -> None:
    """`.loc` has no semicolon, so the instruction after it is parsed on its own."""
    assert not intrinsics.is_pure(".loc 1 1 0\nld.global.u32 $result, [$at];")
    assert not intrinsics.is_pure("mov.u32 t, 1;\n.loc 1 2 0\nst.global.u32 [$at], t;")
    assert intrinsics.is_pure(
        "{\n.reg .u32 t;\n.loc 1 2 0\nmov.u32 t, $a;\nadd.u32 $result, t, 1;\n}"
    )
    assert intrinsics.is_pure("// ld.global.u32 $result, [$at];\nadd.u32 $result, $a, 1;")
