from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pato:
    """A single pattern in the flock, copyable as one self-contained module.

    module: file name inside `patos/` (without directory), e.g. `registry.py`.
    summary: one-line description shown by `patos list`.
    exports: public symbols the module defines and the package re-exports.
    extras: optional dependency groups this pato needs (empty for zero-dep patos).
    doc: documentation page path under `docs/`, e.g. `patos/registry.md`.
    """

    module: str
    summary: str
    exports: tuple[str, ...]
    extras: tuple[str, ...] = ()
    doc: str = ""


PATOS: dict[str, Pato] = {
    "registry": Pato(
        module="registry.py",
        summary="Self-registering class hierarchies: subclasses enroll via __init_subclass__.",
        exports=("Registry",),
        doc="patos/registry.md",
    ),
    "singleton": Pato(
        module="singleton.py",
        summary="One instance per class, __init__ runs once.",
        exports=("Singleton",),
        doc="patos/singleton.md",
    ),
    "flyweight": Pato(
        module="flyweight.py",
        summary="Intern instances by constructor args (build-once, immutable).",
        exports=("Flyweight", "FlyweightMeta"),
        doc="patos/flyweight.md",
    ),
    "strategy": Pato(
        module="strategy.py",
        summary="A named family of interchangeable implementations, selected at runtime.",
        exports=("Strategy",),
        doc="patos/strategy.md",
    ),
    "dispatch": Pato(
        module="dispatch.py",
        summary="Dispatch on a value the way singledispatch dispatches on a type.",
        exports=("value_dispatch",),
        doc="patos/dispatch.md",
    ),
    "flags": Pato(
        module="flags.py",
        summary="Turn keyword options into a CLI argv tuple.",
        exports=("flags",),
        doc="patos/flags.md",
    ),
    "strflag": Pato(
        module="strflag.py",
        summary="An enum Flag whose members carry a literal string, OR-combinable and iterable.",
        exports=("StrFlag",),
        doc="patos/strflag.md",
    ),
}
