"""The refusal of a pass that cannot be recorded."""


class Unrecordable(Exception):
    """A pass that cannot be recorded: it reads the device back, or the device refused a node."""
