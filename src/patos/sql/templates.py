from string.templatelib import Interpolation, Template

from sqlalchemy import ColumnElement, func, literal


def concat(template: Template) -> ColumnElement[str]:
    """Concatenate a t-string's literals and SQL expressions in their source order."""
    parts = [part.value if isinstance(part, Interpolation) else part for part in template]
    return func.concat(*parts)


def fragment(template: Template) -> ColumnElement[str]:
    """Return a t-string fragment whose `NULL` interpolation erases the whole piece."""
    parts = [part.value if isinstance(part, Interpolation) else part for part in template]
    joined = literal(parts[0]) if isinstance(parts[0], str) else parts[0]
    for part in parts[1:]:
        joined = joined.concat(part)
    return func.coalesce(joined, "")
