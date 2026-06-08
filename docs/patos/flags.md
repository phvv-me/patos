# flags

Turn keyword options into a clean CLI argv tuple.

## When to reach for it

Reach for `flags` when you build command lines for a subprocess and you want the option mapping to
read like Python keyword arguments instead of fragile string concatenation. Booleans become bare
flags, values become `--key value` pairs, and underscores become dashes, so you get a tidy argv
tuple ready to hand to `subprocess.run`.

## The anti-pattern it replaces

It replaces the hand-built list of f-strings, where a forgotten space or a stray quote becomes a
shell bug, and where a `False` option still leaks onto the command line. `flags` makes the mapping
declarative, drops falsy switches, and keeps each token separate so nothing needs shell quoting.

## Usage

```python
import subprocess
from patos import flags


flags(verbose=True, output="out.bin", jobs=4, dry_run=False)
# ("--verbose", "--output", "out.bin", "--jobs", "4")

subprocess.run(["ffmpeg", *flags(i="in.mov", crf=23, preset="slow")])
```

A `True` value yields a bare flag, a falsy value is dropped, and any other value becomes a
`--key value` pair with underscores rewritten to dashes.

## Public API

- `flags(**options)`. Return a `tuple[str, ...]` argv fragment built from the keyword options.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/flags.py"
```
