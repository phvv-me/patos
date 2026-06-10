# strflag

An enum `Flag` whose members carry a literal string, kept OR-combinable and iterable.

## When to reach for it

Reach for `strflag` when an enum is the vocabulary of some option set and each member also has a
literal token you emit. Command-line switches are the classic case, where `ALL` means `-a` and
`BIG` means `--big`. You declare the strings once, combine members with `|`, iterate the
combination, and read each member's token off `.string`. The enum stays the single source of
truth and the wire format rides along with it.

## The anti-pattern it replaces

It replaces hand-rolling a `Flag` subclass per CLI vocabulary, where you bolt a parallel dict or a
match statement onto the enum to map each member to its token. That second structure drifts from
the enum, forgets new members, and scatters the string next to the name it belongs to. `strflag`
folds the token into the member itself, so adding a switch is adding one line and the mapping can
never fall out of sync.

## Usage

```python
from patos import StrFlag


class Opt(StrFlag):
    ALL = "-a"
    LONG = "--long"
    HUMAN = "--human-readable"


combo = Opt.ALL | Opt.LONG
[member.string for member in combo]   # ["-a", "--long"]

Opt.ALL in combo                      # True
Opt.HUMAN in combo                    # False
```

Because each member takes the next power of two, members compose with `|` and the combination
iterates back into its parts, so you can feed the tokens straight into an argv list.

## Public API

- `StrFlag`. Base class. Subclass it and assign each member its literal string. Members are OR-combinable and iterable like any `enum.Flag`.
- `StrFlag.string`. The token to emit. A declared member yields its literal and a composite joins its decomposed members' literals with spaces.
- `StrFlag.literal`. The literal string declared for a single member.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/strflag.py"
```
