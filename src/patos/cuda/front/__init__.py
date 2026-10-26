"""The front of `patos.cuda.typed`: a device function or kernel read the way C reads declarations,
checked, and rebuilt so its annotations convert what they declare, and a `Record` declared and
validated like a model.

It imports no Numba, so it reads and checks device code on any host. `patos.cuda.backend` compiles
what it hands over.
"""
