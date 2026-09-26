"""Replay the Rocq model's *legacy* ORX / ANDX on the Pendulum-semantics interpreter.

The Rocq development (rocq/PISA.v, `IOrx` / `IAndx`) still models the
instructions this project used before 2026-09:

    ORX  d s       d |= s ; s := 0            (not injective)
    ANDX d1 d2 s   d1 ^= d2 & s ; d2 := 0     (not injective)

pisa.py / pisa_interp.py now implement Pendulum's 3-operand forms
(`ORX d s t: d ^= s | t`, `ANDX d s t: d ^= s & t`).  Until the Rocq model is
updated, the cross-check tools replay the verified compiler's code by
expanding each legacy instruction into an exactly equivalent sequence for
the new interpreter (the same expansion tools/pisa2pal.py used for phpisa):

    ORX d s        ->  ANDX d d s ; XOR d s ; XOR s s     ((d & ~s) ^ s = d | s)
    ANDX d1 d2 s   ->  ANDX d1 d2 s ; XOR d2 d2

The expansions contain `XOR r r` and `ANDX d d s`, which are not locally
invertible (pisa.is_wf) — exactly like the legacy instructions they emulate.
They are only ever *executed* here; codegen.py never emits them.
"""

from pisa import ANDX, XOR


def legacy_orx(d: str, s: str) -> list:
    return [ANDX(d, d, s), XOR(d, s), XOR(s, s)]


def legacy_andx(d1: str, d2: str, s: str) -> list:
    return [ANDX(d1, d2, s), XOR(d2, d2)]
