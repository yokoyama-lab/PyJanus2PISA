"""Janus program inversion.

Implements the syntactic inverse of a Janus program.  For a Janus program P
that computes a bijection f: State → State, the inverse P⁻¹ computes f⁻¹.

Inversion rules (Yokoyama & Glück, PEPM 2007):

    Skip⁻¹                            = Skip
    (x op= e)⁻¹                       = x inv(op)= e
    (x[i] op= e)⁻¹                    = x[i] inv(op)= e
    (x₁ <=> x₂)⁻¹                    = x₁ <=> x₂
    (call f)⁻¹                        = uncall f
    (uncall f)⁻¹                      = call f
    (if e₁ then s₁ else s₂ fi e₂)⁻¹  = if e₂ then s₁⁻¹ else s₂⁻¹ fi e₁
    (from e₁ do s₁ loop s₂ until e₂)⁻¹
                                       = from e₂ do s₂⁻¹ loop s₁⁻¹ until e₁
    (s₁ ; s₂ ; … ; sn)⁻¹             = sn⁻¹ ; … ; s₂⁻¹ ; s₁⁻¹

    where   inv(+=) = -=,   inv(-=) = +=,   inv(^=) = ^=  (self-inverse)

The program inverse:
  - Each procedure p with body s becomes procedure p with body s⁻¹.
  - Variable declarations are unchanged.
  - The main procedure name is unchanged.
"""

from copy import deepcopy
from typing import Dict

from syntax import (
    Stmt, Skip, AssignVar, AssignArr, Swap, Call, Uncall,
    If, From, Seq, Print,
    ProcDecl, Program,
)


def _inv_op(op: str) -> str:
    """Return the inverse assignment operator."""
    if op == '+=':
        return '-='
    if op == '-=':
        return '+='
    if op == '^=':
        return '^='   # XOR is self-inverse
    raise ValueError(f"Unknown assignment operator: {op!r}")


def invert_stmt(stmt: Stmt) -> Stmt:
    """Return the syntactic inverse of a statement."""
    if isinstance(stmt, Skip):
        return Skip()

    if isinstance(stmt, AssignVar):
        return AssignVar(stmt.var, _inv_op(stmt.op), deepcopy(stmt.expr))

    if isinstance(stmt, AssignArr):
        return AssignArr(
            stmt.var,
            deepcopy(stmt.idx),
            _inv_op(stmt.op),
            deepcopy(stmt.expr),
        )

    if isinstance(stmt, Swap):
        # Swap is self-inverse
        return Swap(stmt.lhs, deepcopy(stmt.lhs_idx),
                    stmt.rhs, deepcopy(stmt.rhs_idx))

    if isinstance(stmt, Call):
        return Uncall(stmt.proc)

    if isinstance(stmt, Uncall):
        return Call(stmt.proc)

    if isinstance(stmt, If):
        # if e1 then s1 else s2 fi e2  →  if e2 then s1⁻¹ else s2⁻¹ fi e1
        return If(
            test=deepcopy(stmt.fi),
            then_=invert_stmt(stmt.then_),
            else_=invert_stmt(stmt.else_),
            fi=deepcopy(stmt.test),
        )

    if isinstance(stmt, From):
        # from e1 do s1 loop s2 until e2
        # →  from e2 do s2⁻¹ loop s1⁻¹ until e1
        return From(
            from_=deepcopy(stmt.until),
            do_=invert_stmt(stmt.loop_),
            loop_=invert_stmt(stmt.do_),
            until=deepcopy(stmt.from_),
        )

    if isinstance(stmt, Seq):
        # Reverse order and invert each statement
        return Seq([invert_stmt(s) for s in reversed(stmt.stmts)])

    if isinstance(stmt, Print):
        # Print has no state effect; keep it (or drop it in inverse)
        return Print(stmt.var)

    raise TypeError(f"Unknown statement type: {type(stmt).__name__}")


def invert_program(prog: Program) -> Program:
    """Return the syntactic inverse of a Janus program.

    Each procedure body is inverted; variable declarations are unchanged.
    """
    inv_procs = [ProcDecl(p.name, invert_stmt(p.body)) for p in prog.procs]
    return Program(
        vars=deepcopy(prog.vars),
        procs=inv_procs,
        main_proc=prog.main_proc,
    )
