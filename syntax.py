"""Janus AST node definitions."""

from dataclasses import dataclass, field
from typing import List, Optional, Union


# --- Expressions ---

class Expr:
    """Base class for expressions."""
    pass


@dataclass
class Const(Expr):
    value: int


@dataclass
class Var(Expr):
    name: str


@dataclass
class ArrayAccess(Expr):
    name: str
    index: Expr


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


# --- Statements ---

class Stmt:
    """Base class for statements."""
    pass


@dataclass
class Skip(Stmt):
    pass


@dataclass
class AssignVar(Stmt):
    var: str
    op: str       # '+=', '-=', '^='
    expr: Expr


@dataclass
class AssignArr(Stmt):
    var: str
    idx: Expr
    op: str       # '+=', '-=', '^='
    expr: Expr


@dataclass
class Swap(Stmt):
    lhs: str      # variable or array name
    lhs_idx: Optional[Expr]  # None for scalar
    rhs: str
    rhs_idx: Optional[Expr]


@dataclass
class Call(Stmt):
    proc: str


@dataclass
class Uncall(Stmt):
    proc: str


@dataclass
class If(Stmt):
    test: Expr
    then_: Stmt
    else_: Stmt
    fi: Expr


@dataclass
class From(Stmt):
    from_: Expr
    do_: Stmt
    loop_: Stmt
    until: Expr


@dataclass
class Seq(Stmt):
    stmts: List[Stmt]


@dataclass
class Print(Stmt):
    """Print statement (for debugging, not in paper but useful)."""
    var: str


# --- Declarations ---

@dataclass
class VarDecl:
    name: str
    size: int = 1          # 1 for scalar, >1 for array
    init: List[int] = field(default_factory=lambda: [0])


@dataclass
class ProcDecl:
    name: str
    body: Stmt


@dataclass
class Program:
    vars: List[VarDecl]
    procs: List[ProcDecl]
    main_proc: str = "main"
