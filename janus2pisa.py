#!/usr/bin/env python3
"""janus2pisa: Compiler from Janus to PISA assembly.

Based on "Clean Translation of an Imperative Reversible Programming Language"
(Axelsen, CC 2011).

Usage:
    python janus2pisa.py <input.janus> [-o output.pisa]
    python janus2pisa.py --ast <input.janus>     # dump AST only
"""

import sys
import argparse

from lexer import tokenize
from parser import parse
from codegen import compile_program
from pisa import print_program
from inverse import invert_program


def main():
    ap = argparse.ArgumentParser(
        description="Compile Janus programs to PISA assembly"
    )
    ap.add_argument("input", help="Janus source file")
    ap.add_argument("-o", "--output", help="Output PISA file (default: stdout)")
    ap.add_argument("--ast", action="store_true", help="Print AST and exit")
    ap.add_argument("--tokens", action="store_true", help="Print tokens and exit")
    ap.add_argument("--inverse", action="store_true",
                    help="Compile the syntactic inverse P⁻¹ instead of P")
    args = ap.parse_args()

    with open(args.input) as f:
        source = f.read()

    # Tokenize
    tokens = tokenize(source)
    if args.tokens:
        for tok in tokens:
            print(tok)
        return

    # Parse
    program = parse(tokens)
    if args.ast:
        print(f"Variables: {program.vars}")
        print(f"Procedures: {[p.name for p in program.procs]}")
        print(f"Main: {program.main_proc}")
        for proc in program.procs:
            print(f"\nprocedure {proc.name}:")
            _print_stmt(proc.body, indent=2)
        return

    # Optionally invert
    if args.inverse:
        program = invert_program(program)

    # Compile
    pisa_code = compile_program(program)
    output = print_program(pisa_code)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


def _print_stmt(stmt, indent=0):
    """Pretty-print AST statement for --ast mode."""
    from syntax import (Skip, AssignVar, AssignArr, Swap, Call, Uncall,
                        If, From, Seq, Print)
    prefix = " " * indent
    if isinstance(stmt, Skip):
        print(f"{prefix}skip")
    elif isinstance(stmt, AssignVar):
        print(f"{prefix}{stmt.var} {stmt.op} {_fmt_expr(stmt.expr)}")
    elif isinstance(stmt, AssignArr):
        print(f"{prefix}{stmt.var}[{_fmt_expr(stmt.idx)}] {stmt.op} {_fmt_expr(stmt.expr)}")
    elif isinstance(stmt, Swap):
        lhs = stmt.lhs if stmt.lhs_idx is None else f"{stmt.lhs}[{_fmt_expr(stmt.lhs_idx)}]"
        rhs = stmt.rhs if stmt.rhs_idx is None else f"{stmt.rhs}[{_fmt_expr(stmt.rhs_idx)}]"
        print(f"{prefix}{lhs} <=> {rhs}")
    elif isinstance(stmt, Call):
        print(f"{prefix}call {stmt.proc}")
    elif isinstance(stmt, Uncall):
        print(f"{prefix}uncall {stmt.proc}")
    elif isinstance(stmt, If):
        print(f"{prefix}if {_fmt_expr(stmt.test)} then")
        _print_stmt(stmt.then_, indent + 2)
        print(f"{prefix}else")
        _print_stmt(stmt.else_, indent + 2)
        print(f"{prefix}fi {_fmt_expr(stmt.fi)}")
    elif isinstance(stmt, From):
        print(f"{prefix}from {_fmt_expr(stmt.from_)}")
        print(f"{prefix}do")
        _print_stmt(stmt.do_, indent + 2)
        print(f"{prefix}loop")
        _print_stmt(stmt.loop_, indent + 2)
        print(f"{prefix}until {_fmt_expr(stmt.until)}")
    elif isinstance(stmt, Seq):
        for s in stmt.stmts:
            _print_stmt(s, indent)
    elif isinstance(stmt, Print):
        print(f"{prefix}print({stmt.var})")


def _fmt_expr(expr):
    """Format expression for display."""
    from syntax import Const, Var, ArrayAccess, BinOp
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, ArrayAccess):
        return f"{expr.name}[{_fmt_expr(expr.index)}]"
    if isinstance(expr, BinOp):
        return f"({_fmt_expr(expr.left)} {expr.op} {_fmt_expr(expr.right)})"
    return repr(expr)


if __name__ == "__main__":
    main()
