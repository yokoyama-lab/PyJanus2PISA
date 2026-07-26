#!/usr/bin/env python3
"""Differential test of this compiler against the PyJanus interpreter.

Runs the same Janus program two ways and compares the final store:

  1. this repo:  source → PISA assembly → PISA interpreter
  2. PyJanus:    source → jana2014 → PyJanus's own interpreter

The two projects accept different *declaration* syntax but the same statement
syntax, so a source-to-source shim is needed (see `emit_jana2014`).  The
supported common subset is documented in README.md ("Cross-checking against
PyJanus").

Usage:
    python3 tools/pyjanus_crosscheck.py [--pyjanus DIR] [FILE.janus ...]

With no FILE, a set of built-in programs covering the common subset is run.
Exits non-zero if any program's final store differs between the two.
"""

import argparse
import io
import contextlib
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexer import tokenize
from parser import parse
from codegen import compile_program
from pisa_interp import PISAMachine
from syntax import (
    Const, Var, ArrayAccess, BinOp,
    Skip, AssignVar, AssignArr, Swap, Call, Uncall, If, From, Seq, Print,
)

DEFAULT_PYJANUS = os.path.expanduser("~/dev/github.com/yokoyama-lab/PyJanus")


# --- Source-to-source shim: our dialect → PyJanus jana2014 ---------------

def _expr(e) -> str:
    if isinstance(e, Const):
        return str(e.value)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, ArrayAccess):
        return f"{e.name}[{_expr(e.index)}]"
    if isinstance(e, BinOp):
        return f"({_expr(e.left)} {e.op} {_expr(e.right)})"
    raise NotImplementedError(f"expression: {e!r}")


def _stmt(s, ind: int, args: str) -> list:
    """Render a statement; `args` is the argument list for call/uncall."""
    p = " " * ind
    if isinstance(s, Skip):
        return [f"{p}skip"]
    if isinstance(s, AssignVar):
        return [f"{p}{s.var} {s.op} {_expr(s.expr)}"]
    if isinstance(s, AssignArr):
        return [f"{p}{s.var}[{_expr(s.idx)}] {s.op} {_expr(s.expr)}"]
    if isinstance(s, Swap):
        lhs = s.lhs if s.lhs_idx is None else f"{s.lhs}[{_expr(s.lhs_idx)}]"
        rhs = s.rhs if s.rhs_idx is None else f"{s.rhs}[{_expr(s.rhs_idx)}]"
        return [f"{p}{lhs} <=> {rhs}"]
    if isinstance(s, Call):
        return [f"{p}call {s.proc}({args})"]
    if isinstance(s, Uncall):
        return [f"{p}uncall {s.proc}({args})"]
    if isinstance(s, If):
        return ([f"{p}if {_expr(s.test)} then"] + _stmt(s.then_, ind + 4, args) +
                [f"{p}else"] + _stmt(s.else_, ind + 4, args) +
                [f"{p}fi {_expr(s.fi)}"])
    if isinstance(s, From):
        return ([f"{p}from {_expr(s.from_)} do"] + _stmt(s.do_, ind + 4, args) +
                [f"{p}loop"] + _stmt(s.loop_, ind + 4, args) +
                [f"{p}until {_expr(s.until)}"])
    if isinstance(s, Seq):
        out = []
        for sub in s.stmts:
            out.extend(_stmt(sub, ind, args))
        return out
    if isinstance(s, Print):
        raise NotImplementedError("print is not part of the common subset")
    raise NotImplementedError(f"statement: {s!r}")


def emit_jana2014(prog) -> str:
    """Translate a parsed program into PyJanus's jana2014 dialect.

    This repo uses global variable declarations and parameterless procedures;
    jana2014 has no globals, so every global is threaded through each
    procedure as a (call-by-reference) parameter, and `main` declares them
    locally.  Statement syntax is shared and is copied verbatim.
    """
    params = ", ".join(
        f"int {v.name}[]" if v.size > 1 else f"int {v.name}" for v in prog.vars)
    args = ", ".join(v.name for v in prog.vars)

    lines = []
    for proc in prog.procs:
        if proc.name == prog.main_proc:
            continue
        lines.append(f"procedure {proc.name}({params})")
        lines.extend(_stmt(proc.body, 4, args) or ["    skip"])
        lines.append("")

    main = next(p for p in prog.procs if p.name == prog.main_proc)
    lines.append("procedure main()")
    for v in prog.vars:
        lines.append(f"    int {v.name}[{v.size}]" if v.size > 1
                     else f"    int {v.name}")
    lines.extend(_stmt(main.body, 4, args) or ["    skip"])
    return "\n".join(lines) + "\n"


# --- The two execution paths --------------------------------------------

def run_ours(src: str) -> dict:
    """Compile to PISA, interpret, and return {var name: value or [values]}."""
    prog = parse(tokenize(src))
    m = PISAMachine(compile_program(prog))
    m.run()
    store, offset = {}, 0
    for v in prog.vars:
        if v.size > 1:
            store[v.name] = [m.mem.get(offset + i, 0) for i in range(v.size)]
        else:
            store[v.name] = m.mem.get(offset, 0)
        offset += v.size
    return store


def run_pyjanus(src: str, pyjanus_dir: str, scratch: str) -> dict:
    """Run the translated program under PyJanus and parse its printed store."""
    if pyjanus_dir not in sys.path:
        sys.path.insert(0, pyjanus_dir)
    from jana_py import cli

    path = os.path.join(scratch, "_crosscheck.ja")
    with open(path, "w") as f:
        f.write(emit_jana2014(parse(tokenize(src))))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = cli.main(["--std", "jana2014", "-s", path])
    if rc != 0:
        raise RuntimeError(f"PyJanus failed:\n{buf.getvalue()}")

    store = {}
    for line in buf.getvalue().splitlines():
        m = re.match(r"^(\w+)\[\d+\] = \{(.*)\}$", line)
        if m:
            store[m.group(1)] = [int(x) for x in m.group(2).split(",")]
            continue
        m = re.match(r"^(\w+) = (-?\d+)$", line)
        if m:
            store[m.group(1)] = int(m.group(2))
    return store


# --- Built-in programs covering the common subset ------------------------

PROGRAMS = {
    "arithmetic": "int x\nint y\nprocedure main\n  x += 3\n  y += x + 2\n  y -= 1\n  x ^= y",
    "swap": "int x\nint y\nprocedure main\n  x += 7\n  y += 2\n  x <=> y",
    "const-multiply": "int x\nint y\nprocedure main\n  y += 5\n  x += y * 3\n  x += y * 10",
    "conditional": ("int x\nint y\nprocedure main\n  x += 3\n"
                    "  if x = 3 then\n    y += 10\n  else\n    y += 20\n  fi y = 10"),
    "loop": ("int x\nint y\nprocedure main\n"
             "  from x = 0 do\n    x += 1\n  loop\n    y += x\n  until x = 5"),
    "array": ("int a[4]\nint i\nprocedure main\n  a[0] += 5\n  a[1] += 7\n"
              "  i += 2\n  a[i] += 9"),
    "procedure": ("int x\nint y\nprocedure bump\n  x += 1\n  y += x\n"
                  "procedure main\n  call bump\n  call bump\n  uncall bump"),
    "nested": ("int x\nint y\nint z\nprocedure step\n  z += x\n"
               "procedure main\n  x += 2\n"
               "  from y = 0 do\n    y += 1\n    call step\n  loop\n"
               "    if y = 1 then\n      x += 1\n    else\n      x -= 1\n    fi y = 1\n"
               "  until y = 3"),
}

# Divergences that are known compiler bugs rather than test-harness problems.
# Reported but not counted as failures, so the tool stays usable as a
# regression check for everything else.
KNOWN_ISSUES: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="Janus programs (default: built-ins)")
    ap.add_argument("--pyjanus", default=os.environ.get("PYJANUS_DIR", DEFAULT_PYJANUS),
                    help=f"PyJanus checkout (default: {DEFAULT_PYJANUS})")
    ap.add_argument("--scratch", default="/tmp", help="directory for temporary files")
    ap.add_argument("--show", action="store_true", help="print the translated source")
    args = ap.parse_args()

    if not os.path.isdir(args.pyjanus):
        print(f"PyJanus not found at {args.pyjanus}; pass --pyjanus DIR", file=sys.stderr)
        return 2

    cases = ({os.path.basename(f): open(f).read() for f in args.files}
             if args.files else PROGRAMS)

    failures = 0
    for name, src in cases.items():
        try:
            if args.show:
                print(emit_jana2014(parse(tokenize(src))))
            ours = run_ours(src)
            theirs = run_pyjanus(src, args.pyjanus, args.scratch)
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failures += 1
            continue
        if ours == theirs:
            print(f"OK    {name}: {ours}")
        else:
            diff = {k: (ours.get(k), theirs.get(k))
                    for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k)}
            if name in KNOWN_ISSUES:
                print(f"KNOWN {name}: (ours, pyjanus) = {diff}\n      {KNOWN_ISSUES[name]}")
            else:
                print(f"DIFF  {name}: (ours, pyjanus) = {diff}")
                failures += 1

    known = sum(1 for n in cases if n in KNOWN_ISSUES)
    print(f"\n{len(cases) - failures - known}/{len(cases)} programs agree"
          + (f" ({known} known divergence(s))" if known else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
