#!/usr/bin/env python3
"""Differential test of the Python compiler/interpreter against the verified ones.

`rocq/Compile.v` proves a compiler correct, but that compiler is a
*re-implementation* of the scheme `codegen.py` uses — the proof says nothing
about the Python code unless the two are actually compared.  This tool closes
that gap for the straight-line fragment.

`rocq/driver` (the extracted verified compiler, run under the verified machine
model `PISA.run`) emits, per program: the compiled instructions, the final
variable values, and the final scratch registers.  For each program we check

  1. **the Python interpreter** — replay the *verified compiler's* instructions
     on `pisa_interp.PISAMachine` and compare the final store.  A mismatch means
     `pisa_interp.py` disagrees with the formal PISA semantics.
  2. **the Python compiler** — compile the same Janus source with `codegen.py`,
     run it on the same interpreter, and compare the final store.  A mismatch
     means `codegen.py` disagrees with the verified translation.
  3. **cleanliness** — both must leave every scratch register at 0, which is
     `clean_above` in the proof.

Usage:
    make -C rocq -f Makefile.driver     # build rocq/driver first
    python3 tools/rocq_diff.py
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexer import tokenize
from parser import parse
from codegen import compile_program
from pisa_interp import PISAMachine
from pisa import ADD, SUB, XOR, ADDI, SUBI, XORI, NEG, EXCH, LabeledInstr

DRIVER = os.path.join(os.path.dirname(__file__), "..", "rocq", "driver")

# The driver prints register numbers; pisa.py names registers "r<n>".
_BUILD = {
    "ADD":  lambda a: ADD(f"r{a[0]}", f"r{a[1]}"),
    "SUB":  lambda a: SUB(f"r{a[0]}", f"r{a[1]}"),
    "XOR":  lambda a: XOR(f"r{a[0]}", f"r{a[1]}"),
    "ADDI": lambda a: ADDI(f"r{a[0]}", a[1]),
    "SUBI": lambda a: SUBI(f"r{a[0]}", a[1]),
    "XORI": lambda a: XORI(f"r{a[0]}", a[1]),
    "NEG":  lambda a: NEG(f"r{a[0]}"),
    "EXCH": lambda a: EXCH(f"r{a[0]}", f"r{a[1]}"),
}


def parse_driver_output(text: str) -> list:
    """Split the driver's output into one dict per program."""
    cases, cur = [], None
    for line in text.splitlines():
        head, _, rest = line.partition(" ")
        if head == "CASE":
            cur = {"name": rest, "instrs": [], "vars": {}, "regs": {}}
            cases.append(cur)
        elif head == "SOURCE":
            cur["source"] = rest.replace("\\n", "\n")
        elif head == "NVARS":
            cur["nvars"] = int(rest)
        elif head == "I":
            op, *args = rest.split()
            cur["instrs"].append(_BUILD[op]([int(a) for a in args]))
        elif head == "VAR":
            k, v = rest.split()
            cur["vars"][int(k)] = int(v)
        elif head == "REG":
            k, v = rest.split()
            cur["regs"][int(k)] = int(v)
    return cases


def run_bare(instrs, nvars: int) -> tuple:
    """Run a bare instruction list (no DATA/START/FINISH) on the Python machine."""
    from pisa import START, FINISH
    wrapped = ([LabeledInstr("start", START())]
               + [LabeledInstr(None, i) for i in instrs]
               + [LabeledInstr("finish", FINISH())])
    machine = PISAMachine(wrapped)
    machine.run()
    store = {v: machine.mem.get(v, 0) for v in range(nvars)}
    regs = {r: machine._read_reg(f"r{r}") for r in range(3, 9)}
    return store, regs


def run_python_compiler(source: str, nvars: int) -> dict:
    machine = PISAMachine(compile_program(parse(tokenize(source))))
    machine.run()
    return {v: machine.mem.get(v, 0) for v in range(nvars)}


def main() -> int:
    if not os.path.exists(DRIVER):
        print(f"{DRIVER} not found — build it first:\n"
              f"    make -C rocq -f Makefile.driver", file=sys.stderr)
        return 2

    cases = parse_driver_output(
        subprocess.run([DRIVER], capture_output=True, text=True, check=True).stdout)

    failures = 0
    for case in cases:
        name, nvars = case["name"], case["nvars"]
        problems = []

        # 1. the Python interpreter against the verified machine
        store, regs = run_bare(case["instrs"], nvars)
        if store != case["vars"]:
            problems.append(f"interpreter: {store} != verified {case['vars']}")
        dirty = {r: v for r, v in regs.items() if v != 0}
        if dirty:
            problems.append(f"interpreter left garbage: {dirty}")

        # 2. the Python compiler against the verified compiler
        py_store = run_python_compiler(case["source"], nvars)
        if py_store != case["vars"]:
            problems.append(f"codegen.py: {py_store} != verified {case['vars']}")

        # 3. cleanliness as computed by the verified machine
        verified_dirty = {r: v for r, v in case["regs"].items() if v != 0}
        if verified_dirty:
            problems.append(f"verified compiler left garbage: {verified_dirty}")

        if problems:
            failures += 1
            print(f"DIFF  {name}")
            for p in problems:
                print(f"      {p}")
        else:
            print(f"OK    {name}: {case['vars']}")

    print(f"\n{len(cases) - failures}/{len(cases)} programs agree "
          f"(verified compiler vs codegen.py, verified machine vs pisa_interp.py)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
