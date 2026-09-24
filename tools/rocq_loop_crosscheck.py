#!/usr/bin/env python3
"""Cross-check the verified `from/loop/until` compiler against the Python side.

`rocq/CompileLoop.v` proves `compile_l` correct on the PC-based machine of
`rocq/PISACtl.v` and checks its output by `vm_compute` (`ex_loop`, ...).
For each of those programs this script

  1. asks Rocq for the code `compile_l` emits and for the result the verified
     machine computes (`run_l`), via `rocq compile` on a scratch file;
  2. runs the verified compiler's code on **`pisa_interp.py`** and compares the
     final store, `br`-free termination and the scratch registers;
  3. compiles the same program, written in Janus, with **`codegen.py`**
     (`compile_program`) and runs it on `pisa_interp.py` — same store expected;
  4. compares the control-flow skeleton (labels, branches, flag updates) of
     `codegen.py`'s *unoptimized* output with the Rocq layout, up to label
     renaming — i.e. that `compile_l` really is `_gen_from`'s layout.

The last two programs violate an entry / re-entry assertion.  Janus rejects
them; both compilers accept them silently (clean registers).  That is the
defect `CompileLoop.v` documents, so for them step 3 expects agreement too.

Usage:
    make -C rocq                 # the .vo files must exist
    python3 tools/rocq_loop_crosscheck.py
"""

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from lexer import tokenize                      # noqa: E402
from parser import parse                        # noqa: E402
from codegen import CodeGen, compile_program    # noqa: E402
from pisa_interp import PISAMachine             # noqa: E402
from pisa import (ADD, SUB, XOR, ADDI, SUBI, XORI, NEG, EXCH,  # noqa: E402
                  BRA, BEQ, START, FINISH, LabeledInstr)

ROCQ_DIR = os.path.join(ROOT, "rocq")

DECLS = "int x0\nint x1\nint x2\nint c\nint y0\nint y1\nint d\nprocedure main\n"
ROT = "x1 <=> x2 x0 <=> x1"

# Rocq name in CompileLoop.v  ->  the same program in Janus
PROGRAMS = {
    "prog_loop": f"x0 += 1\nfrom x0 do c += 1 loop {ROT} until x2\n",
    "prog_loop_if": ("x0 += 1\nfrom x0 do if x2 then c += 10 else c += 1 fi x2\n"
                     f"loop {ROT} until x2\n"),
    "prog_loop_skip": f"x0 += 1\nfrom x0 do skip loop {ROT} until x2\n",
    "prog_nested_loop": ("x0 += 1\ny0 += 1\n"
                         "from x0 do from y0 do d += 1 loop y0 <=> y1 until y1\n"
                         f"loop {ROT} y0 <=> y1 until x2\n"),
    "prog_v1": f"x2 += 1\nfrom x0 do c += 1 loop {ROT} until x2\n",
    "prog_v2": "x0 += 1\nfrom x0 do c += 1 loop x2 += 1 until x2\n",
}
NVARS = 7


def rocq_eval() -> str:
    lines = ["From Stdlib Require Import ZArith List.",
             "Require Import PISA Src Compile PISACtl CompileIf CompileLoop.",
             "Import ListNotations.",
             "Set Printing Depth 1000000.", "Set Printing Width 1000000."]
    for name in PROGRAMS:
        lines.append(f"Eval vm_compute in fst (compile_l {name} scratch 0%nat).")
        lines.append(f"Eval vm_compute in run_l clr_codegen {name}.")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "LoopDump.v")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        out = subprocess.run(["rocq", "compile", "-Q", ROCQ_DIR, "", path],
                             capture_output=True, text=True, check=True, cwd=d)
    return out.stdout


def split_evals(text: str) -> list:
    """Each `Eval` prints `= <term>\\n : <type>`; return the terms in order."""
    return [m.group(1) for m in re.finditer(r"=\s(.*?)\n\s*:\s", text, re.S)]


LINE_RE = re.compile(
    r"\((None|Some (\d+)), (COp \((\w+)([^()]*)\)|(CBra|CBeq)([^()]*))\)")


def parse_lprog(term: str) -> list:
    term = term.replace("%nat", "")
    out = []
    for m in LINE_RE.finditer(term):
        label = f"L{m.group(2)}" if m.group(2) is not None else None
        if m.group(4):
            op, args = m.group(4), [int(a) for a in m.group(5).split()]
            instr = {
                "IAdd": lambda a: ADD(f"r{a[0]}", f"r{a[1]}"),
                "ISub": lambda a: SUB(f"r{a[0]}", f"r{a[1]}"),
                "IXor": lambda a: XOR(f"r{a[0]}", f"r{a[1]}"),
                "IAddi": lambda a: ADDI(f"r{a[0]}", a[1]),
                "ISubi": lambda a: SUBI(f"r{a[0]}", a[1]),
                "IXori": lambda a: XORI(f"r{a[0]}", a[1]),
                "INeg": lambda a: NEG(f"r{a[0]}"),
                "IExch": lambda a: EXCH(f"r{a[0]}", f"r{a[1]}"),
            }[op](args)
        else:
            args = [int(a) for a in m.group(7).split()]
            instr = (BRA(f"L{args[0]}") if m.group(6) == "CBra"
                     else BEQ(f"r{args[0]}", f"r{args[1]}", f"L{args[2]}"))
        out.append(LabeledInstr(label, instr))
    return out


def parse_result(term: str) -> dict:
    m = re.match(r"\((true|false), (-?\d+), \[(.*?)\], \[(.*?)\]\)", term.strip())

    def ints(s):
        return [int(x) for x in s.replace("(", "").replace(")", "").split(";")]
    return {"end": m.group(1) == "true", "br": int(m.group(2)),
            "mem": ints(m.group(3)), "regs": ints(m.group(4))}


def run_on_interp(code: list) -> tuple:
    wrapped = ([LabeledInstr("start", START())] + code
               + [LabeledInstr("finish", FINISH())])
    m = PISAMachine(wrapped)
    m.run()
    return ([m.mem.get(v, 0) for v in range(NVARS)],
            [m._read_reg(f"r{r}") for r in range(3, 9)], m.br)


def run_codegen(src: str) -> tuple:
    m = PISAMachine(compile_program(parse(tokenize(DECLS + src))))
    m.run()
    return ([m.mem.get(v, 0) for v in range(NVARS)],
            [m._read_reg(f"r{r}") for r in range(3, 9)])


def skeleton(code: list) -> list:
    """Labels, branches and flag updates, with labels renamed by first use."""
    names = {}

    def ren(lab):
        return names.setdefault(lab, f"l{len(names)}")
    out = []
    for li in code:
        i, kind = li.instr, type(li.instr).__name__
        if isinstance(i, BRA):
            out.append((ren(li.label) if li.label else None, "BRA", ren(i.label)))
        elif isinstance(i, BEQ):
            out.append((ren(li.label) if li.label else None, "BEQ", i.rd, i.rs, ren(i.label)))
        elif isinstance(i, XOR) and i.rd == i.rs:
            out.append((ren(li.label) if li.label else None, "XOR", i.rd, i.rs))
        elif isinstance(i, XORI):
            out.append((ren(li.label) if li.label else None, "XORI", i.rd, i.c))
        elif isinstance(i, ADDI) and i.rd == "r0":
            out.append((ren(li.label) if li.label else None, "NOP"))
        elif li.label:
            out.append((ren(li.label), kind))
    return out


def codegen_body(src: str) -> list:
    """`codegen.py`'s unoptimized main body (prologue and `main_bot` removed)."""
    code = CodeGen().gen_program(parse(tokenize(DECLS + src)))
    start = next(k for k, li in enumerate(code) if li.label == "main") + 6
    stop = next(k for k, li in enumerate(code) if li.label == "main_bot")
    return code[start:stop]


def main() -> int:
    terms = split_evals(rocq_eval())
    failures = 0
    for k, (name, src) in enumerate(PROGRAMS.items()):
        code = parse_lprog(terms[2 * k])
        want = parse_result(terms[2 * k + 1])
        problems = []
        if not want["end"] or want["br"] != 0:
            problems.append(f"verified machine did not terminate cleanly: {want}")
        mem, regs, br = run_on_interp(code)
        if (mem, regs, br) != (want["mem"], want["regs"], 0):
            problems.append(f"pisa_interp on verified code: {mem} {regs} br={br}"
                            f" != verified {want['mem']} {want['regs']}")
        py_mem, py_regs = run_codegen(src)
        if (py_mem, py_regs) != (want["mem"], want["regs"]):
            problems.append(f"codegen.py: {py_mem} {py_regs} != verified")
        sk_rocq, sk_py = skeleton(code), skeleton(codegen_body(src))
        if sk_rocq != sk_py:
            problems.append(f"layout differs:\n  rocq {sk_rocq}\n  py   {sk_py}")
        if problems:
            failures += 1
            print(f"DIFF  {name}")
            for p in problems:
                print(f"      {p}")
        else:
            print(f"OK    {name}: mem={want['mem']} regs={want['regs']} "
                  f"({len(code)} lines, {len(sk_rocq)} control lines agree)")
    print(f"\n{len(PROGRAMS) - failures}/{len(PROGRAMS)} programs agree "
          "(verified machine vs pisa_interp.py, verified compiler vs codegen.py, layout)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
