#!/usr/bin/env python3
"""Cross-check the verified `if` / `from/loop/until` compiler against the Python side.

`rocq/CompileLoop.v` proves `compile_l` correct on the PC-based machine of
`rocq/PISACtl.v` and checks its output by `vm_compute` (`ex_loop`, ...).
For each of those programs this script

  1. asks Rocq for the code `compile_l` emits (labels from 1, `finish` =
     label 0) and for the result the verified machine computes (`run_l`,
     which appends the `finish:` line), via `rocq compile` on a scratch file;
  2. runs the verified compiler's code on **`pisa_interp.py`** (between
     `start: START` and `finish: FINISH`) and compares the final store,
     `br` and the scratch registers r3..r8;
  3. compiles the same program, written in Janus, with **`codegen.py`**
     (`compile_program`) and runs it on `pisa_interp.py` — same store and
     registers expected;
  4. compares the control-flow skeleton (labels, branches, flag updates, the
     SLTX pairs of the `e != 0` normalisation, and the SLTX / XORI / ORX /
     ANDX of comparisons and `&&` / `||`, in order) of `codegen.py`'s
     *unoptimized* output with the Rocq layout, up to label renaming — i.e.
     that `compile_l` really is `_gen_if` / `_gen_from`'s layout and emits
     `_gen_binop` / `_gen_uneval_binop`'s operator code.

The last three programs violate an entry / re-entry / `fi` assertion.  Janus
rejects them; both compilers jump to `finish` with the flag r3 = 1, which
`pisa_interp.py` reports as garbage.  Steps 2-3 therefore run with the
garbage check off and compare the dirty register file instead.

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
from pisa import (ADD, SUB, XOR, ADDI, SUBI, XORI, NEG, EXCH, SLTX,  # noqa: E402
                  ORX, ANDX, BRA, BEQ, BNE, START, FINISH, LabeledInstr)

# Overridable so that a mutated copy of the development can be checked
# (mutation testing of this script itself).
ROCQ_DIR = os.environ.get("ROCQ_DIR", os.path.join(ROOT, "rocq"))

DECLS = "int x0\nint x1\nint x2\nint c\nint y0\nint y1\nint d\nprocedure main\n"
ROT = "x1 <=> x2 x0 <=> x1"

# The `finish` label of CompileLoop.v (`fin_label`).
FIN = 0

# Rocq name in CompileLoop.v  ->  the same program in Janus
PROGRAMS = {
    "prog_loop": f"x0 += 1\nfrom x0 do c += 1 loop {ROT} until x2\n",
    "prog_loop_if": ("x0 += 1\nfrom x0 do if x2 then c += 10 else c += 1 fi x2\n"
                     f"loop {ROT} until x2\n"),
    "prog_loop_skip": f"x0 += 1\nfrom x0 do skip loop {ROT} until x2\n",
    "prog_nested_loop": ("x0 += 1\ny0 += 1\n"
                         "from x0 do from y0 do d += 1 loop y0 <=> y1 until y1\n"
                         f"loop {ROT} y0 <=> y1 until x2\n"),
    # non-Boolean tests (valid Janus: truth is nonzero)
    "prog_if5": "x2 += 5\nif x2 then c += 10 else skip fi c\n",
    "prog_if_else": "x1 += 7\nif x0 then c += 1 else x1 += 1 fi x0 + x1 - x1\n",
    "prog_loop3": f"x0 += 3\nfrom x0 do c += 1 loop {ROT} until x2\n",
    # the constants 0/1 are not normalised (`_as_flag`)
    "prog_const": ("if 1 then c += 1 else skip fi 1\n"
                   "from 1 do c += 2 loop skip until 1\n"),
    # violated assertions: both sides jump to finish with r3 = 1
    "prog_v1": f"x2 += 1\nfrom x0 do c += 1 loop {ROT} until x2\n",
    "prog_v2": "x0 += 1\nfrom x0 do c += 1 loop x2 += 1 until x2\n",
    "prog_if_v": "x2 += 5\nif x2 then c += 1 else skip fi x0\n",
    # milestone 3: comparisons and && / || in tests (`_as_flag` keeps them)
    "prog_cmp_and": ("x0 += 2\nx1 += 5\n"
                     "if (x0 < 3) && (x1 != 0) then c += 1 else c += 2 fi c = 1\n"),
    "prog_cmp_or": ("x0 += 5\n"
                    "if (x0 < 3) || (x0 >= 7) then c += 1 else c += 2 fi (c = 1) || (c > 5)\n"),
    "prog_cmp_loop": "from c = 0 do c += 1 loop x1 += 1 until 3 <= c\n",
    "prog_cmp_loop2": ("from (c = 0) && (d <= 0) do c += 2 loop d += 1 "
                       "until (c > 5) || (x1 = 7)\n"),
    # && / || on non-Boolean operands (`_logical_operands`: 1 && 2 is 1, not 1 & 2)
    "prog_logic_nonbool": ("x2 += 1\nx1 += 2\nif x2 && x1 then c += 1 else skip fi c\n"
                           "if x1 || x0 then c += 10 else skip fi c\n"),
    "prog_cmp_neg": ("x0 -= 4\nif (x0 < 0) && (x0 > 0 - 5) then c += 1 else skip fi c = 1\n"
                     "if (x0 <= 0 - 4) && (x0 >= 0 - 4) then d += 1 else skip fi d\n"),
}
NVARS = 7


def rocq_eval() -> str:
    lines = ["From Stdlib Require Import ZArith List.",
             "Require Import PISA Src Compile PISACtl CompileIf CompileLoop.",
             "Import ListNotations.",
             "Set Printing Depth 1000000.", "Set Printing Width 1000000."]
    for name in PROGRAMS:
        lines.append(f"Eval vm_compute in fst (compile_l fin_label {name} scratch 1%nat).")
        lines.append(f"Eval vm_compute in run_l {name}.")
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
    r"\((None|Some (\d+)), (COp \((\w+)([^()]*)\)|(CBra|CBeq|CBne)([^()]*))\)")


def lab(k: int) -> str:
    return "finish" if k == FIN else f"L{k}"


def parse_lprog(term: str) -> list:
    term = term.replace("%nat", "")
    out = []
    for m in LINE_RE.finditer(term):
        label = lab(int(m.group(2))) if m.group(2) is not None else None
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
                "ISltx": lambda a: SLTX(f"r{a[0]}", f"r{a[1]}", f"r{a[2]}"),
                "IOrx": lambda a: ORX(f"r{a[0]}", f"r{a[1]}"),
                "IAndx": lambda a: ANDX(f"r{a[0]}", f"r{a[1]}", f"r{a[2]}"),
            }[op](args)
        else:
            args = [int(a) for a in m.group(7).split()]
            kind = m.group(6)
            if kind == "CBra":
                instr = BRA(lab(args[0]))
            elif kind == "CBeq":
                instr = BEQ(f"r{args[0]}", f"r{args[1]}", lab(args[2]))
            else:
                instr = BNE(f"r{args[0]}", f"r{args[1]}", lab(args[2]))
        out.append(LabeledInstr(label, instr))
    return out


def parse_result(term: str) -> dict:
    m = re.match(r"\((true|false), (-?\d+), \[(.*?)\], \[(.*?)\]\)", term.strip())

    def ints(s):
        return [int(x) for x in s.replace("(", "").replace(")", "").split(";")]
    return {"end": m.group(1) == "true", "br": int(m.group(2)),
            "mem": ints(m.group(3)), "regs": ints(m.group(4))}


# Registers compared on pisa_interp.py: r3..r8 must equal the verified
# machine's (`observe_l` reports those), r9..r31 must be 0 — comparisons and
# `&&` / `||` use scratch registers above r8.
ALL_REGS = range(3, 32)


def padded(regs: list) -> list:
    return regs + [0] * (len(ALL_REGS) - len(regs))


def run_on_interp(code: list) -> tuple:
    wrapped = ([LabeledInstr("start", START())] + code
               + [LabeledInstr("finish", FINISH())])
    m = PISAMachine(wrapped)
    m.check_clean = False          # violations end dirty; registers are compared below
    m.run()
    return ([m.mem.get(v, 0) for v in range(NVARS)],
            [m._read_reg(f"r{r}") for r in ALL_REGS], m.br)


def run_codegen(src: str) -> tuple:
    m = PISAMachine(compile_program(parse(tokenize(DECLS + src))))
    m.check_clean = False
    m.run()
    return ([m.mem.get(v, 0) for v in range(NVARS)],
            [m._read_reg(f"r{r}") for r in ALL_REGS])


def skeleton(code: list) -> list:
    """Labels, branches, flag updates and SLTX pairs, labels renamed by first use.

    SLTX / XORI name their register only when it is a flag register (one
    that is branched on): the value registers of comparisons differ, because
    Compile.v puts an operator's result in its target register and the
    operands above it, while `codegen.py` allocates the result after the
    operands.  ORX / ANDX (`=`, `!=`, `&&`, `||`) are listed by name.
    """
    names = {}

    def ren(lab):
        return names.setdefault(lab, f"l{len(names)}")
    # Flag registers: the ones branched on.  `XOR r r` on a flag register is
    # a layout-level flag clear (what `_gen_from` used to emit) and is kept;
    # on any other register it is codegen.py's straight-line garbage clear
    # (`_clear_garbage` / `_nonzero_into` after a binary operator), which
    # Compile.v's expression code avoids by unevaluating operands instead.
    flags = {li.instr.rd for li in code if isinstance(li.instr, (BEQ, BNE))}
    out = []
    for li in code:
        i, kind = li.instr, type(li.instr).__name__
        lb = ren(li.label) if li.label else None
        if isinstance(i, BRA):
            out.append((lb, "BRA", ren(i.label)))
        elif isinstance(i, (BEQ, BNE)):
            out.append((lb, kind, i.rd, i.rs, ren(i.label)))
        elif isinstance(i, XOR) and i.rd == i.rs and i.rd in flags:
            out.append((lb, "XOR", i.rd, i.rs))
        elif isinstance(i, XORI):
            out.append((lb, "XORI", i.rd if i.rd in flags else "v", i.c))
        elif isinstance(i, SLTX):
            # the value register differs (Compile.v's gen_expr vs codegen's
            # register allocator); the flag register and the r0 side do not
            out.append((lb, "SLTX", i.rd if i.rd in flags else "v",
                        i.rs == "r0", i.rt == "r0"))
        elif isinstance(i, (ORX, ANDX)):
            out.append((lb, kind))
        elif isinstance(i, ADDI) and i.rd == "r0":
            out.append((lb, "NOP"))
        elif li.label:
            out.append((lb, kind))
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
        if (mem, regs, br) != (want["mem"], padded(want["regs"]), 0):
            problems.append(f"pisa_interp on verified code: {mem} {regs} br={br}"
                            f" != verified {want['mem']} {want['regs']}")
        py_mem, py_regs = run_codegen(src)
        if (py_mem, py_regs) != (want["mem"], padded(want["regs"])):
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
            flag = "  (flag dirty: assertion violated)" if any(want["regs"]) else ""
            print(f"OK    {name}: mem={want['mem']} regs={want['regs']} "
                  f"({len(code)} lines, {len(sk_rocq)} control lines agree){flag}")
    print(f"\n{len(PROGRAMS) - failures}/{len(PROGRAMS)} programs agree "
          "(verified machine vs pisa_interp.py, verified compiler vs codegen.py, layout)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
