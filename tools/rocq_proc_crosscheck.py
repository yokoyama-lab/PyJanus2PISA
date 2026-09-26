#!/usr/bin/env python3
"""Cross-check the verified procedure compiler (rocq/CompileProc.v) against the Python side.

`rocq/CompileProc.v` proves `compile_p` / `whole` correct on the machine of
`rocq/PISAProc.v` (PISACtl.v plus `pisa_interp.py`'s software call stack).
For each program of `rocq/TestProc.v` this script

  1. asks Rocq (`vm_compute`) for the whole program `whole Γ main k` — with
     the stack offset `k` that `codegen.py` chooses for the same source —,
     for the verified machine's final state on it, and for the source
     semantics' final store (`run_src`, the fuel interpreter proved sound for
     `exec_p`);
  2. renders the verified code as PISA (procedure labels `pF`, `pF_top`,
     `pF_bot`, `pF_inv`, …, so that `pisa_interp.py` recognises the
     procedures by its `_top`/`_bot` convention) and runs it on
     **`pisa_interp.py`**: store, r1..r8 and `br` must equal the verified
     machine's — this checks the call-stack model against the interpreter;
  3. compiles the Janus source with **`codegen.py`** and runs it: the store
     must equal the source semantics' (except for the known defect below);
  4. compares, procedure by procedure (forward and `_inv` companions), the
     prologue instruction for instruction and the control skeleton of the
     body (as in `rocq_loop_crosscheck.py`) of `codegen.py`'s *unoptimised*
     output with the verified layout, and the `start`/`finish` wrapper;
  5. if a PyJanus checkout is available (`PYJANUS_DIR`, default
     ~/dev/github.com/yokoyama-lab/PyJanus), runs the source under PyJanus
     and compares with the source semantics — an independent check that
     `exec_p` is Janus's.

`g_s2` (a call in the `loop` part S2 of a `from` loop) and `g_rec_s2`
(recursion through S2, forwards and backwards) are ordinary programs now:
`_gen_from` used to run S2 with the loop flag r3 = 1 while the callee's body
assumes r3 = 0 (verified layout c = 17, codegen.py c = 5 and d = 10, Janus
c = 15); it runs S2 with the flag at 0 since the fix, and `wf_p` no longer
excludes calls there (`s2_call_works` in TestProc.v).

`g_finv` (procedures named `f` and `f_inv` with `uncall f`) is an ordinary
program too: the companion of `f` used to be the label `f_inv`, the user's
procedure.  codegen.py now derives procedure labels with `_proc_label`
(every `_` doubled).  No program is expected to diverge any more.

Usage:
    make -C rocq                 # the .vo files must exist
    python3 tools/rocq_proc_crosscheck.py
"""

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexer import tokenize                      # noqa: E402
from parser import parse                        # noqa: E402
from codegen import CodeGen, compile_program, _proc_label    # noqa: E402
from pisa_interp import PISAMachine             # noqa: E402
from pisa import (ADD, SUB, XOR, ADDI, SUBI, XORI, NEG, EXCH, SLTX,  # noqa: E402
                  ORX, ANDX, BRA, BEQ, BNE, SWAPBR, START, FINISH, LabeledInstr)
from rocq_loop_crosscheck import skeleton, split_evals   # noqa: E402

ROCQ_DIR = os.environ.get("ROCQ_DIR", os.path.join(ROOT, "rocq"))
PYJANUS_DIR = os.environ.get(
    "PYJANUS_DIR", os.path.expanduser("~/dev/github.com/yokoyama-lab/PyJanus"))

DECLS = "int x0\nint x1\nint x2\nint c\nint y0\nint y1\nint d\n"
NVARS = 7
ROT = "x1 <=> x2\n  x0 <=> x1"

# Rocq environment -> (Janus procedure names in the environment's order,
#                      index of main, Janus source of the procedures, kind)
PROGRAMS = {
    "g_call": (["f", "main"], 1,
               "procedure f\n  c += 1\n  x0 += 2\n"
               "procedure main\n  call f\n  call f\n", "ok"),
    "g_uncall": (["f", "main"], 1,
                 "procedure f\n  c += 3\n  x1 ^= c\n"
                 "procedure main\n  call f\n  call f\n  uncall f\n", "ok"),
    "g_nested": (["f", "g", "main"], 2,
                 "procedure f\n  c += 1\n"
                 "procedure g\n  call f\n  d += 1\n  call f\n"
                 "procedure main\n  call g\n  call g\n  uncall g\n", "ok"),
    "g_loop": (["f", "main"], 1,
               "procedure f\n  c += 1\n"
               "procedure main\n  x0 += 1\n"
               f"  from x0 do call f loop {ROT} until x2\n  call f\n", "ok"),
    "g_if": (["f", "main"], 1,
             "procedure f\n  c += 10\n"
             "procedure main\n  x2 += 1\n  if x2 then call f else skip fi c\n"
             "  if x0 then call f else uncall f fi x0\n", "ok"),
    "g_rec": (["f", "main"], 1,
              "procedure f\n  if x0 then\n    x0 -= 1\n    c += 1\n    call f\n    x0 += 1\n"
              "  else skip fi x0\n"
              "procedure main\n  x0 += 3\n  call f\n  call f\n  uncall f\n", "ok"),
    "g_loop_if": (["f", "main"], 1,
                  "procedure f\n  d += 1\n"
                  "procedure main\n  x0 += 1\n"
                  f"  from x0 do if x2 then call f else call f fi x2 loop {ROT} until x2\n"
                  "  call f\n", "ok"),
    "g_s2": (["f", "main"], 1,
             "procedure f\n  c += 5\n"
             "procedure main\n  x0 += 1\n"
             f"  from x0 do skip loop call f\n  {ROT} until x2\n  call f\n", "ok"),
    "g_rec_s2": (["h", "main"], 1,
                 "procedure h\n  d += 1\n  if y0 then\n    y0 -= 1\n    y1 += 1\n"
                 "    from y1 do skip loop\n      y1 -= 1\n      call h\n      x1 += 1\n"
                 "    until x1\n    x1 -= 1\n    y0 += 1\n  else skip fi y0\n"
                 "procedure main\n  y0 += 2\n  call h\n  uncall h\n  call h\n", "ok"),
    "g_finv": (["f", "f_inv", "main"], 2,
               "procedure f\n  x0 += 1\n  x0 += 1\n"
               "procedure f_inv\n  x1 += 100\n"
               "procedure main\n  call f\n  call f\n  call f_inv\n  call f_inv\n  uncall f\n",
               "ok"),
    # milestone 3: comparisons and && in a recursive procedure and a loop
    "g_rec_cmp": (["f", "main"], 1,
                  "procedure f\n  if (x0 > 0) && (c < 100) then\n    x0 -= 1\n    c += 1\n"
                  "    call f\n    x0 += 1\n  else skip fi x0 > 0\n"
                  "procedure main\n  x0 += 3\n  call f\n  call f\n  uncall f\n"
                  "  from d = 0 do d += 1 loop skip until d >= 2\n", "ok"),
}


def codegen_k(src: str) -> int:
    """The stack offset `codegen.py` loads into r1 (`start: START; ADDI r1 k`)."""
    code = CodeGen().gen_program(parse(tokenize(src)))
    s = next(i for i, li in enumerate(code) if li.label == "start")
    assert isinstance(code[s + 1].instr, ADDI) and code[s + 1].instr.rd == "r1"
    return code[s + 1].instr.c


def rocq_eval(jobs: list) -> list:
    lines = ["From Stdlib Require Import ZArith List.",
             "Require Import PISA Src Compile PISACtl CompileIf CompileLoop "
             "PISAProc SrcProc CompileProc TestProc.",
             "Import ListNotations.",
             "Set Printing Depth 1000000.", "Set Printing Width 1000000.",
             "Definition obs_k (Γ : penv) (main : nat) (k : Z) :=",
             "  match pexec_fuel mach_fuel (ptab (length Γ)) (whole Γ main k) (start_state Γ) with",
             "  | Some (c, st) => Some (Nat.eqb (cpc c) (length (whole Γ main k)), cbr c,",
             "      length st, map (fun a => mem (cst c) (Z.of_nat a)) (seq 0 7),",
             "      map (regs (cst c)) (seq 1 8))",
             "  | None => None end."]
    for env, main, k in jobs:
        lines.append(f"Eval vm_compute in whole {env} {main}%nat {k}%Z.")
        lines.append(f"Eval vm_compute in obs_k {env} {main}%nat {k}%Z.")
        lines.append(f"Eval vm_compute in run_src {env} {main}%nat.")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ProcDump.v")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        out = subprocess.run(["rocq", "compile", "-Q", ROCQ_DIR, "", path],
                             capture_output=True, text=True, check=True, cwd=d)
    return split_evals(out.stdout)


ARG = r"(?:-?\d+|\(-\d+\))"
LINE_RE = re.compile(
    r"\((None|Some (\d+)), (?:COp \((\w+)((?:\s+" + ARG + r")*)\)"
    r"|(CBra|CBeq|CBne|CSwapbr)((?:\s+\d+)*))\)")


def label_name(lab: int, np: int) -> str:
    """Rocq label -> PISA label: 0 is `finish`, then procedure labels, then L<n>."""
    if lab == 0:
        return "finish"
    if lab < 1 + 6 * np:
        f, r = divmod(lab - 1, 6)
        d, kind = divmod(r, 3)
        base = f"p{f}" + ("_inv" if d else "")
        return base + ("", "_top", "_bot")[kind]
    return f"L{lab}"


def parse_lprog(term: str, np: int) -> list:
    term = term.replace("%nat", "").replace("%Z", "")
    out = []
    for m in LINE_RE.finditer(term):
        label = label_name(int(m.group(2)), np) if m.group(2) is not None else None
        if m.group(3):
            op = m.group(3)
            a = [int(x.strip("()")) for x in re.findall(ARG, m.group(4))]
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
            }[op](a)
        else:
            kind = m.group(5)
            a = [int(x) for x in m.group(6).split()]
            if kind == "CBra":
                instr = BRA(label_name(a[0], np))
            elif kind == "CBeq":
                instr = BEQ(f"r{a[0]}", f"r{a[1]}", label_name(a[2], np))
            elif kind == "CBne":
                instr = BNE(f"r{a[0]}", f"r{a[1]}", label_name(a[2], np))
            else:
                instr = SWAPBR(f"r{a[0]}")
        out.append(LabeledInstr(label, instr))
    # the wrapper: `start: START; ADDI r1 k; BRA main; finish: FINISH`
    assert out[-4].label is None and out[-1].label == "finish"
    out[-4] = LabeledInstr("start", START())
    out[-1] = LabeledInstr("finish", FINISH())
    return out


def ints(s: str) -> list:
    return [int(x) for x in s.replace("(", "").replace(")", "").split(";") if x.strip()]


def parse_obs(term: str) -> dict:
    term = term.replace("%nat", "").replace("%Z", "")
    if term.strip().startswith("None"):
        return {"end": False, "br": None, "depth": None, "mem": None, "regs": [None]}
    m = re.match(r"\s*Some \((true|false), (-?\d+|\(-\d+\)), (\d+), \[(.*?)\], \[(.*?)\]\)",
                 term)
    return {"end": m.group(1) == "true", "br": int(m.group(2).strip("()")),
            "depth": int(m.group(3)), "mem": ints(m.group(4)), "regs": ints(m.group(5))}


def parse_src(term: str):
    term = term.strip()
    if term.startswith("None"):
        return None
    return ints(re.match(r"Some \[(.*?)\]", term).group(1))


def run_machine(code: list) -> tuple:
    m = PISAMachine(code)
    m.check_clean = False
    try:
        m.run(max_steps=1_000_000)
    except Exception as e:          # a mutated layout may loop or get stuck
        return (f"{type(e).__name__}: {e}", None, None)
    return ([m.mem.get(v, 0) for v in range(NVARS)],
            [m._read_reg(f"r{r}") for r in range(1, 9)], m.br)


def proc_parts(code: list, name: str):
    labels = {li.label: i for i, li in enumerate(code) if li.label}
    if name not in labels or name + "_bot" not in labels:
        return None
    s, e = labels[name], labels[name + "_bot"]
    return [li.instr for li in code[s:s + 6]], code[s + 6:e]


def pyjanus_store(src: str):
    if not os.path.isdir(PYJANUS_DIR):
        return None
    from pyjanus_crosscheck import run_pyjanus
    with tempfile.TemporaryDirectory() as d:
        st = run_pyjanus(src, PYJANUS_DIR, d)
    return [st[v] for v in ["x0", "x1", "x2", "c", "y0", "y1", "d"]]


def main() -> int:
    jobs, meta = [], []
    for env, (names, main_ix, procs, kind) in PROGRAMS.items():
        src = DECLS + procs
        k = codegen_k(src)
        jobs.append((env, main_ix, k))
        meta.append((env, names, src, k, kind))
    terms = rocq_eval(jobs)
    failures = 0
    for i, (env, names, src, k, kind) in enumerate(meta):
        np = len(names)
        code = parse_lprog(terms[3 * i], np)
        want = parse_obs(terms[3 * i + 1])
        source = parse_src(terms[3 * i + 2])
        problems, notes = [], []
        # 1. the verified machine ends cleanly
        if not want["end"] or want["br"] != 0 or want["depth"] != 0:
            problems.append(f"verified machine did not end cleanly: {want}")
        if want["regs"][0] != k or any(want["regs"][1:]):
            problems.append(f"verified machine registers r1..r8 = {want['regs']}")
        # 2. pisa_interp.py on the verified code
        mem, regs, br = run_machine(code)
        if (mem, regs, br) != (want["mem"], want["regs"], 0):
            problems.append(f"pisa_interp on verified code: {mem} {regs} br={br}"
                            f" != verified machine {want['mem']} {want['regs']}")
        # 3. codegen.py + pisa_interp.py
        py_mem, py_regs, _ = run_machine(compile_program(parse(tokenize(src))))
        if want["mem"] != source:
            problems.append(f"verified machine {want['mem']} != source {source}")
        if (py_mem, (py_regs or [None] * 8)[2:]) != (source, [0] * 6):
            problems.append(f"codegen.py: {py_mem} {py_regs} != source {source}")
        # 4. layout: prologue, body skeleton, wrapper
        gen = CodeGen().gen_program(parse(tokenize(src)))
        compared = 0
        for f, name in enumerate(names):
            for suffix in ("", "_inv"):
                py = proc_parts(gen, _proc_label(name) + suffix)
                if py is None:
                    continue            # dead or never uncalled: codegen omits it
                rq = proc_parts(code, f"p{f}" + suffix)
                if rq is None or rq[0] != py[0]:
                    problems.append(f"prologue of {name + suffix} differs")
                    continue
                if skeleton(rq[1]) != skeleton(py[1]):
                    problems.append(f"body of {name + suffix} differs:\n"
                                    f"  rocq {skeleton(rq[1])}\n  py   {skeleton(py[1])}")
                compared += 1
        s = next(j for j, li in enumerate(gen) if li.label == "start")
        wrap_py = [type(li.instr).__name__ for li in gen[s:s + 4]]
        wrap_rq = [type(li.instr).__name__ for li in code[-4:]]
        if (wrap_py != wrap_rq or code[-3].instr != gen[s + 1].instr
                or code[-2].instr.label != f"p{names.index('main')}"
                or gen[s + 2].instr.label != "main"):
            problems.append(f"wrapper differs: {wrap_rq} vs {wrap_py}")
        notes.append(f"{compared} procedure bodies agree")
        # 5. PyJanus
        pj = pyjanus_store(src)
        if pj is not None:
            if pj != source:
                problems.append(f"PyJanus {pj} != source semantics {source}")
            else:
                notes.append("PyJanus agrees")
        if problems:
            failures += 1
            print(f"DIFF  {env}")
            for p in problems:
                print(f"      {p}")
        else:
            print(f"OK    {env}: store={source} k={k} ({len(code)} lines); " + "; ".join(notes))
    print(f"\n{len(meta) - failures}/{len(meta)} programs as expected "
          "(verified machine vs pisa_interp.py, codegen.py vs source semantics, layout"
          + (", PyJanus" if os.path.isdir(PYJANUS_DIR) else "") + ")")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
