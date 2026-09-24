#!/usr/bin/env python3
"""difftest_pisa: differential testing of PISA interpreters.

Three sub-commands:

  janus  FILES...   For each Janus source: compile with janus2pisa (forward) and
                    with --inverse (backward); run both on pisa_interp.PISAMachine
                    and, after conversion to PAL (tools/pisa2pal.py), on phpisa
                    (tools/phpisa_dump.php); diff final variable memory and
                    registers.  Then run forward followed by the inverse on both
                    interpreters (round trip) and check the store returns to the
                    initial one.  Reports per program:
                        fwd / bwd  : OK | MISMATCH(first differing location) | ERROR(side: msg)
                        rt-pisa / rt-php : round trip on each interpreter
  rfcl   FILES...   For each RL program: rl-to-pisa -> PAL -> phpisa (and the
                    same PAL on PISAMachine through tools/pal2pisa.py); oracle is
                    `rl-run` (final values of the output variables).
  pal    FILE.pal   Run a PAL program (Rlang-compiler output) on phpisa and on
                    PISAMachine via pal2pisa; compare OUTPUT lines and registers;
                    list the constructs each interpreter lacks.

Corpus files whose first line is ``// SKIP: reason`` are reported as SKIPPED.

Modes for `janus` (--mode): ``faithful`` converts instruction by instruction;
``pendulum-calls`` additionally rewrites janus2pisa's procedure protocol into
Pendulum's and relocates the stack (see pisa2pal.py); ``pendulum-cf`` also
lowers janus2pisa's `if`/`from` direct jumps into paired Pendulum branches.
Anything other than `faithful` is an adapter transformation and is marked as
such in the report.  ``both`` runs faithful and pendulum-cf.

Examples:
    python3 tools/difftest_pisa.py janus tests/difftest_corpus/*.janus --mode both
    python3 tools/difftest_pisa.py rfcl --rfcl ~/rfcl ~/rfcl/examples/*.rl
    python3 tools/difftest_pisa.py pal ~/Rlang-compiler/test/sch/sch.pal --max-steps 3000000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from lexer import tokenize, LexError                       # noqa: E402
from parser import parse, ParseError                       # noqa: E402
from codegen import compile_program, CodeGenError          # noqa: E402
from inverse import invert_program                         # noqa: E402
from pisa import print_program, DATA, LabeledInstr         # noqa: E402
from pisa_interp import PISAMachine, PISAError             # noqa: E402
import pisa2pal                                            # noqa: E402
import pal2pisa                                            # noqa: E402

DEFAULT_PHPISA = os.environ.get("PHPISA_DIR", os.path.join(ROOT, "..", "phpisa"))
DEFAULT_RFCL = os.environ.get("RFCL_DIR", os.path.join(ROOT, "..", "rfcl"))
PHP = shutil.which("php")


# ---------------------------------------------------------------------------
# phpisa runner
# ---------------------------------------------------------------------------

@dataclass
class PhpResult:
    ok: bool
    finished: bool
    error: Optional[str]
    regs: List[object]
    mem: Dict[int, object]
    pc: int
    br: int
    dir: int
    output: List[str]
    stderr: str
    steps_exceeded: bool = False


def run_phpisa(pal_text: str, phpisa_dir: str, keep_path: Optional[str] = None,
               init_regs: Optional[Dict[str, int]] = None, max_steps: int = 5_000_000) -> PhpResult:
    if PHP is None:
        return PhpResult(False, False, "php executable not found", [], {}, 0, 0, 0, [], "")
    if keep_path:
        path = keep_path
        with open(path, "w") as f:
            f.write(pal_text)
    else:
        fd, path = tempfile.mkstemp(suffix=".pal")
        with os.fdopen(fd, "w") as f:
            f.write(pal_text)
    cmd = [PHP, os.path.join(HERE, "phpisa_dump.php"), "--phpisa", phpisa_dir, "--max-steps", str(max_steps)]
    if init_regs:
        cmd += ["--regs", json.dumps(init_regs)]
    cmd.append(path)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    finally:
        if not keep_path:
            os.unlink(path)
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        msg = (p.stdout + p.stderr).strip().replace("\n", " | ")[:300]
        return PhpResult(False, False, f"phpisa produced no JSON: {msg}", [], {}, 0, 0, 0, [], p.stderr)
    if not d.get("ok", False) and "registers" not in d:
        return PhpResult(False, False, d.get("error"), [], {}, 0, 0, 0, [], p.stderr)
    regs = [d["registers"].get(f"R{i}", 0) for i in range(32)]
    mem = {int(k): v for k, v in d["mem"].items()}
    out = [l for l in d["output"].splitlines() if l and not l.startswith("Pendulum Terminated")]
    return PhpResult(d["ok"], d["finished"], d.get("error"), regs, mem, d["pc"], d["br"], d["dir"], out,
                     p.stderr.strip(), d.get("steps_exceeded", False))


# ---------------------------------------------------------------------------
# pisa_interp runner
# ---------------------------------------------------------------------------

@dataclass
class PisaResult:
    ok: bool
    error: Optional[str]
    regs: List[int]
    mem: Dict[int, int]
    br: int
    dirty: List[str]
    steps: int


def run_pisa(code: List[LabeledInstr], init_mem: Optional[Dict[int, int]] = None,
             max_steps: int = 5_000_000) -> PisaResult:
    try:
        m = PISAMachine(code)
    except PISAError as e:
        return PisaResult(False, f"load: {e}", [], {}, 0, [], 0)
    m.check_clean = False
    if init_mem:
        m.mem.update(init_mem)
    try:
        m.run(max_steps=max_steps)
    except PISAError as e:
        return PisaResult(False, str(e), list(m.regs), dict(m.mem), m.br, [], m._steps)
    dirty = [f"r{i}={v}" for i, v in enumerate(m.regs) if i >= 3 and v != 0]
    return PisaResult(True, None, list(m.regs), dict(m.mem), m.br, dirty, m._steps)


# ---------------------------------------------------------------------------
# janus sub-command
# ---------------------------------------------------------------------------

@dataclass
class ProgResult:
    name: str
    mode: str
    status: str                       # OK / SKIPPED / PARSE-ERROR / COMPILE-ERROR
    fwd: str = "-"
    bwd: str = "-"
    rt_pisa: str = "-"
    rt_php: str = "-"
    notes: List[str] = field(default_factory=list)
    detail: List[str] = field(default_factory=list)
    size_fwd: int = 0
    size_pal: int = 0


def _data_words(code: List[LabeledInstr]) -> List[int]:
    vals = []
    for li in code:
        if isinstance(li.instr, DATA):
            vals.append(li.instr.value)
        else:
            break
    return vals


def _patch_data(ir: List[pisa2pal.Ins], mem: Dict[int, int]) -> List[pisa2pal.Ins]:
    out, k = [], 0
    for ins in ir:
        if ins.op == "DATA":
            out.append(ins.clone(args=[str(mem.get(k, int(ins.args[0])))]))
            k += 1
        else:
            out.append(ins)
    return out


def _stack_base(ir: List[pisa2pal.Ins]) -> Optional[int]:
    for i, ins in enumerate(ir):
        if ins.op == "START" and i + 1 < len(ir) and ir[i + 1].op == "ADDI" and ir[i + 1].args[0] == "r1":
            return int(ir[i + 1].args[1])
    return None


def _fmt(v) -> str:
    return repr(v) if isinstance(v, float) else str(v)


def compare(pisa: PisaResult, php: PhpResult, nvars: int, base_pisa: Optional[int],
            base_php: Optional[int]) -> Tuple[str, List[str]]:
    """Return (status, detail lines)."""
    detail: List[str] = []
    if not pisa.ok and (not php.ok or not php.finished):
        return (f"ERROR(both: pisa_interp: {pisa.error}; phpisa: {php.error or 'did not reach FINISH'})", detail)
    if not pisa.ok:
        return (f"ERROR(pisa_interp: {pisa.error})", detail)
    if not php.ok or not php.finished:
        why = php.error or f"did not reach FINISH (pc={php.pc}, br={php.br}, dir={php.dir})"
        return (f"ERROR(phpisa: {why})", detail)
    if pisa.dirty:
        # codegen jumps to FINISH as soon as a `fi` or `from` assertion fails,
        # skipping epilogues, so the final registers are not comparable.
        return (f"ERROR(pisa_interp: assertion violated, {', '.join(pisa.dirty)} at FINISH)", detail)
    for a in range(nvars):
        pv, hv = pisa.mem.get(a, 0), php.mem.get(a, 0)
        if pv != hv:
            return (f"MISMATCH(mem[{a}]: pisa_interp={_fmt(pv)} phpisa={_fmt(hv)})", detail)
    for i in range(32):
        pv, hv = pisa.regs[i], php.regs[i]
        if i == 1 and base_pisa is not None and base_php is not None and hv == base_php and pv == base_pisa:
            continue                                    # stack base differs by construction
        if pv != hv:
            return (f"MISMATCH(r{i}: pisa_interp={_fmt(pv)} phpisa={_fmt(hv)})", detail)
    if pisa.br != php.br:
        detail.append(f"note: final br differs (pisa_interp {pisa.br}, phpisa {php.br}); not counted")
    return ("OK", detail)


def difftest_janus(path: str, mode: str, phpisa_dir: str, keep: Optional[str], max_steps: int) -> ProgResult:
    name = os.path.splitext(os.path.basename(path))[0]
    res = ProgResult(name, mode, "OK")
    with open(path) as f:
        src = f.read()
    first = src.splitlines()[0] if src else ""
    m = re.match(r"\s*//\s*SKIP:\s*(.*)", first)
    if m:
        res.status = "SKIPPED"
        res.notes.append(m.group(1).strip())
        return res
    try:
        prog = parse(tokenize(src))
    except (LexError, ParseError) as e:
        res.status = "PARSE-ERROR"; res.notes.append(str(e)); return res
    try:
        code_f = compile_program(prog)
        code_b = compile_program(invert_program(prog))
    except CodeGenError as e:
        res.status = "COMPILE-ERROR"; res.notes.append(str(e)); return res
    res.size_fwd = len(code_f)
    nvars = sum(vd.size for vd in prog.vars)
    init = {i: v for i, v in enumerate(_data_words(code_f))}

    def php_run(code, init_mem=None, tag=""):
        ir = pisa2pal.parse_pyjanus(print_program(code))
        if init_mem is not None:
            ir = _patch_data(ir, init_mem)
        base = _stack_base(ir)
        try:
            conv = pisa2pal.convert(ir, pendulum_calls=(mode != "faithful"),
                                    pendulum_cf=(mode == "pendulum-cf"))
        except pisa2pal.ConvertError as e:
            return None, base, None, f"adapter: {e}", []
        keep_path = os.path.join(keep, f"{name}.{mode}{tag}.pal") if keep else None
        r = run_phpisa(conv.pal, phpisa_dir, keep_path, max_steps=max_steps)
        new_base = conv.lowering.new_stack_base if conv.lowering and conv.lowering.new_stack_base else base
        return r, base, new_base, None, conv.notes

    if keep:
        with open(os.path.join(keep, f"{name}.fwd.pisa"), "w") as f:
            f.write(print_program(code_f) + "\n")
        with open(os.path.join(keep, f"{name}.inv.pisa"), "w") as f:
            f.write(print_program(code_b) + "\n")

    # ---- forward ----
    pf = run_pisa(code_f, max_steps=max_steps)
    hf, base, nbase, err, notes = php_run(code_f, tag=".fwd")
    if err:
        res.fwd = f"ERROR(phpisa-side {err})"
    else:
        res.fwd, d = compare(pf, hf, nvars, base, nbase); res.detail += d
        res.size_pal = hf and len(hf.mem) or 0
        if hf.stderr:
            res.detail.append("phpisa stderr: " + hf.stderr.splitlines()[0][:160])
    if pf.dirty:
        res.detail.append(f"pisa_interp: garbage at FINISH (fwd): {', '.join(pf.dirty)}")
    res.notes += [n for n in notes if "scratch" in n or "split" in n or "large" in n]

    # ---- backward (the inverse program from the same initial state) ----
    pb = run_pisa(code_b, max_steps=max_steps)
    hb, base_b, nbase_b, err, _ = php_run(code_b, tag=".inv")
    if err:
        res.bwd = f"ERROR(phpisa-side {err})"
    else:
        res.bwd, d = compare(pb, hb, nvars, base_b, nbase_b); res.detail += d
    if pb.dirty:
        res.detail.append(f"pisa_interp: garbage at FINISH (bwd): {', '.join(pb.dirty)}")

    # ---- round trip on pisa_interp ----
    if pf.ok:
        pr = run_pisa(code_b, init_mem={a: pf.mem.get(a, 0) for a in range(nvars)}, max_steps=max_steps)
        if not pr.ok:
            res.rt_pisa = f"ERROR({pr.error})"
        else:
            bad = [a for a in range(nvars) if pr.mem.get(a, 0) != init.get(a, 0)]
            res.rt_pisa = "OK" if not bad else f"MISMATCH(mem[{bad[0]}]={pr.mem.get(bad[0], 0)} expected {init.get(bad[0], 0)})"
            if pr.dirty:
                res.detail.append(f"pisa_interp: garbage at FINISH (round trip): {', '.join(pr.dirty)}")
    else:
        res.rt_pisa = "ERROR(fwd failed)"

    # ---- round trip on phpisa ----
    if hf is not None and hf.ok and hf.finished:
        hr, _, _, err, _ = php_run(code_b, init_mem={a: hf.mem.get(a, 0) for a in range(nvars)}, tag=".rt")
        if err:
            res.rt_php = f"ERROR({err})"
        elif not hr.finished:
            res.rt_php = f"ERROR(phpisa: {hr.error or 'did not reach FINISH'})"
        else:
            bad = [a for a in range(nvars) if hr.mem.get(a, 0) != init.get(a, 0)]
            res.rt_php = "OK" if not bad else f"MISMATCH(mem[{bad[0]}]={_fmt(hr.mem.get(bad[0], 0))} expected {init.get(bad[0], 0)})"
    else:
        res.rt_php = "ERROR(fwd failed)"
    return res


# ---------------------------------------------------------------------------
# rfcl sub-command
# ---------------------------------------------------------------------------

def _rl_interface(src: str) -> Tuple[List[str], List[str]]:
    for line in src.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        groups = re.findall(r"\(([^)]*)\)", s)
        if len(groups) >= 2:
            return groups[0].split(), groups[1].split()
        break
    raise ValueError("cannot find the (inputs) (outputs) (temps) header")


def difftest_rfcl(path: str, rfcl_dir: str, phpisa_dir: str, inputs: Optional[List[int]],
                  keep: Optional[str], max_steps: int) -> ProgResult:
    name = os.path.splitext(os.path.basename(path))[0]
    res = ProgResult(name, "rfcl", "OK")
    with open(path) as f:
        src = f.read()
    try:
        in_names, out_names = _rl_interface(src)
    except ValueError as e:
        res.status = "PARSE-ERROR"; res.notes.append(str(e)); return res
    vals = inputs if inputs is not None else [3, 1, 2, 5, 4][:len(in_names)]
    if len(vals) != len(in_names):
        res.status = "ERROR"; res.notes.append(f"need {len(in_names)} inputs {in_names}, got {vals}"); return res
    env = dict(os.environ, PYTHONPATH=rfcl_dir)
    p = subprocess.run([sys.executable, "-m", "pyrev_fl.cli", "rl-to-pisa", os.path.abspath(path)],
                       capture_output=True, text=True, cwd=rfcl_dir, env=env)
    if p.returncode != 0:
        res.status = "COMPILE-ERROR"; res.notes.append(p.stderr.strip()[-200:]); return res
    pisa_text = p.stdout
    m = re.search(r";\s*Variables:\s*(.*)", pisa_text)
    regmap = dict(kv.split("=") for kv in re.split(r",\s*", m.group(1).strip())) if m else {}
    o = subprocess.run([sys.executable, "-m", "pyrev_fl.cli", "rl-run", os.path.abspath(path), *map(str, vals),
                        "--json"], capture_output=True, text=True, cwd=rfcl_dir, env=env)
    if o.returncode != 0:
        res.status = "ORACLE-ERROR"; res.notes.append(o.stderr.strip()[-200:]); return res
    oracle = json.loads(o.stdout)["outputs"]
    res.notes.append("inputs " + ", ".join(f"{n}={v}" for n, v in zip(in_names, vals)))
    try:
        ir = pisa2pal.parse_rfcl(pisa_text)
        conv = pisa2pal.convert(ir)
    except pisa2pal.ConvertError as e:
        res.fwd = f"ERROR(adapter: {e})"; res.bwd = "-"
        return res
    res.notes += conv.notes
    init_regs = {regmap[n]: v for n, v in zip(in_names, vals) if n in regmap}
    keep_path = os.path.join(keep, f"rfcl_{name}.pal") if keep else None
    if keep:
        with open(os.path.join(keep, f"rfcl_{name}.pisa"), "w") as f:
            f.write(pisa_text)
    hr = run_phpisa(conv.pal, phpisa_dir, keep_path, init_regs=init_regs, max_steps=max_steps)
    if not hr.ok or not hr.finished:
        res.fwd = f"ERROR(phpisa: {hr.error or f'did not reach FINISH (pc={hr.pc}, br={hr.br}, dir={hr.dir})'})"
    else:
        bad = [(n, oracle[n], hr.regs[int(regmap[n][1:])]) for n in out_names
               if n in regmap and oracle.get(n) != hr.regs[int(regmap[n][1:])]]
        res.fwd = "OK" if not bad else f"MISMATCH({bad[0][0]}: rl-run={bad[0][1]} phpisa={_fmt(bad[0][2])})"
    # the same PAL on PISAMachine
    try:
        mach, ld = pal2pisa.load(conv.pal)
        for r, v in init_regs.items():
            mach.regs[int(r[1:])] = v
        mach.run(max_steps=max_steps)
        bad = [(n, oracle[n], mach.regs[int(regmap[n][1:])]) for n in out_names
               if n in regmap and oracle.get(n) != mach.regs[int(regmap[n][1:])]]
        res.bwd = "OK" if not bad else f"MISMATCH({bad[0][0]}: rl-run={bad[0][1]} pisa_interp={bad[0][2]})"
    except (pal2pisa.LoadError, PISAError) as e:
        res.bwd = f"ERROR(pisa_interp: {e})"
    return res


# ---------------------------------------------------------------------------
# pal sub-command
# ---------------------------------------------------------------------------

def difftest_pal(path: str, phpisa_dir: str, expect: Optional[str], max_steps: int) -> Dict:
    with open(path) as f:
        text = f.read()
    out: Dict = {"file": path}
    hr = run_phpisa(text, phpisa_dir, max_steps=max_steps)
    out["phpisa"] = {"finished": hr.finished, "error": hr.error, "steps_exceeded": hr.steps_exceeded,
                     "outputs": len(hr.output), "pc": hr.pc, "br": hr.br, "dir": hr.dir,
                     "nonzero_regs": {f"R{i}": v for i, v in enumerate(hr.regs) if v}}
    if expect:
        with open(expect) as f:
            exp_lines = [l.strip() for l in f if re.match(r"\s*R\d+ = ", l)]
        out["expected_output_lines"] = len(exp_lines)
        if exp_lines:
            n = min(len(exp_lines), len(hr.output))
            first_bad = next((i for i in range(n) if exp_lines[i] != hr.output[i]), None)
            out["output_matches_expected_prefix"] = first_bad is None and len(hr.output) >= len(exp_lines)
            out["first_output_difference"] = first_bad
    try:
        mach, ld = pal2pisa.load(text)
        out["pal2pisa_notes"] = ld.notes
        out["pisa_interp_lacks"] = ld.unsupported
        try:
            mach.run(max_steps=max_steps)
            status = "FINISH"
        except PISAError as e:
            status = str(e)
        out["pisa_interp"] = {"status": status, "steps": mach._steps, "outputs": len(mach.output),
                              "nonzero_regs": {f"r{i}": v for i, v in enumerate(mach.regs) if v}}
        n = min(len(mach.output), len(hr.output))
        first_bad = next((i for i in range(n) if mach.output[i] != hr.output[i]), None)
        out["first_output_difference_pisa_vs_php"] = (
            None if first_bad is None else {"index": first_bad, "pisa_interp": mach.output[first_bad],
                                            "phpisa": hr.output[first_bad]})
        out["common_output_prefix"] = n if first_bad is None else first_bad
        out["phpisa_first_outputs"] = hr.output[:6]
        out["pisa_interp_first_outputs"] = mach.output[:6]
    except pal2pisa.LoadError as e:
        out["pisa_interp"] = {"status": f"load error: {e}"}
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _short(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def print_table(results: List[ProgResult]) -> None:
    hdr = f"{'program':28} {'mode':14} {'fwd':40} {'bwd':40} {'rt-pisa':12} {'rt-php':12}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        if r.status != "OK":
            print(f"{r.name:28} {r.mode:14} {r.status}: {'; '.join(r.notes)}")
            continue
        print(f"{r.name:28} {r.mode:14} {_short(r.fwd, 40):40} {_short(r.bwd, 40):40} "
              f"{_short(r.rt_pisa, 12):12} {_short(r.rt_php, 12):12}")
    print()
    for r in results:
        if r.status == "OK" and (r.detail or r.notes or "OK" not in (r.fwd, r.bwd)):
            print(f"[{r.name} / {r.mode}]")
            for k in ("fwd", "bwd", "rt_pisa", "rt_php"):
                v = getattr(r, k)
                if v != "OK":
                    print(f"    {k}: {v}")
            for d in r.detail:
                print(f"    {d}")
            for n in r.notes:
                print(f"    note: {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("janus", "rfcl", "pal"):
        sp = sub.add_parser(c)
        sp.add_argument("files", nargs="+")
        sp.add_argument("--phpisa", default=DEFAULT_PHPISA, help="phpisa checkout (default: $PHPISA_DIR or ../phpisa)")
        sp.add_argument("--max-steps", type=int, default=5_000_000)
        sp.add_argument("--json", help="write results as JSON to this file")
        sp.add_argument("--keep", help="directory in which to keep generated .pisa/.pal files")
    sub.choices["janus"].add_argument("--mode", choices=["faithful", "pendulum-calls", "pendulum-cf", "both"],
                                      default="both")
    sub.choices["rfcl"].add_argument("--rfcl", default=DEFAULT_RFCL, help="rfcl checkout (default: $RFCL_DIR or ../rfcl)")
    sub.choices["rfcl"].add_argument("--inputs", help="comma-separated input values (default: 3,1,2,...)")
    sub.choices["pal"].add_argument("--expect", help="file with expected 'Rn = v' OUTPUT lines")
    args = ap.parse_args(argv)

    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
    if PHP is None:
        print("warning: php not found; phpisa side will report ERROR", file=sys.stderr)

    if args.cmd == "janus":
        modes = ["faithful", "pendulum-cf"] if args.mode == "both" else [args.mode]
        results: List[ProgResult] = []
        for f in args.files:
            for mode in modes:
                results.append(difftest_janus(f, mode, args.phpisa, args.keep, args.max_steps))
        print_table(results)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump([asdict(r) for r in results], fh, indent=1)
        return 0
    if args.cmd == "rfcl":
        inputs = [int(v) for v in args.inputs.split(",")] if args.inputs else None
        results = [difftest_rfcl(f, args.rfcl, args.phpisa, inputs, args.keep, args.max_steps) for f in args.files]
        hdr = f"{'program':20} {'phpisa vs rl-run':44} {'pisa_interp vs rl-run':44}"
        print(hdr); print("-" * len(hdr))
        for r in results:
            if r.status != "OK":
                print(f"{r.name:20} {r.status}: {'; '.join(r.notes)}"); continue
            print(f"{r.name:20} {_short(r.fwd, 44):44} {_short(r.bwd, 44):44}")
        print()
        for r in results:
            print(f"[{r.name}]")
            for k in ("fwd", "bwd"):
                v = getattr(r, k)
                if v not in ("OK", "-"):
                    print(f"    {k}: {v}")
            for n in r.notes:
                print(f"    note: {n}")
        if args.json:
            with open(args.json, "w") as fh:
                json.dump([asdict(r) for r in results], fh, indent=1)
        return 0
    if args.cmd == "pal":
        outs = [difftest_pal(f, args.phpisa, args.expect, args.max_steps) for f in args.files]
        print(json.dumps(outs, indent=1, default=str))
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(outs, fh, indent=1, default=str)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
