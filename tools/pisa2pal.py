#!/usr/bin/env python3
"""pisa2pal: convert PISA assembly dialects to phpisa's PAL input format.

Input dialects
--------------
  pyjanus  (default)  the text printed by janus2pisa.py / pisa.print_program:
                      ``label: OP rN ...``, registers ``r0..r31``, leading
                      ``DATA`` words, ``SLTX``/``ORX``/``ANDX`` compiler
                      pseudo-instructions, ``start: START`` / ``finish: FINISH``.
  rfcl                the "PISA-flavoured" 3-operand comma dialect printed by
                      ``python3 -m pyrev_fl.cli rl-to-pisa`` (rfcl):
                      ``XOR R2, R1, R0``, ``ADD R2, #1, R0``, ``EXCH R3, R4``
                      (register swap), ``BEQ R1, R0, label``, ``HALT``.

Output: PAL as read by phpisa (``php bin/phpisa file.pal``): ``$N`` registers,
2-operand mnemonics, ``.start <label>``, ``FINISH``.

Every mapping decision is listed in docs/DIFFTEST.md ("dialect differences").
The important ones:

  * ``SUBI rd c``           -> ``ADDI $d -c``            (phpisa has no SUBI)
  * ``SLTX rd rs rt``       -> Pendulum branch idiom     (phpisa has no SLTX;
                               ``SUB rs rt; A: BGEZ rs B; XORI rd 1; B: BGEZ rs A; ADD rs rt``)
  * ``ORX rd rs`` (pyjanus: rd |= rs; rs := 0)
                            -> ``ANDX $d $d $s; XOR $d $s; XOR $s $s``   (exact)
  * ``ANDX rd1 rd2 rs`` (pyjanus: rd1 ^= rd2 & rs; rd2 := 0)
                            -> ``ANDX $d1 $d2 $s; XOR $d2 $d2``         (exact)
  * immediates are 11-bit two's complement in phpisa (encoder truncates
    silently!) so ``ADDI``/``XORI`` with |c| > 1023 are expanded: small ones
    into several ``ADDI``, large ones (< 2^32) through a scratch register that
    is never mentioned by the program (``ADDI``/``RL 10`` chunks, then
    ``ADD``/``SUB``/``XOR`` and the exact un-computation).
  * branch offsets are also 11-bit: an out-of-range branch is an error.
  * ``r0`` is a hard-wired zero in pisa_interp but an ordinary register in
    phpisa: any write to r0 other than the ``ADDI r0 0`` NOP is an error.
  * ``--pendulum-calls``: rewrite janus2pisa's 6-instruction procedure prologue
    (which is a no-op on pisa_interp and is NOT valid Pendulum code, since the
    branch target is a data instruction) into Axelsen's protocol
    ``f: SWAPBR r2; NEG r2; SUBI r1 1; EXCH r2 r1 ... EXCH r2 r1; ADDI r1 1``
    and move the stack above the code (phpisa's memory IS the program array,
    so janus2pisa's stack at address nvars+3 would overwrite instructions).
    This is an ADAPTER TRANSFORMATION; results obtained with it are marked.

No dependencies; Python 3.10+.  Usage:

    python3 tools/pisa2pal.py prog.pisa -o prog.pal
    python3 tools/pisa2pal.py --dialect rfcl prog.pisa -o prog.pal
    python3 tools/pisa2pal.py --pendulum-calls prog.pisa
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

IMM_MIN, IMM_MAX = -1024, 1023          # phpisa: sign_extend11 / encode(..., imm & 0x7FF)
SPLIT_LIMIT = 8 * IMM_MAX               # up to 8 ADDIs before we resort to a scratch register
NUM_REGS = 32


class ConvertError(Exception):
    pass


@dataclass
class Ins:
    """One instruction of the neutral IR (pyjanus2pisa operand order)."""
    label: Optional[str]
    op: str
    args: List[str]
    src: str = ""

    def clone(self, **kw) -> "Ins":
        d = dict(label=self.label, op=self.op, args=list(self.args), src=self.src)
        d.update(kw)
        return Ins(**d)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_PYJ_LINE = re.compile(r"^\s*(?:([A-Za-z_][A-Za-z0-9_]*)\s*:)?\s*([A-Za-z]+)\s*(.*?)\s*$")


def parse_pyjanus(text: str) -> List[Ins]:
    """Parse janus2pisa's printed assembly into the IR."""
    out: List[Ins] = []
    pending_label: Optional[str] = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$", line)
        if m:                                   # label on its own line
            pending_label = m.group(1)
            continue
        m = _PYJ_LINE.match(line)
        if not m:
            raise ConvertError(f"line {lineno}: cannot parse {raw!r}")
        label, op, rest = m.group(1), m.group(2).upper(), m.group(3)
        if label is None and pending_label is not None:
            label = pending_label
        pending_label = None
        args = rest.split()
        out.append(Ins(label, op, args, raw))
    if pending_label is not None:
        raise ConvertError(f"dangling label {pending_label!r} at end of input")
    return out


def _rfcl_reg(tok: str, lineno: int) -> str:
    m = re.fullmatch(r"[Rr\$](\d+)", tok)
    if not m:
        raise ConvertError(f"line {lineno}: expected a register, got {tok!r} "
                           f"(rfcl inline expressions such as '(R1 + #2)' are not PISA)")
    return "r" + m.group(1)


def parse_rfcl(text: str) -> List[Ins]:
    """Parse rfcl's ``rl-to-pisa`` output into the IR.

    ``OP Ra, Rb, R0`` -> 2-operand ``OP ra rb`` (third operand must be R0);
    ``OP Ra, #c, R0`` -> ``OPI ra c``; ``EXCH Ra, Rb`` (register swap!) ->
    ``REGSWAP ra rb``; ``HALT`` -> ``FINISH``.  Labels stand on their own line
    and attach to the next instruction.  Execution starts at the first label
    (rfcl's ``entry`` block), recorded as a ``START`` pseudo-instruction that
    the emitter turns into ``.start``.
    """
    out: List[Ins] = []
    pending: Optional[str] = None
    first_label: Optional[str] = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_\-]*)\s*:", line)
        if m:
            lab = m.group(1).replace("-", "_")       # phpisa labels: no '-'
            if pending is not None:
                raise ConvertError(f"line {lineno}: two labels ({pending}, {lab}) on one instruction")
            pending = lab
            if first_label is None:
                first_label = lab
            continue
        parts = [p.strip() for p in re.split(r"[,\s]+", line) if p.strip()]
        op, args = parts[0].upper(), parts[1:]
        label, pending = pending, None
        if op == "HALT":
            out.append(Ins(label, "FINISH", [], raw))
            continue
        if op in ("ADD", "SUB", "XOR"):
            if len(args) != 3:
                raise ConvertError(f"line {lineno}: {op} expects 3 operands: {raw!r}")
            if args[2].upper() not in ("R0", "$0"):
                raise ConvertError(f"line {lineno}: third operand of {op} must be R0: {raw!r}")
            rd = _rfcl_reg(args[0], lineno)
            if args[1].startswith("#"):
                c = int(args[1][1:])
                out.append(Ins(label, op + "I", [rd, str(c)], raw))
            else:
                out.append(Ins(label, op, [rd, _rfcl_reg(args[1], lineno)], raw))
            continue
        if op == "EXCH":                         # rfcl: swap two REGISTERS
            if len(args) != 2:
                raise ConvertError(f"line {lineno}: EXCH expects 2 registers: {raw!r}")
            out.append(Ins(label, "REGSWAP", [_rfcl_reg(args[0], lineno), _rfcl_reg(args[1], lineno)], raw))
            continue
        if op in ("BEQ", "BNE"):
            if len(args) != 3:
                raise ConvertError(f"line {lineno}: {op} expects 3 operands: {raw!r}")
            out.append(Ins(label, op, [_rfcl_reg(args[0], lineno), _rfcl_reg(args[1], lineno),
                                       args[2].replace("-", "_")], raw))
            continue
        if op in ("BLT", "BGT", "BLE", "BGE"):
            raise ConvertError(f"line {lineno}: {op} (register-register compare-and-branch) has no "
                               f"phpisa/Pendulum equivalent: {raw!r}")
        if op in ("BRA", "RBRA"):
            out.append(Ins(label, op, [args[0].replace("-", "_")], raw))
            continue
        raise ConvertError(f"line {lineno}: unknown rfcl instruction {op}: {raw!r}")
    if pending is not None:
        raise ConvertError(f"dangling label {pending!r} at end of input")
    if first_label is None:
        raise ConvertError("rfcl program has no label (no entry block)")
    # Mark the entry point: START before the first instruction (shares its label).
    entry = out[0]
    if entry.label != first_label:
        raise ConvertError("first instruction does not carry the first label")
    out.insert(0, Ins(first_label, "START", [], "; entry"))
    entry.label = None
    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _regnum(r: str) -> int:
    m = re.fullmatch(r"[rR\$](\d+)", r)
    if not m:
        raise ConvertError(f"bad register {r!r}")
    n = int(m.group(1))
    if not 0 <= n < NUM_REGS:
        raise ConvertError(f"register {r} out of range (phpisa has R0..R31)")
    return n


def _pal_reg(r: str) -> str:
    return f"${_regnum(r)}"


def used_registers(ir: List[Ins]) -> set:
    used = set()
    for ins in ir:
        for a in ins.args:
            if re.fullmatch(r"[rR\$]\d+", a):
                used.add(_regnum(a))
    return used


def pick_scratch(ir: List[Ins]) -> Optional[int]:
    used = used_registers(ir)
    for n in range(NUM_REGS - 1, 2, -1):
        if n not in used:
            return n
    return None


def _in_imm(c: int) -> bool:
    return IMM_MIN <= c <= IMM_MAX


# --------------------------------------------------------------------------
# --pendulum-calls lowering (adapter transformation, see module docstring)
# --------------------------------------------------------------------------

_PROLOGUE = [("SUBI", ["r1", "1"]), ("EXCH", ["r2", "r1"]), ("SWAPBR", ["r2"]),
             ("NEG", ["r2"]), ("EXCH", ["r2", "r1"]), ("ADDI", ["r1", "1"])]


@dataclass
class LoweringInfo:
    procedures: List[str] = field(default_factory=list)
    old_stack_base: Optional[int] = None
    new_stack_base: Optional[int] = None


def pendulum_calls(ir: List[Ins], new_stack_base: int) -> Tuple[List[Ins], LoweringInfo]:
    """Rewrite janus2pisa procedure prologues into Axelsen's Pendulum protocol
    and relocate the stack base.  Returns (new_ir, info)."""
    info = LoweringInfo()
    labels = {ins.label: i for i, ins in enumerate(ir) if ins.label}
    procs = [l for l in labels if l + "_top" in labels and l + "_bot" in labels]
    out: List[Ins] = []
    i = 0
    while i < len(ir):
        ins = ir[i]
        if ins.label in procs:
            window = ir[i:i + 6]
            ok = len(window) == 6 and all(w.op == op and w.args == args
                                          for w, (op, args) in zip(window, _PROLOGUE))
            if not ok:
                raise ConvertError(f"procedure {ins.label}: prologue does not match janus2pisa's "
                                   f"6-instruction pattern; cannot lower")
            f = ins.label
            info.procedures.append(f)
            out.append(Ins(f, "SWAPBR", ["r2"], "; pendulum-calls prologue"))
            out.append(Ins(None, "NEG", ["r2"], ""))
            out.append(Ins(None, "SUBI", ["r1", "1"], ""))
            out.append(Ins(None, "EXCH", ["r2", "r1"], ""))
            i += 6
            continue
        if ins.label and ins.label.endswith("_bot") and ins.label[:-4] in procs:
            out.append(Ins(None, "EXCH", ["r2", "r1"], "; pendulum-calls epilogue"))
            out.append(Ins(None, "ADDI", ["r1", "1"], ""))
        out.append(ins)
        i += 1
    # stack relocation: `start: START ; ADDI r1 K ; BRA main` and `finish: FINISH ; SUBI r1 K`
    for j, ins in enumerate(out):
        if ins.op == "START" and j + 1 < len(out) and out[j + 1].op == "ADDI" and out[j + 1].args[0] == "r1":
            info.old_stack_base = int(out[j + 1].args[1])
            out[j + 1] = out[j + 1].clone(args=["r1", str(new_stack_base)])
        if ins.op == "FINISH" and j + 1 < len(out) and out[j + 1].op == "SUBI" and out[j + 1].args[0] == "r1":
            out[j + 1] = out[j + 1].clone(args=["r1", str(new_stack_base)])
    if info.old_stack_base is not None:
        info.new_stack_base = new_stack_base
    return out, info


# --------------------------------------------------------------------------
# --pendulum-cf lowering: janus2pisa's `if` / `from` direct jumps -> paired branches
# --------------------------------------------------------------------------
#
# janus2pisa's codegen (codegen.py `_gen_if`, `_gen_from`) jumps to DATA
# instructions (`BRA if_assert_K`, `BRA from_do_Z`, `BEQ rt r0 from_loop_L`,
# `BRA from_exit_E`), which pisa_interp executes as direct jumps but which a
# Pendulum machine (phpisa) cannot: the branch register stays non-zero and the
# next instruction jumps again.  The rewrite below inserts, at every such
# landing site, a conditional branch that cancels the incoming offset exactly
# when control arrived by the jump (Axelsen CC 2011, Fig. 11/12 style), using
# the compiler's own path flag rt:
#
#   if:    else path arrives at if_assert with rt = 0, then path falls through
#          with rt = 1   ->  `if_assert: BEQ rt r0 <the BRA if_assert>`
#   from:  `from_loop: BEQ rt r0 <the BEQ that jumps here>`      (rt = 0 there)
#          `from_exit: BRA <the BRA that jumps here>`            (paired BRAs)
#          `from_do:   BNE rt r0 <the BRA from_do>; XOR rt rt`   (rt = 1 from the
#          loop body, 0 on first entry; the re-entry check `XORI rt 1; BNE rt
#          r0 finish` that codegen emits just before `BRA from_do` is dropped
#          so that rt discriminates the paths.  Under Pendulum semantics a
#          violated re-entry assertion then leaves BR non-zero, as in Axelsen)
#
# This is an ADAPTER TRANSFORMATION (results marked "pendulum-cf").  The
# lowered code is phpisa-only: pisa_interp would treat the inserted
# conditionals as direct jumps and loop forever.

@dataclass
class CFInfo:
    ifs: int = 0
    loops: int = 0


def pendulum_cf(ir: List[Ins]) -> Tuple[List[Ins], CFInfo]:
    """Insert cancelling branches at janus2pisa's direct-jump landing sites.

    Landing sites are never relabelled: each inserted instruction gets a fresh
    label and the jumping instruction is retargeted to it.  This matters
    because codegen's `remove_nops` forwards the label of a removed NOP to the
    next instruction, so e.g. an empty loop's `from_exit` may alias
    `main_bot` (a paired branch) or `from_do` may alias `from_test`.
    """
    info = CFInfo()
    labels = {ins.label: i for i, ins in enumerate(ir) if ins.label}
    inserts: Dict[int, List[Ins]] = {}          # insert BEFORE index
    retarget: Dict[int, str] = {}              # jumping instruction -> new target label
    newlabel: Dict[int, str] = {}              # give an unlabelled instruction a label
    remove: set = set()
    counter = [0]

    def fresh(p: str) -> str:
        counter[0] += 1
        return f"_{p}{counter[0]}"

    def label_of(i: int, prefix: str) -> str:
        if ir[i].label:
            return ir[i].label
        if i not in newlabel:
            newlabel[i] = fresh(prefix)
        return newlabel[i]

    def land(target_idx: int, jumper_idx: int, prefix: str, body: List[Ins]) -> None:
        lab = fresh(prefix)
        body[0] = body[0].clone(label=lab)
        inserts.setdefault(target_idx, []).extend(body)
        retarget[jumper_idx] = lab

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            raise ConvertError(f"pendulum-cf: {msg}")

    # ---- if:  if_end_Q: BRA P ; P: BRA if_end_Q ; P+1 = if_false_M: BRA if_test_N ;
    #           if_test_N: BEQ rt r0 if_false_M ; D = Q-1: BRA if_assert_K
    for e, ins in enumerate(ir):
        if not (ins.label and ins.label.startswith("if_end_") and ins.op == "BRA"):
            continue
        p = labels.get(ins.args[0])
        expect(p is not None and ir[p].op == "BRA" and ir[p].args[0] == ins.label,
               f"{ins.label}: expected paired `if_assert_true: BRA {ins.label}`")
        f = p + 1
        expect(ir[f].label is not None and ir[f].label.startswith("if_false_") and ir[f].op == "BRA",
               f"{ins.label}: expected `if_false: BRA if_test` after if_assert_true")
        t = labels[ir[f].args[0]]
        expect(ir[t].op == "BEQ" and ir[t].args[2] == ir[f].label,
               f"{ins.label}: if_test is not `BEQ rt r0 {ir[f].label}`")
        rt = ir[t].args[0]
        d = e - 1
        expect(ir[d].op == "BRA" and ir[d].args[0] in labels,
               f"{ins.label}: expected `BRA if_assert_K` right before it")
        k = labels[ir[d].args[0]]
        dlab = label_of(d, "ifj")
        land(k, d, "ifa", [Ins(None, "BEQ", [rt, "r0", dlab], "; pendulum-cf: cancel else-path jump")])
        info.ifs += 1

    # ---- from:  A: BEQ rt r0 from_loop_L ; A+1: XORI rt 1 ; A+2: BRA E ;
    #             L: XORI rt 1 ; ... ; D-2: XORI rt 1 ; D-1: BNE rt r0 finish ;
    #             D = E-1: BRA J ; E: <exit>
    for a, ins in enumerate(ir):
        if not (ins.op == "BEQ" and ins.args[1] == "r0" and ins.args[2].startswith("from_loop_")):
            continue
        rt = ins.args[0]
        l = labels[ins.args[2]]
        expect(ir[a + 1].op == "XORI" and ir[a + 1].args == [rt, "1"] and ir[a + 2].op == "BRA",
               f"{ins.args[2]}: expected `XORI rt 1; BRA exit` after the loop test")
        expect(ir[l].op == "XORI" and ir[l].args == [rt, "1"],
               f"{ins.args[2]}: loop body does not start with `XORI {rt} 1`")
        e = labels[ir[a + 2].args[0]]
        d = e - 1
        expect(ir[d].op == "BRA" and ir[d].args[0] in labels,
               f"{ins.args[2]}: expected the back-edge `BRA from_do` right before the exit")
        expect(ir[d - 2].op == "XORI" and ir[d - 2].args == [rt, "1"]
               and ir[d - 1].op == "BNE" and ir[d - 1].args == [rt, "r0", "finish"],
               f"{ins.args[2]}: expected `XORI {rt} 1; BNE {rt} r0 finish` right before the back-edge")
        j = labels[ir[d].args[0]]
        alab = label_of(a, "frt")
        land(l, a, "frl", [Ins(None, "BEQ", [rt, "r0", alab], "; pendulum-cf: cancel loop-test jump")])
        blab = label_of(a + 2, "frx")
        land(e, a + 2, "fre", [Ins(None, "BRA", [blab], "; pendulum-cf: pair with the exit BRA")])
        dlab = label_of(d, "frd")
        remove.update((d - 2, d - 1))
        land(j, d, "frj", [Ins(None, "BNE", [rt, "r0", dlab], "; pendulum-cf: cancel back-edge jump"),
                           Ins(None, "XOR", [rt, rt], "; pendulum-cf: clear flag")])
        info.loops += 1

    out: List[Ins] = []
    for i, ins in enumerate(ir):
        out.extend(inserts.get(i, []))
        if i in remove:
            continue
        if i in newlabel:
            ins = ins.clone(label=newlabel[i])
        if i in retarget:
            ins = ins.clone(args=ins.args[:-1] + [retarget[i]])
        out.append(ins)
    return out, info


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

@dataclass
class Result:
    pal: str
    start_label: Optional[str]
    notes: List[str]
    label_index: Dict[str, int]      # label -> line index in the PAL program (== phpisa address)
    length: int
    scratch: Optional[int]
    lowering: Optional[LoweringInfo] = None
    cf: Optional[CFInfo] = None


class Emitter:
    def __init__(self, ir: List[Ins], emulate_sltx: bool = True, scratch: Optional[int] = None):
        self.ir = ir
        self.emulate_sltx = emulate_sltx
        self.scratch = scratch if scratch is not None else pick_scratch(ir)
        self.lines: List[Tuple[Optional[str], str]] = []   # (label, text)
        self.notes: List[str] = []
        self.counter = 0
        self.scratch_used = False

    # -- low level -------------------------------------------------------
    def emit(self, text: str, label: Optional[str] = None) -> None:
        self.lines.append((label, text))

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"_{prefix}{self.counter}"

    def need_scratch(self, what: str) -> str:
        if self.scratch is None:
            raise ConvertError(f"{what}: needs a scratch register but all 32 are used by the program")
        self.scratch_used = True
        return f"${self.scratch}"

    # -- constants -------------------------------------------------------
    def _chunks32(self, v: int) -> List[int]:
        """Split 0 <= v < 2^32 into 10-bit chunks, most significant first."""
        assert 0 <= v < (1 << 32)
        chunks = [(v >> s) & 0x3FF for s in (30, 20, 10, 0)]
        while len(chunks) > 1 and chunks[0] == 0:
            chunks.pop(0)
        return chunks

    def build_scratch(self, v: int, what: str) -> Tuple[str, List[str]]:
        """Emit code that puts 0 <= v < 2^32 into the scratch register (assumed 0).
        Returns (scratch, uncompute_lines)."""
        if not 0 <= v < (1 << 32):
            raise ConvertError(f"{what}: constant {v} needs a > 32-bit pattern; not expressible "
                               f"with phpisa's 11-bit immediates and 32-bit rotates")
        s = self.need_scratch(what)
        chunks = self._chunks32(v)
        forward: List[str] = []
        for k, ch in enumerate(chunks):
            if ch:
                forward.append(f"ADDI {s} {ch}")
            if k != len(chunks) - 1:
                forward.append(f"RL {s} 10")
        for t in forward:
            self.emit(t)
        undo: List[str] = []
        for t in reversed(forward):
            op, _, rest = t.partition(" ")
            if op == "ADDI":
                r, c = rest.split()
                undo.append(f"ADDI {r} {-int(c)}")
            else:
                undo.append(t.replace("RL ", "RR "))
        return s, undo

    def emit_addi(self, rd: str, c: int, label: Optional[str]) -> None:
        d = _pal_reg(rd)
        if c == 0:
            self.emit(f"ADDI {d} 0", label)
            return
        if _in_imm(c):
            self.emit(f"ADDI {d} {c}", label)
            return
        if abs(c) <= SPLIT_LIMIT:
            self.notes.append(f"ADDI {rd} {c}: immediate outside phpisa's 11-bit range, split into several ADDI")
            sign = 1 if c > 0 else -1
            rest = abs(c)
            first = True
            while rest:
                step = min(rest, IMM_MAX)
                self.emit(f"ADDI {d} {sign * step}", label if first else None)
                first = False
                rest -= step
            return
        self.notes.append(f"ADDI {rd} {c}: large immediate built in scratch register ${self.scratch}")
        s, undo = self.build_scratch(abs(c), f"ADDI {rd} {c}")
        # label goes on the first emitted line
        if label is not None:                      # label goes on the first build line
            first_idx = len(self.lines) - len(undo)
            self.lines[first_idx] = (label, self.lines[first_idx][1])
        self.emit(f"{'ADD' if c > 0 else 'SUB'} {d} {s}")
        for t in undo:
            self.emit(t)

    def emit_xori(self, rd: str, c: int, label: Optional[str]) -> None:
        d = _pal_reg(rd)
        if _in_imm(c):
            self.emit(f"XORI {d} {c}", label)
            return
        self.notes.append(f"XORI {rd} {c}: large immediate built in scratch register ${self.scratch}")
        if c >= 0:
            s, undo = self.build_scratch(c, f"XORI {rd} {c}")
            post = [f"XOR {d} {s}"]
        else:
            # x ^ c = ~(x ^ ~c),  ~c >= 0 ;  ~y = -y - 1  (NEG ; ADDI -1)
            s, undo = self.build_scratch(~c, f"XORI {rd} {c}")
            post = [f"XOR {d} {s}", f"NEG {d}", f"ADDI {d} -1"]
        if label is not None:
            first_idx = len(self.lines) - len(undo)
            self.lines[first_idx] = (label, self.lines[first_idx][1])
        for t in post:
            self.emit(t)
        for t in undo:
            self.emit(t)

    # -- SLTX -----------------------------------------------------------
    def emit_sltx(self, rd: str, rs: str, rt: str, label: Optional[str]) -> None:
        """rd ^= (rs < rt) as a Pendulum branch idiom (see module docstring)."""
        if not self.emulate_sltx:
            raise ConvertError(f"SLTX {rd} {rs} {rt}: phpisa has no SLTX (emulation disabled)")
        d, s, t = _pal_reg(rd), _pal_reg(rs), _pal_reg(rt)
        if rs == rt:
            self.emit(f"ADDI $0 0", label)       # rs < rs is always false: labelled NOP
            return
        if rd in (rs, rt):
            raise ConvertError(f"SLTX {rd} {rs} {rt}: destination aliases a source; not emulable")
        a, b = self.fresh("slt"), self.fresh("slt")
        self.notes.append(f"SLTX {rd} {rs} {rt}: emulated with SUB/BGEZ-pair/ADD (phpisa has no SLTX)")
        if _regnum(rs) == 0:                      # 0 < rt  <=>  rt > 0 : skip the flip when rt <= 0
            a = label or a
            self.emit(f"BLEZ {t} {b}", a)
            self.emit(f"XORI {d} 1")
            self.emit(f"BLEZ {t} {a}", b)
            return
        if _regnum(rt) == 0:                      # rs < 0 : skip the flip when rs >= 0
            a = label or a
            self.emit(f"BGEZ {s} {b}", a)
            self.emit(f"XORI {d} 1")
            self.emit(f"BGEZ {s} {a}", b)
            return
        self.emit(f"SUB {s} {t}", label)
        self.emit(f"BGEZ {s} {b}", a)
        self.emit(f"XORI {d} 1")
        self.emit(f"BGEZ {s} {a}", b)
        self.emit(f"ADD {s} {t}")

    # -- main loop --------------------------------------------------------
    def check_r0_write(self, ins: Ins, rd: str) -> None:
        if _regnum(rd) == 0:
            if ins.op == "ADDI" and int(ins.args[1]) == 0:
                return                            # janus2pisa's labelled NOP
            raise ConvertError(f"{ins.src.strip()!r}: writes r0, which is hard-wired to 0 in "
                               f"pisa_interp but an ordinary register in phpisa")

    def run(self) -> None:
        for ins in self.ir:
            op, a, label = ins.op, ins.args, ins.label
            try:
                if op in ("ADD", "SUB", "XOR"):
                    self.check_r0_write(ins, a[0])
                    self.emit(f"{op} {_pal_reg(a[0])} {_pal_reg(a[1])}", label)
                elif op == "NEG":
                    self.check_r0_write(ins, a[0])
                    self.emit(f"NEG {_pal_reg(a[0])}", label)
                elif op == "ADDI":
                    self.check_r0_write(ins, a[0])
                    self.emit_addi(a[0], int(a[1]), label)
                elif op == "SUBI":
                    self.check_r0_write(ins, a[0])
                    self.emit_addi(a[0], -int(a[1]), label)
                elif op == "XORI":
                    self.check_r0_write(ins, a[0])
                    self.emit_xori(a[0], int(a[1]), label)
                elif op == "ORX":                     # pyjanus: rd |= rs ; rs := 0
                    if len(a) != 2:
                        raise ConvertError(f"{ins.src.strip()!r}: pyjanus ORX takes 2 registers "
                                           f"(phpisa's 3-operand ORX has different semantics)")
                    self.check_r0_write(ins, a[0]); self.check_r0_write(ins, a[1])
                    d, s = _pal_reg(a[0]), _pal_reg(a[1])
                    self.emit(f"ANDX {d} {d} {s}", label)   # d ^= d & s      -> d & ~s
                    self.emit(f"XOR {d} {s}")               # (d & ~s) ^ s    -> d | s
                    self.emit(f"XOR {s} {s}")               # s := 0 (irreversible, as in pisa_interp)
                    self.notes.append(f"ORX {a[0]} {a[1]}: pyjanus 'or-and-clear' expanded to ANDX/XOR/XOR")
                elif op == "ANDX":                    # pyjanus: rd1 ^= rd2 & rs ; rd2 := 0
                    if len(a) != 3:
                        raise ConvertError(f"{ins.src.strip()!r}: pyjanus ANDX takes 3 registers")
                    self.check_r0_write(ins, a[0]); self.check_r0_write(ins, a[1])
                    d1, d2, s = _pal_reg(a[0]), _pal_reg(a[1]), _pal_reg(a[2])
                    self.emit(f"ANDX {d1} {d2} {s}", label)
                    self.emit(f"XOR {d2} {d2}")
                    self.notes.append(f"ANDX {a[0]} {a[1]} {a[2]}: pyjanus 'and-and-clear' expanded to ANDX/XOR")
                elif op == "SLTX":
                    self.check_r0_write(ins, a[0])
                    self.emit_sltx(a[0], a[1], a[2], label)
                elif op == "EXCH":
                    self.check_r0_write(ins, a[0])
                    self.emit(f"EXCH {_pal_reg(a[0])} {_pal_reg(a[1])}", label)
                elif op == "REGSWAP":                 # rfcl EXCH: swap two registers
                    x, y = _pal_reg(a[0]), _pal_reg(a[1])
                    self.emit(f"XOR {x} {y}", label)
                    self.emit(f"XOR {y} {x}")
                    self.emit(f"XOR {x} {y}")
                    self.notes.append(f"EXCH {a[0]} {a[1]} (rfcl register swap): expanded to an XOR triple")
                elif op in ("BRA", "RBRA"):
                    self.emit(f"{op} {a[0]}", label)
                elif op in ("BEQ", "BNE"):
                    self.emit(f"{op} {_pal_reg(a[0])} {_pal_reg(a[1])} {a[2]}", label)
                elif op in ("BGEZ", "BGTZ", "BLEZ", "BLTZ"):
                    self.emit(f"{op} {_pal_reg(a[0])} {a[1]}", label)
                elif op == "SWAPBR":
                    self.emit(f"SWAPBR {_pal_reg(a[0])}", label)
                elif op == "DATA":
                    self.emit(f"DATA {int(a[0])}", label)
                elif op == "START":
                    self.emit("START", label)
                elif op == "FINISH":
                    self.emit("FINISH", label)
                else:
                    raise ConvertError(f"{ins.src.strip()!r}: instruction {op} is not expressible in phpisa")
            except (ValueError, IndexError) as e:
                raise ConvertError(f"{ins.src.strip()!r}: malformed operands ({e})") from e


def convert(ir: List[Ins], *, emulate_sltx: bool = True, pendulum_calls: bool = False,
            pendulum_cf: bool = False, scratch: Optional[int] = None) -> Result:
    lowering = None
    cf = None
    if pendulum_cf:
        ir, cf = pendulum_cf_ir(ir)
        pendulum_calls = True
    if pendulum_calls:
        # fixpoint on the length: the stack base must lie above the emitted code
        base = len(ir) + 64
        for _ in range(4):
            lowered, lowering = pendulum_calls_ir(ir, base)
            em = Emitter(lowered, emulate_sltx, scratch); em.run()
            if base >= len(em.lines) + 32:
                break
            base = len(em.lines) + 64
    else:
        em = Emitter(ir, emulate_sltx, scratch); em.run()

    # start label: the (first) START instruction's label
    start_label = None
    for lab, txt in em.lines:
        if txt == "START" and lab:
            start_label = lab
            break
    if start_label is None:
        raise ConvertError("no labelled START instruction: cannot emit .start (phpisa would start at line 0)")

    # label table + collision check (phpisa upper-cases everything)
    label_index: Dict[str, int] = {}
    upper: Dict[str, str] = {}
    for i, (lab, _) in enumerate(em.lines):
        if lab is None:
            continue
        if lab in label_index:
            raise ConvertError(f"duplicate label {lab}")
        if lab.upper() in upper:
            raise ConvertError(f"labels {upper[lab.upper()]} and {lab} collide (phpisa is case-insensitive)")
        upper[lab.upper()] = lab
        label_index[lab] = i
    # branch range check (phpisa encodes offsets in 11 bits and truncates silently)
    for i, (lab, txt) in enumerate(em.lines):
        parts = txt.split()
        if parts[0] in ("BRA", "RBRA", "BEQ", "BNE", "BGEZ", "BGTZ", "BLEZ", "BLTZ"):
            target = parts[-1]
            if target not in label_index:
                raise ConvertError(f"line {i}: branch to unknown label {target}")
            off = label_index[target] - i
            if not _in_imm(off):
                raise ConvertError(f"line {i}: branch offset {off} to {target} exceeds phpisa's 11-bit range")

    width = max((len(l) for l in label_index), default=0) + 1
    out = [";; pendulum pal file -- generated by tools/pisa2pal.py", f"        .start {start_label}"]
    for lab, txt in em.lines:
        out.append(f"{(lab + ':').ljust(width)} {txt}" if lab else f"{' ' * width} {txt}")
    notes = sorted(set(em.notes))
    if em.scratch_used:
        notes.append(f"scratch register ${em.scratch} used for large immediates")
    return Result("\n".join(out) + "\n", start_label, notes, label_index, len(em.lines),
                  em.scratch if em.scratch_used else None, lowering, cf)


def pendulum_calls_ir(ir: List[Ins], base: int) -> Tuple[List[Ins], LoweringInfo]:
    return pendulum_calls(ir, base)


def pendulum_cf_ir(ir: List[Ins]) -> Tuple[List[Ins], CFInfo]:
    return pendulum_cf(ir)


def parse(text: str, dialect: str) -> List[Ins]:
    if dialect == "pyjanus":
        return parse_pyjanus(text)
    if dialect == "rfcl":
        return parse_rfcl(text)
    raise ConvertError(f"unknown dialect {dialect}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--dialect", choices=["pyjanus", "rfcl"], default="pyjanus")
    ap.add_argument("--pendulum-calls", action="store_true",
                    help="rewrite janus2pisa procedure prologues into the Pendulum call protocol "
                         "and relocate the stack above the code (adapter transformation)")
    ap.add_argument("--pendulum-cf", action="store_true",
                    help="also lower janus2pisa's if/from direct jumps into paired Pendulum branches "
                         "(implies --pendulum-calls; adapter transformation)")
    ap.add_argument("--no-sltx-emulation", action="store_true", help="error on SLTX instead of emulating it")
    ap.add_argument("--scratch", type=int, help="register number to use as scratch for large immediates")
    ap.add_argument("-v", "--verbose", action="store_true", help="print mapping notes to stderr")
    args = ap.parse_args(argv)
    with open(args.input) as f:
        text = f.read()
    try:
        ir = parse(text, args.dialect)
        res = convert(ir, emulate_sltx=not args.no_sltx_emulation,
                      pendulum_calls=args.pendulum_calls, pendulum_cf=args.pendulum_cf,
                      scratch=args.scratch)
    except ConvertError as e:
        print(f"pisa2pal: error: {e}", file=sys.stderr)
        return 1
    if args.verbose:
        for n in res.notes:
            print(f"pisa2pal: note: {n}", file=sys.stderr)
        if res.cf:
            print(f"pisa2pal: note: pendulum-cf lowering: {res.cf.ifs} if, {res.cf.loops} from", file=sys.stderr)
        if res.lowering and res.lowering.procedures:
            print(f"pisa2pal: note: pendulum-calls lowering applied to {res.lowering.procedures}; "
                  f"stack base {res.lowering.old_stack_base} -> {res.lowering.new_stack_base}", file=sys.stderr)
    if args.output:
        with open(args.output, "w") as f:
            f.write(res.pal)
    else:
        sys.stdout.write(res.pal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
