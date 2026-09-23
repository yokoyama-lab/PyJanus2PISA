#!/usr/bin/env python3
"""pal2pisa: load phpisa / Rlang-compiler PAL text into pisa.py dataclasses and
run it on an extended pisa_interp.PISAMachine.

PAL constructs and how they are handled
---------------------------------------
  ``$N`` / ``RN`` registers, ``;`` comments, ``LABEL:`` prefixes, commas   parsed
  ``.start L`` / ``.START L``      PISAMachine insists on a label named ``start``;
                                  when L != start we append ``start: BRA L`` at
                                  the END of the code (so no address shifts)
  ``ADDI $d LABEL`` / ``-LABEL``  label immediates (phpisa: label_address) are
                                  resolved to the line index / its negation
  ``DATA v`` anywhere             PISAMachine only loads the LEADING run of DATA
                                  words into memory; we preload mem[i] = v for
                                  every DATA at line i (phpisa semantics: memory
                                  is the program array)
  ``OUTPUT $r`` / ``SHOW $r``     recorded as "Rn = v" lines (PALMachine.output)
  ``START``                       no-op
  ``RL RR RLV RRV SLLX SRLX SRAX SLLVX SRLVX SRAVX ANDX(3-op) ORX(3-op) NORX
  ANDIX ORIX``                    implemented in PALMachine with phpisa's
                                  semantics (32-bit rotates, XOR-into-dest)
  ``BGTZ BLEZ BLTZ``              NOT supported: PISAMachine.run dispatches
                                  conditionals by class (BEQ/BNE/BGEZ only)
  ``SUBI``                        does not exist in PAL (phpisa has no SUBI)

Note that phpisa's 3-operand ``ANDX $d $s $t`` (d ^= s & t) is NOT
pisa.ANDX (janus2pisa's ``ANDX rd1 rd2 rs``: rd1 ^= rd2 & rs; rd2 := 0), and
phpisa's ``ORX $d $s $t`` is not pisa.ORX (rd |= rs; rs := 0).  Separate
classes ANDX3 / ORX3 are used here.

Usage:  python3 tools/pal2pisa.py file.pal [--max-steps N] [--regs R1=5,...]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pisa import (Instr, LabeledInstr, ADD, SUB, NEG, XOR, ADDI, XORI, EXCH,   # noqa: E402
                  BRA, RBRA, BEQ, BNE, BGEZ, SWAPBR, DATA, START, FINISH)
from pisa_interp import PISAMachine, PISAError                                # noqa: E402


class LoadError(Exception):
    pass


# --- PAL-only instructions (phpisa semantics) -------------------------------

@dataclass
class OUTPUT(Instr):
    rd: str

@dataclass
class ANDX3(Instr):
    rd: str; rs: str; rt: str

@dataclass
class ORX3(Instr):
    rd: str; rs: str; rt: str

@dataclass
class NORX(Instr):
    rd: str; rs: str; rt: str

@dataclass
class ANDIX(Instr):
    rd: str; rs: str; c: int

@dataclass
class ORIX(Instr):
    rd: str; rs: str; c: int

@dataclass
class ROT(Instr):            # RL / RR (imm) and RLV / RRV (register amount)
    rd: str; left: bool; amt: Optional[int]; rt: Optional[str]

@dataclass
class SHIFTX(Instr):         # SLLX / SRLX / SRAX and the V forms
    rd: str; rs: str; kind: str; amt: Optional[int]; rt: Optional[str]


MASK32 = 0xFFFFFFFF


class PALMachine(PISAMachine):
    """PISAMachine plus the PAL-only instructions above and OUTPUT capture."""

    def __init__(self, code: List[LabeledInstr], data_init: Optional[Dict[int, int]] = None):
        super().__init__(code)
        self.output: List[str] = []
        if data_init:
            self.mem.update(data_init)

    def _exec_data(self, instr: Instr) -> None:
        if isinstance(instr, OUTPUT):
            self.output.append(f"R{int(instr.rd[1:])} = {self._read_reg(instr.rd)}")
        elif isinstance(instr, ANDX3):
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ (self._read_reg(instr.rs) & self._read_reg(instr.rt)))
        elif isinstance(instr, ORX3):
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ (self._read_reg(instr.rs) | self._read_reg(instr.rt)))
        elif isinstance(instr, NORX):
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ ~(self._read_reg(instr.rs) | self._read_reg(instr.rt)))
        elif isinstance(instr, ANDIX):
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ (self._read_reg(instr.rs) & instr.c))
        elif isinstance(instr, ORIX):
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ (self._read_reg(instr.rs) | instr.c))
        elif isinstance(instr, ROT):
            v = self._read_reg(instr.rd) & MASK32
            amt = (instr.amt if instr.amt is not None else self._read_reg(instr.rt)) & 31
            if instr.left:
                v = ((v << amt) | (v >> (32 - amt))) & MASK32
            else:
                v = ((v >> amt) | (v << (32 - amt))) & MASK32
            self._write_reg(instr.rd, v)
        elif isinstance(instr, SHIFTX):
            src = self._read_reg(instr.rs)
            amt = instr.amt if instr.amt is not None else self._read_reg(instr.rt)
            if instr.kind == "SLL":
                val = src << amt
            elif instr.kind == "SRA":
                val = src >> amt
            else:                                   # SRL: phpisa masks to 32 bits first
                val = (src & MASK32) >> amt
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ val)
        else:
            super()._exec_data(instr)


# --- parsing ----------------------------------------------------------------

@dataclass
class Loaded:
    code: List[LabeledInstr]
    start_label: str
    data_init: Dict[int, int]
    label_index: Dict[str, int]
    notes: List[str]
    unsupported: List[str]          # "line N: OP" for constructs PISAMachine lacks


def _reg(tok: str, lineno: int) -> str:
    m = re.fullmatch(r"[\$Rr](\d+)", tok)
    if not m:
        raise LoadError(f"line {lineno}: expected register, got {tok!r}")
    return f"r{int(m.group(1))}"


def parse_pal(text: str) -> Loaded:
    rows: List[Tuple[Optional[str], str, List[str], int]] = []
    start_label: Optional[str] = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        if line.upper().startswith(".START"):
            start_label = line.split()[1]
            continue
        label = None
        if ":" in line:
            lab, _, rest = line.partition(":")
            label, line = lab.strip(), rest.strip()          # case kept as written
            if not line:
                raise LoadError(f"line {lineno}: label {label} on an empty line is not supported")
        parts = line.split()
        rows.append((label, parts[0].upper(), parts[1:], lineno))

    # phpisa is case-insensitive (it upper-cases every line); PISAMachine is
    # case-sensitive, so labels keep their written form and references are
    # resolved through their upper-cased spelling.
    canon: Dict[str, str] = {}
    for lab, _, _, lineno in rows:
        if lab and lab.upper() in canon and canon[lab.upper()] != lab:
            raise LoadError(f"line {lineno}: labels {canon[lab.upper()]} and {lab} collide case-insensitively")
        if lab:
            canon[lab.upper()] = lab
    label_index = {lab: i for i, (lab, _, _, _) in enumerate(rows) if lab}

    def target(tok: str, lineno: int) -> str:
        if tok.upper() not in canon:
            raise LoadError(f"line {lineno}: branch to unknown label {tok}")
        return canon[tok.upper()]

    def imm(tok: str, lineno: int, allow_label: bool) -> int:
        try:
            return int(tok)
        except ValueError:
            pass
        if allow_label:
            neg = tok.startswith("-")
            name = (tok[1:] if neg else tok).upper()
            if name in canon:
                v = label_index[canon[name]]
                return -v if neg else v
            raise LoadError(f"line {lineno}: label immediate {tok!r} not found")
        raise LoadError(f"line {lineno}: immediate {tok!r} is not a number")

    code: List[LabeledInstr] = []
    data_init: Dict[int, int] = {}
    notes: List[str] = []
    unsupported: List[str] = []
    for idx, (label, op, a, lineno) in enumerate(rows):
        ins: Optional[Instr] = None
        try:
            if op == "DATA":
                v = imm(a[0], lineno, False); ins = DATA(v); data_init[idx] = v
            elif op == "ADD":   ins = ADD(_reg(a[0], lineno), _reg(a[1], lineno))
            elif op == "SUB":   ins = SUB(_reg(a[0], lineno), _reg(a[1], lineno))
            elif op == "XOR":   ins = XOR(_reg(a[0], lineno), _reg(a[1], lineno))
            elif op == "NEG":   ins = NEG(_reg(a[0], lineno))
            elif op == "ADDI":  ins = ADDI(_reg(a[0], lineno), imm(a[1], lineno, True))
            elif op == "XORI":  ins = XORI(_reg(a[0], lineno), imm(a[1], lineno, False))
            elif op == "EXCH":  ins = EXCH(_reg(a[0], lineno), _reg(a[1], lineno))
            elif op == "SWAPBR": ins = SWAPBR(_reg(a[0], lineno))
            elif op == "BRA":   ins = BRA(target(a[0], lineno))
            elif op == "RBRA":  ins = RBRA(target(a[0], lineno))
            elif op == "BEQ":   ins = BEQ(_reg(a[0], lineno), _reg(a[1], lineno), target(a[2], lineno))
            elif op == "BNE":   ins = BNE(_reg(a[0], lineno), _reg(a[1], lineno), target(a[2], lineno))
            elif op == "BGEZ":  ins = BGEZ(_reg(a[0], lineno), target(a[1], lineno))
            elif op in ("BGTZ", "BLEZ", "BLTZ"):
                unsupported.append(f"line {lineno}: {op} (PISAMachine.run has no case for it)")
                ins = ADDI("r0", 0)
            elif op == "START":  ins = START()
            elif op == "FINISH": ins = FINISH()
            elif op in ("OUTPUT", "SHOW"): ins = OUTPUT(_reg(a[0], lineno))
            elif op == "SUBI":
                raise LoadError(f"line {lineno}: SUBI is not a PAL instruction (phpisa has no SUBI)")
            elif op == "ANDX":  ins = ANDX3(_reg(a[0], lineno), _reg(a[1], lineno), _reg(a[2], lineno))
            elif op == "ORX":   ins = ORX3(_reg(a[0], lineno), _reg(a[1], lineno), _reg(a[2], lineno))
            elif op == "NORX":  ins = NORX(_reg(a[0], lineno), _reg(a[1], lineno), _reg(a[2], lineno))
            elif op == "ANDIX": ins = ANDIX(_reg(a[0], lineno), _reg(a[1], lineno), imm(a[2], lineno, False))
            elif op == "ORIX":  ins = ORIX(_reg(a[0], lineno), _reg(a[1], lineno), imm(a[2], lineno, False))
            elif op == "RL":    ins = ROT(_reg(a[0], lineno), True, imm(a[1], lineno, False), None)
            elif op == "RR":    ins = ROT(_reg(a[0], lineno), False, imm(a[1], lineno, False), None)
            elif op == "RLV":   ins = ROT(_reg(a[0], lineno), True, None, _reg(a[1], lineno))
            elif op == "RRV":   ins = ROT(_reg(a[0], lineno), False, None, _reg(a[1], lineno))
            elif op in ("SLLX", "SRLX", "SRAX"):
                ins = SHIFTX(_reg(a[0], lineno), _reg(a[1], lineno), op[:3], imm(a[2], lineno, False), None)
            elif op in ("SLLVX", "SRLVX", "SRAVX"):
                ins = SHIFTX(_reg(a[0], lineno), _reg(a[1], lineno), op[:3], None, _reg(a[2], lineno))
            elif op in ("NOOP", "DUMP", "WARN"):
                ins = ADDI("r0", 0); notes.append(f"line {lineno}: {op} treated as NOP")
            else:
                raise LoadError(f"line {lineno}: unknown PAL instruction {op}")
        except IndexError:
            raise LoadError(f"line {lineno}: missing operands for {op}")
        code.append(LabeledInstr(label, ins))

    if start_label is None:
        start_label = rows[0][0] or "_PAL_ENTRY"
        if rows[0][0] is None:
            code[0] = LabeledInstr(start_label, code[0].instr)
            label_index[start_label] = 0
            canon[start_label.upper()] = start_label
        notes.append("no .start directive: phpisa starts at line 0")
    if start_label.upper() not in canon:
        raise LoadError(f".start {start_label}: label not found")
    start_label = canon[start_label.upper()]
    if start_label != "start":
        # PISAMachine insists on a label spelled exactly 'start'
        code.append(LabeledInstr("start", BRA(start_label)))
        notes.append(f"appended 'start: BRA {start_label}' for PISAMachine (.start {start_label})")
    return Loaded(code, start_label, data_init, label_index, notes, unsupported)


def load(text: str) -> Tuple[PALMachine, Loaded]:
    ld = parse_pal(text)
    m = PALMachine(ld.code, ld.data_init)
    m.check_clean = False
    return m, ld


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a PAL file on the extended pisa_interp machine")
    ap.add_argument("file")
    ap.add_argument("--max-steps", type=int, default=10_000_000)
    ap.add_argument("--regs", default="", help="initial registers, e.g. R1=5,R2=3")
    args = ap.parse_args(argv)
    with open(args.file) as f:
        text = f.read()
    try:
        m, ld = load(text)
    except LoadError as e:
        print(f"pal2pisa: error: {e}", file=sys.stderr)
        return 1
    for n in ld.notes:
        print(f"pal2pisa: note: {n}", file=sys.stderr)
    for u in ld.unsupported:
        print(f"pal2pisa: unsupported: {u}", file=sys.stderr)
    for kv in filter(None, args.regs.split(",")):
        k, v = kv.split("=")
        m.regs[int(k.strip().lstrip("Rr$"))] = int(v)
    try:
        m.run(max_steps=args.max_steps)
        status = "FINISH"
    except PISAError as e:
        status = f"error: {e}"
    for line in m.output:
        print(line)
    print(f"pal2pisa: {status}; steps={m._steps}")
    print(m.dump_state())
    return 0


if __name__ == "__main__":
    sys.exit(main())
