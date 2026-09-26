"""PISA instruction set type definitions and printer."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union


class Instr:
    """Base class for PISA instructions."""
    pass


# --- Arithmetic ---

@dataclass
class ADD(Instr):
    rd: str; rs: str

@dataclass
class SUB(Instr):
    rd: str; rs: str

@dataclass
class NEG(Instr):
    rd: str

@dataclass
class XOR(Instr):
    rd: str; rs: str

@dataclass
class ADDI(Instr):
    rd: str; c: int

@dataclass
class SUBI(Instr):
    rd: str; c: int

@dataclass
class XORI(Instr):
    rd: str; c: int


# --- Logic ---

# Pendulum's 3-operand XOR-accumulating forms (as in phpisa):
#   ORX  rd rs rt:  rd ^= rs | rt
#   ANDX rd rs rt:  rd ^= rs & rt
# Self-inverse when rd is neither rs nor rt (see is_wf).  Until 2026-09 this
# project used a 2-operand "or-and-clear" ORX (rd |= rs; rs := 0) and an
# "and-and-clear" ANDX (rd1 ^= rd2 & rs; rd2 := 0), which are not injective.

@dataclass
class ORX(Instr):
    rd: str; rs: str; rt: str

@dataclass
class ANDX(Instr):
    rd: str; rs: str; rt: str

@dataclass
class SLTX(Instr):
    rd: str; rs: str; rt: str


# --- Data movement ---

@dataclass
class EXCH(Instr):
    rd: str; rs: str


# --- Control flow ---

@dataclass
class BRA(Instr):
    label: str

@dataclass
class RBRA(Instr):
    label: str

@dataclass
class BEQ(Instr):
    rd: str; rs: str; label: str

@dataclass
class BNE(Instr):
    rd: str; rs: str; label: str

@dataclass
class BGEZ(Instr):
    rd: str; label: str

@dataclass
class SWAPBR(Instr):
    rd: str


# --- Pseudo/directives ---

@dataclass
class DATA(Instr):
    value: int

@dataclass
class START(Instr):
    pass

@dataclass
class FINISH(Instr):
    pass


# --- Labeled instruction ---

@dataclass
class LabeledInstr:
    label: Optional[str]
    instr: Instr


def format_instr(instr: Instr) -> str:
    """Format a single instruction to string."""
    name = type(instr).__name__
    if isinstance(instr, (ADD, SUB, XOR, EXCH)):
        return f"{name} {instr.rd} {instr.rs}"
    if isinstance(instr, NEG):
        return f"{name} {instr.rd}"
    if isinstance(instr, (ADDI, SUBI, XORI)):
        return f"{name} {instr.rd} {instr.c}"
    if isinstance(instr, (ANDX, ORX, SLTX)):
        return f"{name} {instr.rd} {instr.rs} {instr.rt}"
    if isinstance(instr, (BRA, RBRA)):
        return f"{name} {instr.label}"
    if isinstance(instr, (BEQ, BNE)):
        return f"{name} {instr.rd} {instr.rs} {instr.label}"
    if isinstance(instr, BGEZ):
        return f"{name} {instr.rd} {instr.label}"
    if isinstance(instr, SWAPBR):
        return f"{name} {instr.rd}"
    if isinstance(instr, DATA):
        return f"{name} {instr.value}"
    if isinstance(instr, START):
        return "START"
    if isinstance(instr, FINISH):
        return "FINISH"
    return name


def is_wf(instr: Instr) -> bool:
    """Is `instr` locally invertible (injective on machine states)?

    This is the well-formedness condition every instruction emitted by
    codegen.py must satisfy, so that the compiled program is reversible
    instruction by instruction (not merely as a whole):

      ADD / SUB / XOR rd rs     rd != rs     (XOR r r clears r, ADD r r doubles)
      ANDX / ORX / SLTX rd rs rt
                                rd not in {rs, rt}   (rd ^= f(rs, rt) is then
                                self-inverse)
      EXCH rd ra                rd != ra and rd != r0   (EXCH r r loses the
                                register; EXCH r0 ra loses the memory word)
      SWAPBR rd                 rd != r0     (r0 would lose br)
      NEG, ADDI, SUBI, XORI, branches, DATA, START, FINISH
                                always

    A write to r0 through ADD/SUB/XOR/ANDX/ORX/SLTX/NEG/ADDI/... is discarded
    and therefore a no-op, which is trivially invertible.  pisa_interp.py still
    *executes* ill-formed instructions (e.g. `XOR r r`); codegen must not emit
    them (test_wf.py checks every compiled program).
    """
    if isinstance(instr, (ADD, SUB, XOR)):
        return instr.rd != instr.rs
    if isinstance(instr, (ANDX, ORX, SLTX)):
        return instr.rd != instr.rs and instr.rd != instr.rt
    if isinstance(instr, EXCH):
        return instr.rd != instr.rs and instr.rd != "r0"
    if isinstance(instr, SWAPBR):
        return instr.rd != "r0"
    if isinstance(instr, (NEG, ADDI, SUBI, XORI, BRA, RBRA, BEQ, BNE, BGEZ,
                          DATA, START, FINISH)):
        return True
    return False


def print_program(labeled_instrs: List[LabeledInstr]) -> str:
    """Format a list of labeled instructions to PISA assembly text."""
    lines = []
    for li in labeled_instrs:
        text = format_instr(li.instr)
        if li.label:
            lines.append(f"{li.label}: {text}")
        else:
            lines.append(f"       {text}")
    return "\n".join(lines)
