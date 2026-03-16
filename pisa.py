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

@dataclass
class ORX(Instr):
    rd: str; rs: str

@dataclass
class ANDX(Instr):
    rd1: str; rd2: str; rs: str

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
    if isinstance(instr, (ADD, SUB, XOR, ORX, EXCH)):
        return f"{name} {instr.rd} {instr.rs}"
    if isinstance(instr, NEG):
        return f"{name} {instr.rd}"
    if isinstance(instr, (ADDI, SUBI, XORI)):
        return f"{name} {instr.rd} {instr.c}"
    if isinstance(instr, ANDX):
        return f"{name} {instr.rd1} {instr.rd2} {instr.rs}"
    if isinstance(instr, SLTX):
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
