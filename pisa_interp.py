"""PISA interpreter for programs compiled by janus2pisa.

Execution model
---------------
janus2pisa emits PISA code whose control flow uses TWO distinct mechanisms:

1.  **Paired-branch (Pendulum) mechanism** – used for:
     * Procedure call / return (f_top/f_bot wrappers)
     * If/else entry merge (if_false_1 ↔ if_test_2 pair)
     * If/else exit merge (if_assert_true ↔ if_end pair)

    At load time we identify all "paired" branch edges:
      A is paired with B if:
        (a) A is a BRA/RBRA targeting B AND B is a BRA/RBRA targeting A, OR
        (b) A is a BRA/RBRA targeting B AND B is a conditional
            (BEQ/BNE/BGEZ) targeting A.
    For paired edges, the Pendulum br register is updated and the standard
    Pendulum control logic applies:
        br += (target − pc)
        if br ≠ 0:  pc += br
        else:       pc += 1  (fall-through)

2.  **Simple direct jump** – all other BRA/RBRA instructions and all
    conditional branches (BEQ/BNE/BGEZ) when br = 0 at the time of
    execution.
        if condition/unconditional:  pc = target
        else:                        pc += 1

3.  **Data instructions** – always advance pc by 1, regardless of br.

4.  **SWAPBR** – exchanges the register with br; pc += 1 (treated as data).
    br itself is never used to control pc for data instructions.

Procedure call handling
-----------------------
Calls are recognised as BRA instructions whose target label is a known
procedure name.  We use a **software call stack** (not the Pendulum br
mechanism) to save / restore the return address:
  – CALL:   push (pc + 1) onto call stack; execute the prologue normally.
  – RETURN: detected when execution reaches ``f_bot: BRA f_top``.  We
    run the epilogue (second pass of the prologue) to keep memory coherent,
    then pop the call stack for the actual return jump.

Reference
---------
  Axelsen. "Clean Translation of an Imperative Reversible Programming
  Language." CC 2011.  Figs. 5, 11, 12.

  Axelsen, Glück, Yokoyama. "Reversible Machine Code and Its Abstract
  Processor Architecture." CSR 2007.  Fig. 2.
"""

import sys
import os
from typing import List, Dict, Set

sys.path.insert(0, os.path.dirname(__file__))

from pisa import (
    LabeledInstr, Instr,
    ADD, SUB, NEG, XOR, ADDI, SUBI, XORI,
    ORX, ANDX, SLTX,
    EXCH, BRA, RBRA, BEQ, BNE, BGEZ, SWAPBR,
    DATA, START, FINISH,
)


class PISAError(Exception):
    pass


class PISAMachine:
    """PISA interpreter with software call stack.

    Registers:
        r0  = always 0 (reads 0, writes discarded)
        r1  = rsp (stack pointer)
        r2  = rro (return offset register, managed by SWAPBR)
        r3..r31 = general purpose
        br  = branch register (tracked for paired branches; never used
              for data-instruction pc control)

    Memory: word-addressed sparse dict.  DATA c instructions
    initialise memory at addresses 0, 1, 2, … at load time.
    """

    NUM_REGS = 32

    def __init__(self, code: List[LabeledInstr]):
        self.code = code

        # ---- label map ----
        self.label_map: Dict[str, int] = {}
        for idx, li in enumerate(code):
            if li.label is not None:
                self.label_map[li.label] = idx

        # ---- identify procedure names ----
        # A label ``f`` is a procedure name if both ``f_top`` and ``f_bot``
        # also exist as labels.
        self._proc_names: Set[str] = set()
        for label in list(self.label_map.keys()):
            if label.endswith('_top'):
                base = label[:-4]
                if base in self.label_map and (base + '_bot') in self.label_map:
                    self._proc_names.add(base)

        # ---- identify paired branch instructions ----
        # paired_pcs: set of pc indices that participate in a Pendulum pair.
        self._paired_pcs: Set[int] = set()
        self._detect_paired_branches()

        # ---- registers ----
        self.regs = [0] * self.NUM_REGS
        self.br = 0

        # ---- memory ----
        self.mem: Dict[int, int] = {}

        # ---- software call stack ----
        self._call_stack: List[int] = []

        # direction bit (tracked but not used for pc control here)
        self.dir = 1

        # ---- initialise memory from DATA instructions ----
        data_addr = 0
        for li in code:
            if isinstance(li.instr, DATA):
                self.mem[data_addr] = li.instr.value
                data_addr += 1
            else:
                break

        self._data_words = data_addr

        # ---- find start pc ----
        if 'start' not in self.label_map:
            raise PISAError("No 'start' label found in program")
        self._start_pc = self.label_map['start']

        self._steps = 0

    # ------------------------------------------------------------------
    # Paired-branch detection
    # ------------------------------------------------------------------

    def _detect_paired_branches(self) -> None:
        """Scan code and mark pc indices that participate in Pendulum pairs.

        An index A is paired if one of the following holds:
          (a) code[A] is BRA/RBRA targeting B, and code[B] is BRA/RBRA
              targeting A.
          (b) code[A] is BRA/RBRA targeting B, and code[B] is a conditional
              branch (BEQ/BNE/BGEZ) whose label target is A.
        """
        for a, li in enumerate(self.code):
            instr = li.instr
            if not isinstance(instr, (BRA, RBRA)):
                continue
            b = self.label_map.get(instr.label)
            if b is None:
                continue
            b_instr = self.code[b].instr
            if isinstance(b_instr, (BRA, RBRA)):
                # Check b → a
                if self.label_map.get(b_instr.label) == a:
                    self._paired_pcs.add(a)
                    self._paired_pcs.add(b)
            elif isinstance(b_instr, (BEQ, BNE, BGEZ)):
                # Check b's branch target → a
                if self.label_map.get(b_instr.label) == a:
                    self._paired_pcs.add(a)
                    # b (the conditional) is not marked as paired; it uses
                    # the br-cancel path only when br ≠ 0 on arrival.

    # ------------------------------------------------------------------
    # Register / memory helpers
    # ------------------------------------------------------------------

    def _read_reg(self, name: str) -> int:
        if name == 'r0':
            return 0
        return self.regs[int(name[1:])]

    def _write_reg(self, name: str, value: int) -> None:
        if name == 'r0':
            return
        self.regs[int(name[1:])] = value

    def _mem_read(self, addr: int) -> int:
        return self.mem.get(addr, 0)

    def _mem_write(self, addr: int, value: int) -> None:
        self.mem[addr] = value

    # ------------------------------------------------------------------
    # Data instruction execution
    # ------------------------------------------------------------------

    def _exec_data(self, instr: Instr) -> None:
        """Execute one data / special instruction (does not touch pc)."""

        if isinstance(instr, ADD):
            self._write_reg(instr.rd, self._read_reg(instr.rd) + self._read_reg(instr.rs))
        elif isinstance(instr, SUB):
            self._write_reg(instr.rd, self._read_reg(instr.rd) - self._read_reg(instr.rs))
        elif isinstance(instr, NEG):
            self._write_reg(instr.rd, -self._read_reg(instr.rd))
        elif isinstance(instr, XOR):
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ self._read_reg(instr.rs))
        elif isinstance(instr, ADDI):
            self._write_reg(instr.rd, self._read_reg(instr.rd) + instr.c)
        elif isinstance(instr, SUBI):
            self._write_reg(instr.rd, self._read_reg(instr.rd) - instr.c)
        elif isinstance(instr, XORI):
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ instr.c)
        elif isinstance(instr, ORX):
            self._write_reg(instr.rd, self._read_reg(instr.rd) | self._read_reg(instr.rs))
            self._write_reg(instr.rs, 0)
        elif isinstance(instr, ANDX):
            val = self._read_reg(instr.rd2) & self._read_reg(instr.rs)
            self._write_reg(instr.rd1, self._read_reg(instr.rd1) ^ val)
            self._write_reg(instr.rd2, 0)
        elif isinstance(instr, SLTX):
            bit = 1 if self._read_reg(instr.rs) < self._read_reg(instr.rt) else 0
            self._write_reg(instr.rd, self._read_reg(instr.rd) ^ bit)
        elif isinstance(instr, EXCH):
            addr = self._read_reg(instr.rs)
            old_reg = self._read_reg(instr.rd)
            old_mem = self._mem_read(addr)
            self._write_reg(instr.rd, old_mem)
            self._mem_write(addr, old_reg)
        elif isinstance(instr, SWAPBR):
            old_reg = self._read_reg(instr.rd)
            old_br = self.br
            self._write_reg(instr.rd, old_br)
            self.br = old_reg
        elif isinstance(instr, (DATA, START)):
            pass  # no-op at runtime
        else:
            raise PISAError(f"Unknown instruction: {type(instr).__name__}")

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def run(self, max_steps: int = 10_000_000) -> Dict[int, int]:
        """Run from START to FINISH; return final memory state."""
        pc = self._start_pc
        self._steps = 0

        while self._steps < max_steps:
            self._steps += 1

            if pc < 0 or pc >= len(self.code):
                raise PISAError(f"PC out of range: pc={pc} at step {self._steps}")

            li = self.code[pc]
            instr = li.instr

            # ---- FINISH: halt ----
            if isinstance(instr, FINISH):
                return dict(self.mem)

            # ================================================================
            # BRA / RBRA
            # ================================================================
            if isinstance(instr, (BRA, RBRA)):
                target_label = instr.label
                target_pc = self.label_map[target_label]

                if target_label in self._proc_names:
                    # ---- CALL: use software call stack ----
                    self._call_stack.append(pc + 1)
                    pc = target_pc

                elif (target_label.endswith('_bot')
                      and target_label[:-4] in self._proc_names):
                    # ---- RETURN: f_bot: BRA f_top ----
                    proc_name = target_label[:-4]
                    f_top_pc = self.label_map[proc_name + '_top']
                    proc_pc = self.label_map[proc_name]
                    # Run the epilogue (second prologue pass) to keep memory
                    # coherent.  The prologue lives at:
                    #   f_top_pc+1 .. proc_pc+5  (SUBI EXCH SWAPBR NEG EXCH ADDI)
                    epilogue_start = f_top_pc + 1
                    epilogue_end = proc_pc + 6  # exclusive
                    for ep in range(epilogue_start, epilogue_end):
                        self._exec_data(self.code[ep].instr)
                    # Return via software call stack
                    if not self._call_stack:
                        raise PISAError(
                            f"RETURN without CALL at pc={pc}, proc={proc_name}"
                        )
                    pc = self._call_stack.pop()

                elif pc in self._paired_pcs:
                    # ---- Paired Pendulum branch ----
                    offset = target_pc - pc
                    self.br += offset
                    if self.br != 0:
                        pc = pc + self.br
                    else:
                        pc += 1  # fall-through

                else:
                    # ---- Simple direct jump ----
                    pc = target_pc

            # ================================================================
            # Conditional branches
            # ================================================================
            elif isinstance(instr, BEQ):
                taken = (self._read_reg(instr.rd) == self._read_reg(instr.rs))
                if self.br != 0:
                    # Pendulum cancel path (arrived via a paired BRA)
                    if taken:
                        self.br += (self.label_map[instr.label] - pc)
                    if self.br != 0:
                        pc = pc + self.br
                    else:
                        # Cancel resolved: jump to the instruction after the
                        # paired BRA that sent us here (label_target + 1).
                        pc = self.label_map[instr.label] + 1
                else:
                    # Direct conditional jump
                    if taken:
                        pc = self.label_map[instr.label]
                    else:
                        pc += 1

            elif isinstance(instr, BNE):
                taken = (self._read_reg(instr.rd) != self._read_reg(instr.rs))
                if self.br != 0:
                    if taken:
                        self.br += (self.label_map[instr.label] - pc)
                    if self.br != 0:
                        pc = pc + self.br
                    else:
                        pc = self.label_map[instr.label] + 1
                else:
                    if taken:
                        pc = self.label_map[instr.label]
                    else:
                        pc += 1

            elif isinstance(instr, BGEZ):
                taken = (self._read_reg(instr.rd) >= 0)
                if self.br != 0:
                    if taken:
                        self.br += (self.label_map[instr.label] - pc)
                    if self.br != 0:
                        pc = pc + self.br
                    else:
                        pc = self.label_map[instr.label] + 1
                else:
                    if taken:
                        pc = self.label_map[instr.label]
                    else:
                        pc += 1

            # ================================================================
            # Data / special instructions  (pc always +1)
            # ================================================================
            else:
                self._exec_data(instr)
                pc += 1

        raise PISAError(
            f"Exceeded max_steps={max_steps}. "
            f"Possible infinite loop at step {self._steps}, pc={pc}, br={self.br}."
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_var(self, offset: int) -> int:
        """Read variable at memory offset (0-indexed DATA address)."""
        return self._mem_read(offset)

    def dump_state(self) -> str:
        """Return a debug string of current machine state."""
        lines = [
            f"br={self.br}  dir={self.dir}",
            f"steps={self._steps}",
            f"call_stack_depth={len(self._call_stack)}",
        ]
        nonzero_regs = [(i, self.regs[i]) for i in range(self.NUM_REGS) if self.regs[i] != 0]
        if nonzero_regs:
            reg_str = "  ".join(f"r{i}={v}" for i, v in nonzero_regs)
            lines.append(f"regs: {reg_str}")
        else:
            lines.append("regs: all zero")
        if self.mem:
            mem_str = "  ".join(f"[{k}]={v}" for k, v in sorted(self.mem.items()))
            lines.append(f"mem: {mem_str}")
        else:
            lines.append("mem: empty")
        return "\n".join(lines)
