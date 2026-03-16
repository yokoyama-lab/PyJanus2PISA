#!/usr/bin/env python3
"""Tests for Janus program inversion.

For each test program P, we verify the round-trip property:
    P⁻¹(P(initial)) = initial

That is: running the inverse program on the output of the forward program
recovers the original input state.

We also test inversion of specific programs against known correct inverses.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from lexer import tokenize
from parser import parse
from codegen import compile_program
from pisa_interp import PISAMachine
from inverse import invert_program, invert_stmt
from syntax import (
    Skip, AssignVar, AssignArr, Swap, Call, Uncall, If, From, Seq, Const, Var
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_program(src: str, max_steps: int = 1_000_000) -> dict:
    """Compile and run a Janus program; return final memory."""
    prog = parse(tokenize(src))
    code = compile_program(prog)
    m = PISAMachine(code)
    m.run(max_steps=max_steps)
    return dict(m.mem)


def run_inverse(src: str, max_steps: int = 1_000_000) -> dict:
    """Compile and run the inverse of a Janus program; return final memory."""
    prog = parse(tokenize(src))
    inv = invert_program(prog)
    code = compile_program(inv)
    m = PISAMachine(code)
    m.run(max_steps=max_steps)
    return dict(m.mem)


def round_trip(src: str, max_steps: int = 1_000_000) -> dict:
    """Run P then P⁻¹ on zero-initialised memory; return final memory.

    To simulate P⁻¹(P(0)):
      1. Run P on 0-memory → get output memory state.
      2. Patch the inverse program's DATA section to start from that state.
      3. Run P⁻¹ → should recover 0-memory.

    Here we simulate it by composing at the PISA level: we pre-load the
    memory output from step 1 into a second machine that runs P⁻¹.
    """
    # Step 1: run P
    prog = parse(tokenize(src))
    code_fwd = compile_program(prog)
    m_fwd = PISAMachine(code_fwd)
    m_fwd.run(max_steps=max_steps)

    # Step 2: compile P⁻¹
    inv = invert_program(prog)
    code_inv = compile_program(inv)
    m_inv = PISAMachine(code_inv)

    # Patch memory: copy P's output into P⁻¹'s initial memory (DATA section)
    data_words = m_inv._data_words
    for addr in range(data_words):
        m_inv.mem[addr] = m_fwd.mem.get(addr, 0)

    m_inv.run(max_steps=max_steps)
    # Return only the data-region
    return {addr: m_inv.mem.get(addr, 0) for addr in range(data_words)}


# ---------------------------------------------------------------------------
# Structural inversion tests
# ---------------------------------------------------------------------------

class TestInvertStmt(unittest.TestCase):
    """Unit tests for invert_stmt on AST nodes."""

    def test_skip_self_inverse(self):
        self.assertIsInstance(invert_stmt(Skip()), Skip)

    def test_assign_plus_becomes_minus(self):
        s = AssignVar('x', '+=', Const(1))
        inv = invert_stmt(s)
        self.assertIsInstance(inv, AssignVar)
        self.assertEqual(inv.op, '-=')
        self.assertEqual(inv.var, 'x')

    def test_assign_minus_becomes_plus(self):
        s = AssignVar('x', '-=', Const(1))
        inv = invert_stmt(s)
        self.assertEqual(inv.op, '+=')

    def test_assign_xor_self_inverse(self):
        s = AssignVar('x', '^=', Const(5))
        inv = invert_stmt(s)
        self.assertEqual(inv.op, '^=')

    def test_call_becomes_uncall(self):
        s = Call('f')
        inv = invert_stmt(s)
        self.assertIsInstance(inv, Uncall)
        self.assertEqual(inv.proc, 'f')

    def test_uncall_becomes_call(self):
        s = Uncall('f')
        inv = invert_stmt(s)
        self.assertIsInstance(inv, Call)
        self.assertEqual(inv.proc, 'f')

    def test_swap_self_inverse(self):
        s = Swap('x', None, 'y', None)
        inv = invert_stmt(s)
        self.assertIsInstance(inv, Swap)
        self.assertEqual(inv.lhs, 'x')
        self.assertEqual(inv.rhs, 'y')

    def test_if_swaps_test_and_fi(self):
        # if e1 then s1 else s2 fi e2  →  if e2 then s1⁻¹ else s2⁻¹ fi e1
        e1 = Const(1)
        e2 = Const(2)
        s1 = AssignVar('x', '+=', Const(1))
        s2 = Skip()
        s = If(test=e1, then_=s1, else_=s2, fi=e2)
        inv = invert_stmt(s)
        self.assertIsInstance(inv, If)
        self.assertEqual(inv.test, e2)
        self.assertEqual(inv.fi, e1)
        self.assertIsInstance(inv.then_, AssignVar)
        self.assertEqual(inv.then_.op, '-=')
        self.assertIsInstance(inv.else_, Skip)

    def test_from_swaps_from_until_and_do_loop(self):
        # from e1 do s1 loop s2 until e2
        # →  from e2 do s2⁻¹ loop s1⁻¹ until e1
        e1 = Const(0)
        e2 = Const(5)
        s1 = AssignVar('x', '+=', Const(1))
        s2 = Skip()
        s = From(from_=e1, do_=s1, loop_=s2, until=e2)
        inv = invert_stmt(s)
        self.assertIsInstance(inv, From)
        self.assertEqual(inv.from_, e2)
        self.assertEqual(inv.until, e1)
        # do = s2⁻¹ = Skip⁻¹ = Skip
        self.assertIsInstance(inv.do_, Skip)
        # loop = s1⁻¹ = (x += 1)⁻¹ = (x -= 1)
        self.assertIsInstance(inv.loop_, AssignVar)
        self.assertEqual(inv.loop_.op, '-=')

    def test_seq_reversal(self):
        stmts = [
            AssignVar('x', '+=', Const(1)),
            AssignVar('y', '+=', Const(2)),
            AssignVar('z', '+=', Const(3)),
        ]
        s = Seq(stmts)
        inv = invert_stmt(s)
        self.assertIsInstance(inv, Seq)
        self.assertEqual(len(inv.stmts), 3)
        # Order reversed; each inverted
        self.assertEqual(inv.stmts[0].var, 'z')  # was last
        self.assertEqual(inv.stmts[1].var, 'y')
        self.assertEqual(inv.stmts[2].var, 'x')  # was first
        for s2 in inv.stmts:
            self.assertEqual(s2.op, '-=')

    def test_double_inverse_is_identity(self):
        """Inverting twice recovers the original statement structure."""
        s = AssignVar('x', '+=', Const(7))
        self.assertEqual(invert_stmt(invert_stmt(s)), s)

        s2 = If(
            test=Const(0),
            then_=AssignVar('x', '-=', Const(1)),
            else_=Skip(),
            fi=Const(1),
        )
        inv_inv = invert_stmt(invert_stmt(s2))
        self.assertEqual(inv_inv.test, s2.test)
        self.assertEqual(inv_inv.fi, s2.fi)


# ---------------------------------------------------------------------------
# Round-trip tests: P⁻¹(P(0)) = 0
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):
    """Verify that P⁻¹(P(initial)) = initial for various programs."""

    def _check_zero(self, src):
        result = round_trip(src)
        for k, v in result.items():
            self.assertEqual(v, 0, f"Memory[{k}] = {v} ≠ 0 after round-trip")

    def test_single_increment(self):
        """x += 1; inverse: x -= 1.  Round trip: x = 0."""
        self._check_zero("int x\nprocedure main\n  x += 1")

    def test_add_five(self):
        """x += 5; round trip."""
        self._check_zero("int x\nprocedure main\n  x += 5")

    def test_two_variables(self):
        """x += 3; y += 7; round trip."""
        self._check_zero("""int x
int y
procedure main
  x += 3
  y += 7""")

    def test_sequence(self):
        """x += 1; x += 2; x += 3; round trip."""
        self._check_zero("""int x
procedure main
  x += 1
  x += 2
  x += 3""")

    def test_if_true_path(self):
        """if x = 0 then x += 10 else skip fi x = 10; round trip."""
        self._check_zero("""int x
procedure main
  if x = 0 then x += 10 else skip fi x = 10""")

    def test_if_else_path(self):
        """Valid if/else: fi must discriminate branches.
        if x = 0 then x ^= 1 else skip fi x = 1
        - then path (x=0): x^=1 → x=1, fi=(x=1)=T ✓
        - else path (x=1): skip → x=1, fi=(x=1)=T  ← not reachable from x=0
        Round trip from x=0 → x=1 → x=0.
        """
        self._check_zero("""int x
procedure main
  if x = 0 then x ^= 1 else skip fi x = 1""")

    def test_loop_count_to_5(self):
        """from x = 0 do x += 1 loop skip until x = 5; round trip."""
        self._check_zero("""int x
procedure main
  from x = 0
  do x += 1
  loop skip
  until x = 5""")

    def test_loop_accumulate(self):
        """Accumulating loop: x = 3, y = 6; round trip → 0."""
        self._check_zero("""int x
int y
procedure main
  from x = 0
  do
    x += 1
    y += 2
  loop skip
  until x = 3""")

    def test_procedure_call(self):
        """call inc; round trip."""
        self._check_zero("""int x
procedure inc
  x += 1
procedure main
  call inc""")

    def test_nested_calls(self):
        """Nested procedure calls; round trip."""
        self._check_zero("""int x
procedure add2
  x += 2
procedure double_add2
  call add2
  call add2
procedure main
  call double_add2""")

    def test_swap(self):
        """Swap after setting values; round trip."""
        self._check_zero("""int x
int y
procedure main
  x += 3
  y += 7
  x <=> y""")

    def test_xor_assign(self):
        """x ^= 15; round trip (XOR is self-inverse)."""
        self._check_zero("""int x
procedure main
  x ^= 15""")

    def test_if_with_comparison(self):
        """If with comparison expression; round trip."""
        self._check_zero("""int x
int y
procedure main
  x += 5
  if x = 5 then y += 1 else skip fi y = 1""")


# ---------------------------------------------------------------------------
# Semantic correctness: run P⁻¹ starting from P's output
# ---------------------------------------------------------------------------

class TestSemanticInverse(unittest.TestCase):
    """Test that the inverse program computes the expected reverse function."""

    def test_increment_inverse_is_decrement(self):
        """P: x += 5.  P⁻¹ on x=5 should give x=0."""
        # Run forward: x starts 0, ends 5
        fwd = run_program("int x\nprocedure main\n  x += 5")
        self.assertEqual(fwd.get(0, 0), 5)
        # Run inverse: pre-load x=5, should give x=0
        prog = parse(tokenize("int x\nprocedure main\n  x += 5"))
        inv = invert_program(prog)
        code = compile_program(inv)
        m = PISAMachine(code)
        m.mem[0] = 5   # pre-load x=5
        m.run()
        self.assertEqual(m.mem.get(0, 0), 0)

    def test_loop_inverse_counts_down(self):
        """P: count x from 0 to 5.  P⁻¹ on x=5 should give x=0."""
        prog = parse(tokenize("""int x
procedure main
  from x = 0
  do x += 1
  loop skip
  until x = 5"""))
        inv = invert_program(prog)
        code = compile_program(inv)
        m = PISAMachine(code)
        m.mem[0] = 5
        m.run()
        self.assertEqual(m.mem.get(0, 0), 0)

    def test_swap_inverse_is_swap(self):
        """P: set x=3,y=7 then swap → x=7,y=3.  P⁻¹(7,3) = (0,0) after full round-trip."""
        src = """int x
int y
procedure main
  x += 3
  y += 7
  x <=> y"""
        # Forward: x→7, y→3
        fwd = run_program(src)
        self.assertEqual(fwd.get(0, 0), 7)
        self.assertEqual(fwd.get(1, 0), 3)
        # Inverse from (7,3) → (0,0)
        prog = parse(tokenize(src))
        inv = invert_program(prog)
        code = compile_program(inv)
        m = PISAMachine(code)
        m.mem[0] = 7
        m.mem[1] = 3
        m.run()
        self.assertEqual(m.mem.get(0, 0), 0)
        self.assertEqual(m.mem.get(1, 0), 0)

    def test_if_inverse_on_true_path(self):
        """P: if x=0 then x+=10 else skip fi x=10.  P⁻¹(x=10) = x=0."""
        prog = parse(tokenize("""int x
procedure main
  if x = 0 then x += 10 else skip fi x = 10"""))
        inv = invert_program(prog)
        code = compile_program(inv)
        m = PISAMachine(code)
        m.mem[0] = 10
        m.run()
        self.assertEqual(m.mem.get(0, 0), 0)

    def test_procedure_call_inverse(self):
        """P: call inc (inc: x+=1).  P⁻¹ turns call into uncall.  P⁻¹(1) = 0."""
        prog = parse(tokenize("""int x
procedure inc
  x += 1
procedure main
  call inc"""))
        inv = invert_program(prog)
        code = compile_program(inv)
        m = PISAMachine(code)
        m.mem[0] = 1
        m.run()
        self.assertEqual(m.mem.get(0, 0), 0)

    def test_accumulate_inverse(self):
        """P accumulates: loop 3 times adding 2 to y.  P⁻¹(x=3,y=6) = (x=0,y=0)."""
        prog = parse(tokenize("""int x
int y
procedure main
  from x = 0
  do
    x += 1
    y += 2
  loop skip
  until x = 3"""))
        inv = invert_program(prog)
        code = compile_program(inv)
        m = PISAMachine(code)
        m.mem[0] = 3   # x
        m.mem[1] = 6   # y
        m.run()
        self.assertEqual(m.mem.get(0, 0), 0)
        self.assertEqual(m.mem.get(1, 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
