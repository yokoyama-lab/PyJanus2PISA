#!/usr/bin/env python3
"""Test suite for the PISA interpreter (pisa_interp.py).

Each test compiles a small Janus program and runs it on PISAMachine,
then checks the final memory state.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from lexer import tokenize
from parser import parse
from codegen import compile_program
from pisa_interp import PISAMachine, PISAError


def compile_and_run(src: str, max_steps: int = 1_000_000) -> PISAMachine:
    """Compile Janus source and run it; return the machine after FINISH."""
    prog = parse(tokenize(src))
    code = compile_program(prog)
    machine = PISAMachine(code)
    machine.run(max_steps=max_steps)
    return machine


class TestSimpleArithmetic(unittest.TestCase):
    """Basic single-variable arithmetic."""

    def test_increment(self):
        """x += 1 → x = 1."""
        m = compile_and_run("int x\nprocedure main\n  x += 1")
        self.assertEqual(m.get_var(0), 1)

    def test_decrement(self):
        """x -= 1 → x = -1."""
        m = compile_and_run("int x\nprocedure main\n  x -= 1")
        self.assertEqual(m.get_var(0), -1)

    def test_xor_assign(self):
        """x ^= 3 → x = 3."""
        m = compile_and_run("int x\nprocedure main\n  x ^= 3")
        self.assertEqual(m.get_var(0), 3)

    def test_add_const_5(self):
        """x += 5 → x = 5."""
        m = compile_and_run("int x\nprocedure main\n  x += 5")
        self.assertEqual(m.get_var(0), 5)

    def test_two_increments(self):
        """x += 1; x += 1 → x = 2."""
        src = """int x
procedure main
  x += 1
  x += 1"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 2)

    def test_increment_then_decrement(self):
        """x += 3; x -= 1 → x = 2."""
        src = """int x
procedure main
  x += 3
  x -= 1"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 2)

    def test_two_variables(self):
        """x += 1; y += 2 → x=1, y=2."""
        src = """int x
int y
procedure main
  x += 1
  y += 2"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 1)   # x at offset 0
        self.assertEqual(m.get_var(1), 2)   # y at offset 1


class TestSwap(unittest.TestCase):
    """Swap statement."""

    def test_swap_both_zero(self):
        """x <=> y where both are 0 → both remain 0."""
        src = """int x
int y
procedure main
  x <=> y"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 0)
        self.assertEqual(m.get_var(1), 0)

    def test_swap_nonzero(self):
        """Swap after setting x=3, y=7 → x=7, y=3."""
        src = """int x
int y
procedure main
  x += 3
  y += 7
  x <=> y"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 7)   # x
        self.assertEqual(m.get_var(1), 3)   # y

    def test_swap_asymmetric(self):
        """x=10, y=0 swap → x=0, y=10."""
        src = """int x
int y
procedure main
  x += 10
  x <=> y"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 0)
        self.assertEqual(m.get_var(1), 10)


class TestIf(unittest.TestCase):
    """If-then-else statement."""

    def test_if_true_branch(self):
        """if x = 0 then x += 10 else skip fi x = 10 → x = 10."""
        src = """int x
procedure main
  if x = 0 then x += 10 else skip fi x = 10"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 10)

    def test_if_else_branch(self):
        """if x = 1 then x += 10 else x -= 1 fi x = -1; x starts at 0."""
        src = """int x
procedure main
  if x = 1 then x += 10 else x -= 1 fi x = -1"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), -1)

    def test_if_nested(self):
        """Two sequential ifs."""
        src = """int x
int y
procedure main
  if x = 0 then x += 5 else skip fi x = 5
  if y = 0 then y += 3 else skip fi y = 3"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 5)
        self.assertEqual(m.get_var(1), 3)


class TestFrom(unittest.TestCase):
    """From-do-loop-until loop."""

    def test_count_to_5(self):
        """from x = 0 do x += 1 loop skip until x = 5 → x = 5."""
        src = """int x
procedure main
  from x = 0
  do x += 1
  loop skip
  until x = 5"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 5)

    def test_count_to_1(self):
        """Loop that executes once → x = 1."""
        src = """int x
procedure main
  from x = 0
  do x += 1
  loop skip
  until x = 1"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 1)

    def test_loop_with_loop_body(self):
        """Loop with non-trivial loop body.

        from x = 0
        do   x += 1
        loop x += 0   (skip)
        until x = 3
        → x = 3
        """
        src = """int x
procedure main
  from x = 0
  do x += 1
  loop skip
  until x = 3"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 3)

    def test_loop_accumulate(self):
        """Accumulate sum using two variables.

        Loop x from 0 to 3, each iteration y += 2: y = 6.
        """
        src = """int x
int y
procedure main
  from x = 0
  do
    x += 1
    y += 2
  loop skip
  until x = 3"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 3)
        self.assertEqual(m.get_var(1), 6)


class TestProcCall(unittest.TestCase):
    """Procedure calls."""

    def test_call_once(self):
        """call inc → x = 1."""
        src = """int x
procedure inc
  x += 1
procedure main
  call inc"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 1)

    def test_call_twice(self):
        """call inc; call inc → x = 2."""
        src = """int x
procedure inc
  x += 1
procedure main
  call inc
  call inc"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 2)

    def test_call_three_times(self):
        """call inc × 3 → x = 3."""
        src = """int x
procedure inc
  x += 1
procedure main
  call inc
  call inc
  call inc"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 3)

    def test_nested_call(self):
        """Procedure calling another procedure."""
        src = """int x
procedure add2
  x += 2
procedure double_add2
  call add2
  call add2
procedure main
  call double_add2"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 4)

    def test_call_with_if(self):
        """Procedure with if inside."""
        src = """int x
procedure maybe_inc
  if x = 0 then x += 1 else skip fi x = 1
procedure main
  call maybe_inc"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 1)

    def test_call_loop_proc(self):
        """Procedure containing a loop."""
        src = """int x
procedure count_to_3
  from x = 0
  do x += 1
  loop skip
  until x = 3
procedure main
  call count_to_3"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 3)


class TestSkip(unittest.TestCase):
    """Skip statement."""

    def test_skip_alone(self):
        """Skip does nothing → x remains 0."""
        src = "int x\nprocedure main\n  skip"
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 0)

    def test_skip_in_seq(self):
        """skip before an increment → x = 1."""
        src = """int x
procedure main
  skip
  x += 1"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 1)


class TestArray(unittest.TestCase):
    """Array access."""

    def test_array_assign(self):
        """a[0] += 5 → a[0] = 5."""
        src = """int a[2]
procedure main
  a[0] += 5"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 5)
        self.assertEqual(m.get_var(1), 0)

    def test_array_two_elements(self):
        """a[0] += 1; a[1] += 2 → a[0]=1, a[1]=2."""
        src = """int a[2]
procedure main
  a[0] += 1
  a[1] += 2"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 1)
        self.assertEqual(m.get_var(1), 2)


class TestMemoryModel(unittest.TestCase):
    """Test r0 is always zero and other memory edge cases."""

    def test_r0_is_zero(self):
        """After any computation, r0 must be 0 (checked via EXCH with mem)."""
        src = """int x
int y
procedure main
  x += 7
  y += 3"""
        m = compile_and_run(src)
        # r0 register should be 0
        self.assertEqual(m.regs[0], 0)

    def test_large_value(self):
        """Test with a larger constant."""
        src = """int x
procedure main
  x += 1000"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 1000)

    def test_negative_result(self):
        """x -= 42 → x = -42."""
        src = """int x
procedure main
  x -= 42"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), -42)


class TestExpressions(unittest.TestCase):
    """Test expression evaluation in assignments."""

    def test_add_expr(self):
        """y += x + x where x=3 → y=6."""
        src = """int x
int y
procedure main
  x += 3
  y += x + x"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 3)
        self.assertEqual(m.get_var(1), 6)

    def test_sub_expr(self):
        """y += x - 1 where x=5 → y=4."""
        src = """int x
int y
procedure main
  x += 5
  y += x - 1"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 5)
        self.assertEqual(m.get_var(1), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
