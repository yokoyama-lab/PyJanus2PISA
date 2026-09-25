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
        """if x = 1 then x += 10 else x -= 1 fi x = 11; x starts at 0.

        The exit assertion must discriminate the branches: it holds after the
        then branch (0+1 -> 11) and fails after the else branch (0 -> -1).
        Written as `fi x = -1` this program is invalid Janus — PyJanus reports
        "Assertion failed: should be false" — and is now rejected here too.
        """
        src = """int x
procedure main
  if x = 1 then x += 10 else x -= 1 fi x = 11"""
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

    def test_if_nonboolean_predicates(self):
        """Janus truth is `nonzero`: `if 5 ... fi 7` is valid and takes then.

        The path flag used to be XOR-ed with the raw values, leaving 5^7=2
        in a register (reported as garbage at FINISH).
        """
        src = """int x
procedure main
  if 5 then x += 1 else x += 2 fi 7"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 1)

    def test_if_nonboolean_variable_test(self):
        """`if y then ... fi x` with y = 3 (then) and y = 0 (else)."""
        for y0, expected in ((3, 10), (0, 20)):
            src = f"""int x
int y
procedure main
  y += {y0}
  if y then x += 10 else x += 20 fi x - 20"""
            m = compile_and_run(src)
            self.assertEqual(m.get_var(0), expected, f"y0={y0}")

    def test_if_nonboolean_violation_still_detected(self):
        """then path with a false (zero) exit assertion is still rejected."""
        src = """int x
procedure main
  if 5 then x += 1 else x += 2 fi x - 1"""
        with self.assertRaises(PISAError):
            compile_and_run(src)

    def test_if_constant_zero_assertion(self):
        """`fi 0` compiles to no evaluation code; the else path must still
        find the join label (it used to jump to an undefined `if_assert`)."""
        src = """int x
procedure main
  if 0 then x += 10 else x += 20 fi 0"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 20)


    def test_if_violation_not_masked_by_later_code(self):
        """A violated `fi` leaves 1 in the flag register, which is then freed.
        Without the check at the join, the next statement's `||` reuses that
        register as if it were zero and cancels the 1: the program finished
        clean (found by random testing against a reference evaluator)."""
        src = """int x
int y
int z
procedure main
  z += 2
  if 0 then skip else x += z fi x
  y += (z > 1) || x"""
        with self.assertRaises(PISAError):
            compile_and_run(src)


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


    def test_entry_assertion_violation_detected(self):
        """`from i = 0` entered with i = 5 used to run on (here: forever)."""
        src = """int i
procedure main
  i += 5
  from i = 0 do i += 1 loop skip until i = 8"""
        with self.assertRaises(PISAError) as cm:
            compile_and_run(src, max_steps=100_000)
        self.assertIn("garbage", str(cm.exception))

    def test_reentry_assertion_violation_detected(self):
        """tests/difftest_corpus/s10-loop-assert: `y = 0` still holds when the
        body is re-entered.  The flag used to be wiped with `XOR rt rt`."""
        src = """int y
int x
procedure main
  from y = 0 do x += 1 loop skip until x = 3"""
        with self.assertRaises(PISAError) as cm:
            compile_and_run(src)
        self.assertIn("garbage", str(cm.exception))

    def test_nonboolean_predicates(self):
        """Janus truth is `nonzero`: `from 1 - i` holds at entry (i = 0) and
        not on re-entry (i = 1); `until i - 1` is false at i = 1, true at 2."""
        src = """int i
procedure main
  from 1 - i do i += 1 loop skip until i - 1"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 2)

    def test_nonboolean_predicate_violation_detected(self):
        """`from 3 - i` is still true (2) on re-entry at i = 1."""
        src = """int i
procedure main
  from 3 - i do i += 1 loop skip until i = 3"""
        with self.assertRaises(PISAError):
            compile_and_run(src)

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


class TestLabelNamespace(unittest.TestCase):
    """Procedure names must not collide with labels the compiler makes up.

    Each procedure is called at least twice so it is not inlined away.
    Expected values are PyJanus's.
    """

    def test_user_proc_named_like_companion(self):
        """`f_inv` is also the companion label of `f` once `uncall f` appears."""
        src = """int x
int y
procedure f
  x += 1
procedure f_inv
  y += 100
procedure main
  call f_inv
  call f_inv
  call f
  call f
  call f
  uncall f"""
        m = compile_and_run(src)
        self.assertEqual((m.get_var(0), m.get_var(1)), (2, 200))

    def test_user_proc_named_like_fresh_label(self):
        """`if_false_1` is the first label `_gen_if` makes up."""
        src = """int c
procedure if_false_1
  c += 100
procedure main
  if c = 0 then c += 1 else skip fi c = 1
  call if_false_1
  call if_false_1"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 201)

    def test_underscores_in_several_names(self):
        """`f_`, `f` and `f__inv`: the escaping must stay injective."""
        src = """int c
procedure f_
  c += 1
procedure f
  c += 10
procedure f__inv
  c += 1000
procedure main
  call f_
  call f_
  uncall f_
  call f
  call f
  uncall f
  call f__inv
  call f__inv"""
        m = compile_and_run(src)
        self.assertEqual(m.get_var(0), 2011)

    def test_proc_labels_are_disjoint_from_generated_ones(self):
        """Every user-derived label has only even runs of `_`."""
        import re
        from codegen import _proc_label, _inv_proc_name
        names = ["f", "f_", "_f", "f_inv", "f__inv", "g_top", "if_false_1", "a___b"]
        user = {_proc_label(n) for n in names}
        generated = ({_proc_label(n) + s for n in names for s in ("_top", "_bot")}
                     | {_inv_proc_name(n) for n in names}
                     | {_inv_proc_name(n) + s for n in names for s in ("_top", "_bot")})
        self.assertEqual(len(user), len(names))
        self.assertFalse(user & generated)
        for lab in user:
            self.assertTrue(all(len(r) % 2 == 0 for r in re.findall(r"_+", lab)), lab)


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


class TestLogicalOperators(unittest.TestCase):
    """`&&` / `||` are logical (truth = nonzero, result 0/1), not bitwise."""

    def _value(self, expr, **vals):
        names = sorted(vals)
        decls = "".join(f"int {n}\n" for n in names)
        inits = "".join(f"  {n} += {vals[n]}\n" for n in names)
        src = f"{decls}int r\nprocedure main\n{inits}  r += {expr}"
        return compile_and_run(src).get_var(len(names))

    def test_and_of_nonboolean_operands(self):
        """1 && 2 used to evaluate to 1 & 2 = 0 (constant folding gave 1)."""
        self.assertEqual(self._value("a && b", a=1, b=2), 1)
        self.assertEqual(self._value("a && b", a=-3, b=4), 1)
        self.assertEqual(self._value("a && b", a=0, b=4), 0)
        self.assertEqual(self._value("a && b", a=5, b=0), 0)

    def test_or_of_nonboolean_operands(self):
        """1 || 2 used to evaluate to 1 | 2 = 3."""
        self.assertEqual(self._value("a || b", a=1, b=2), 1)
        self.assertEqual(self._value("a || b", a=0, b=-7), 1)
        self.assertEqual(self._value("a || b", a=0, b=0), 0)

    def test_matches_constant_folding(self):
        self.assertEqual(self._value("1 && 2"), 1)
        self.assertEqual(self._value("1 || 2"), 1)

    def test_comparison_operands_unchanged(self):
        self.assertEqual(self._value("(a < b) && (b < 9)", a=1, b=2), 1)
        self.assertEqual(self._value("(a > b) || (b > 9)", a=1, b=2), 0)

    def test_as_if_test(self):
        src = """int a
int b
int x
procedure main
  a += 1
  b += 2
  if a && b then x += 10 else x += 20 fi x = 10"""
        self.assertEqual(compile_and_run(src).get_var(2), 10)

    def test_nested_chain_fits_in_registers(self):
        """Each nonzero test costs one register (4 instructions: SLTX / NEG)."""
        self.assertEqual(self._value("((a && b) && c) && d", a=1, b=2, c=3, d=4), 1)
        self.assertEqual(self._value("((a || b) || c) || d", a=0, b=0, c=0, d=4), 1)

    def test_bitwise_or_operands_released(self):
        """ORX zeroes its source; keeping the sources as garbage until the end
        of the statement ran out of registers here."""
        self.assertEqual(self._value("((a | b) | (c | d)) + ((a | c) | (b | d))",
                                     a=1, b=2, c=4, d=8), 30)

    def test_compound_operand_garbage_cleared(self):
        """`e != 0` for a compound e clears the garbage of evaluating and
        uncomputing e at once; left until the end of the statement, it
        exhausted the registers here (the unfixed code compiled this)."""
        src = """int x[4]
int a
int b
int c
int d
int e
int r
procedure main
  x[1] += 5
  x[3] += 7
  a += 1
  c += 2
  d += 1
  e += 1
  r += x[a + b] && x[c + d] && e"""
        self.assertEqual(compile_and_run(src).get_var(9), 1)


class TestReservedNames(unittest.TestCase):
    def test_procedure_named_finish_rejected(self):
        """`finish` is where violated assertions jump; a procedure of that name
        used to shadow it, so its body never ran and nothing was reported."""
        from codegen import CodeGenError
        for name in ("finish", "start"):
            src = f"int x\nprocedure {name}\n  x += 1\nprocedure main\n  call {name}"
            with self.assertRaises(CodeGenError):
                compile_and_run(src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
