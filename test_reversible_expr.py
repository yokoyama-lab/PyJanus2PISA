"""Instruction-level reversibility of the generated code.

* `pisa.is_wf` — the local-invertibility predicate — and the Pendulum
  semantics of ORX / ANDX in the interpreter;
* every program of the corpus and of the cross-check tools compiles to
  well-formed code, unoptimised and optimised, forward and inverted;
* each operator on negative, zero and non-0/1 operands, nested expressions,
  and the clean-expression invariant (only the result register is dirty
  after gen_expr; running the code backwards clears it);
* round trips (P then P^-1, call then uncall).

See docs/EXPR_LOWERING.md for the lowering these tests pin down.
"""

import glob
import os
import random
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from lexer import tokenize                                  # noqa: E402
from parser import parse                                    # noqa: E402
from syntax import Var, Const, ArrayAccess                  # noqa: E402
from pisa import (                                          # noqa: E402
    LabeledInstr, is_wf, format_instr,
    ADD, SUB, NEG, XOR, ADDI, SUBI, XORI, ORX, ANDX, SLTX, EXCH,
    BRA, RBRA, BEQ, BNE, BGEZ, SWAPBR, DATA, START, FINISH,
)
from codegen import CodeGen, CodeGenError, compile_program  # noqa: E402
from inverse import invert_program                          # noqa: E402
from pisa_interp import PISAMachine                         # noqa: E402

CORPUS = os.path.join(ROOT, "tests", "difftest_corpus")

# Corpus programs that are not valid Janus and are rejected by codegen.
CORPUS_REJECTED = {"s9-self-assign.janus"}


def _bad(code):
    return [format_instr(li.instr) for li in code if not is_wf(li.instr)]


def _all_wf(test, prog, what):
    """Assert is_wf on the unoptimised and the optimised code of prog."""
    raw = CodeGen().gen_program(prog)
    test.assertEqual(_bad(raw), [], f"{what}: unoptimised")
    opt = compile_program(prog)
    test.assertEqual(_bad(opt), [], f"{what}: optimised")
    return len(raw) + len(opt)


def _run(src, max_steps=2_000_000):
    m = PISAMachine(compile_program(parse(tokenize(src))))
    m.run(max_steps=max_steps)          # raises if a register is left dirty
    return m


# ---------------------------------------------------------------------------
# is_wf and the ORX / ANDX semantics
# ---------------------------------------------------------------------------

class TestIsWf(unittest.TestCase):

    def test_register_register(self):
        for cls in (ADD, SUB, XOR):
            self.assertTrue(is_wf(cls("r3", "r4")))
            self.assertFalse(is_wf(cls("r3", "r3")), cls.__name__)

    def test_three_operand(self):
        for cls in (ANDX, ORX, SLTX):
            self.assertTrue(is_wf(cls("r3", "r4", "r5")))
            self.assertTrue(is_wf(cls("r3", "r4", "r4")))
            self.assertTrue(is_wf(cls("r3", "r4", "r0")))
            self.assertFalse(is_wf(cls("r3", "r3", "r4")), cls.__name__)
            self.assertFalse(is_wf(cls("r3", "r4", "r3")), cls.__name__)

    def test_exch(self):
        self.assertTrue(is_wf(EXCH("r3", "r4")))
        self.assertFalse(is_wf(EXCH("r3", "r3")))
        self.assertFalse(is_wf(EXCH("r0", "r4")))

    def test_always_wf(self):
        for i in (NEG("r3"), ADDI("r3", 5), SUBI("r3", -2), XORI("r3", 1),
                  BRA("l"), RBRA("l"), BEQ("r3", "r0", "l"),
                  BNE("r3", "r0", "l"), BGEZ("r3", "l"), SWAPBR("r2"),
                  DATA(3), START(), FINISH()):
            self.assertTrue(is_wf(i), format_instr(i))
        self.assertFalse(is_wf(SWAPBR("r0")))

    def test_format_three_operand(self):
        self.assertEqual(format_instr(ORX("r3", "r4", "r5")), "ORX r3 r4 r5")
        self.assertEqual(format_instr(ANDX("r3", "r4", "r5")), "ANDX r3 r4 r5")


def _exec(instr, regs, mem=None):
    """Run one data instruction on a fresh machine with the given registers."""
    m = PISAMachine([LabeledInstr("start", START()),
                     LabeledInstr("finish", FINISH())])
    for r, v in regs.items():
        m._write_reg(r, v)
    for a, v in (mem or {}).items():
        m.mem[a] = v
    m._exec_data(instr)
    return {r: m._read_reg(r) for r in regs}, dict(m.mem)


class TestPendulumOrxAndx(unittest.TestCase):
    """ORX / ANDX are Pendulum's rd ^= rs | rt and rd ^= rs & rt."""

    def test_orx(self):
        regs, _ = _exec(ORX("r3", "r4", "r5"), {"r3": 0b1000, "r4": 0b0110, "r5": 0b0011})
        self.assertEqual(regs, {"r3": 0b1000 ^ 0b0111, "r4": 0b0110, "r5": 0b0011})

    def test_andx(self):
        regs, _ = _exec(ANDX("r3", "r4", "r5"), {"r3": 0b1001, "r4": 0b0110, "r5": 0b0011})
        self.assertEqual(regs, {"r3": 0b1001 ^ 0b0010, "r4": 0b0110, "r5": 0b0011})

    def test_negative_operands(self):
        regs, _ = _exec(ORX("r3", "r4", "r5"), {"r3": 0, "r4": -4, "r5": 1})
        self.assertEqual(regs["r3"], -3)
        regs, _ = _exec(ANDX("r3", "r4", "r5"), {"r3": 0, "r4": -4, "r5": 6})
        self.assertEqual(regs["r3"], 4)

    def test_wf_instructions_are_locally_invertible(self):
        """instr ; invert(instr) is the identity on random states, for every
        well-formed data instruction (the property is_wf promises)."""
        R = random.Random(1)
        cg = CodeGen()
        regs = ["r0", "r3", "r4", "r5"]
        forms = []
        for a in regs:
            for b in regs:
                forms += [ADD(a, b), SUB(a, b), XOR(a, b), EXCH(a, b)]
                for c in regs:
                    forms += [ANDX(a, b, c), ORX(a, b, c), SLTX(a, b, c)]
            forms += [NEG(a), ADDI(a, 7), SUBI(a, 3), XORI(a, 5)]
        checked = 0
        for instr in forms:
            if not is_wf(instr):
                continue
            inv = cg._invert_instr(instr)
            for _ in range(20):
                state = {r: R.randint(-9, 9) for r in regs[1:]}
                # EXCH addresses: keep them small and non-negative
                mem = {a: R.randint(-9, 9) for a in range(10)}
                if isinstance(instr, EXCH) and instr.rs != "r0":
                    state[instr.rs] = R.randint(0, 9)   # a valid address
                m = PISAMachine([LabeledInstr("start", START()),
                                 LabeledInstr("finish", FINISH())])
                for r, v in state.items():
                    m._write_reg(r, v)
                m.mem.update(mem)
                m._exec_data(instr)
                m._exec_data(inv)
                after = {r: m._read_reg(r) for r in regs[1:]}
                self.assertEqual((after, {a: m.mem.get(a, 0) for a in range(10)}),
                                 (state, mem), format_instr(instr))
                checked += 1
        self.assertGreater(checked, 1000)


# ---------------------------------------------------------------------------
# Every known program compiles to well-formed code
# ---------------------------------------------------------------------------

def _corpus():
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.janus"))):
        with open(f) as fh:
            yield os.path.basename(f), fh.read()


def _tool_programs():
    """Programs of tools/pyjanus_crosscheck.py and the Rocq cross-checks."""
    import pyjanus_crosscheck
    import rocq_loop_crosscheck
    import rocq_proc_crosscheck
    for name, src in pyjanus_crosscheck.PROGRAMS.items():
        yield f"pyjanus:{name}", src
    for name, body in rocq_loop_crosscheck.PROGRAMS.items():
        yield f"rocq_loop:{name}", rocq_loop_crosscheck.DECLS + body
    for name, (_n, _m, procs, _k) in rocq_proc_crosscheck.PROGRAMS.items():
        yield f"rocq_proc:{name}", rocq_proc_crosscheck.DECLS + procs


class TestAllProgramsWellFormed(unittest.TestCase):

    def _check(self, name, src):
        prog = parse(tokenize(src))
        _all_wf(self, prog, name)
        _all_wf(self, invert_program(prog), name + " (inverted)")

    def test_corpus(self):
        n = 0
        for name, src in _corpus():
            with self.subTest(name=name):
                if name in CORPUS_REJECTED:
                    with self.assertRaises(CodeGenError):
                        compile_program(parse(tokenize(src)))
                    continue
                self._check(name, src)
                n += 1
        self.assertGreaterEqual(n, 25)

    def test_tool_programs(self):
        n = 0
        for name, src in _tool_programs():
            with self.subTest(name=name):
                self._check(name, src)
                n += 1
        self.assertGreaterEqual(n, 35)

    def test_rocq_diff_programs(self):
        """The programs of tools/rocq_diff.py (needs rocq/driver)."""
        driver = os.path.join(ROOT, "rocq", "driver")
        if not os.path.exists(driver):
            self.skipTest("rocq/driver not built (make -C rocq -f Makefile.driver)")
        import rocq_diff
        out = subprocess.run([driver], capture_output=True, text=True,
                             check=True).stdout
        cases = rocq_diff.parse_driver_output(out)
        self.assertGreater(len(cases), 5)
        for case in cases:
            with self.subTest(name=case["name"]):
                self._check(case["name"], case["source"])

    def test_random_expressions(self):
        """Deep random expressions over every operator (the stress shape)."""
        R = random.Random(2026)
        for i in range(150):
            e = _rand_expr(R, 4, ["a", "b", "c", "x[a & 3]", "2", "0 - 3"])
            src = ("int x[4]\nint a\nint b\nint c\nint r\nprocedure main\n"
                   f"  r += {e}\n  if {e} then r += 1 else r += 2 fi r = 1 || r = 2")
            with self.subTest(i=i, e=e):
                _all_wf(self, parse(tokenize(src)), e)


# ---------------------------------------------------------------------------
# Semantics of each operator
# ---------------------------------------------------------------------------

OPS = ['+', '-', '^', '&', '|', '&&', '||', '=', '!=', '<', '>', '<=', '>=']


def _ref(op, x, y):
    return {'+': x + y, '-': x - y, '^': x ^ y, '&': x & y, '|': x | y,
            '*': x * y,
            '&&': int(bool(x) and bool(y)), '||': int(bool(x) or bool(y)),
            '=': int(x == y), '!=': int(x != y), '<': int(x < y),
            '>': int(x > y), '<=': int(x <= y), '>=': int(x >= y)}[op]


def _init(var, v):
    return f"{var} += {v}" if v >= 0 else f"{var} -= {-v}"


def _rand_expr(R, d, leaves, ops=OPS):
    if d == 0 or R.random() < 0.2:
        return R.choice(leaves)
    return f"({_rand_expr(R, d - 1, leaves, ops)} {R.choice(ops)} {_rand_expr(R, d - 1, leaves, ops)})"


def _eval(e, env):
    if isinstance(e, Const):
        return e.value
    if isinstance(e, Var):
        return env[e.name]
    if isinstance(e, ArrayAccess):
        return env[e.name][_eval(e.index, env)]
    return _ref(e.op, _eval(e.left, env), _eval(e.right, env))


VALUES = [-5, -2, -1, 0, 1, 2, 3, 7]


class TestOperatorSemantics(unittest.TestCase):
    """r += a op b for every operator on a grid of values (incl. negative
    and non-0/1 operands of && / ||); the interpreter also checks that no
    register is left dirty at FINISH."""

    def _check_op(self, op, fmt):
        for x in VALUES:
            for y in VALUES:
                src = ("int a\nint b\nint r\nprocedure main\n"
                       f"  {_init('a', x)}\n  {_init('b', y)}\n  r += {fmt}")
                with self.subTest(op=op, a=x, b=y, e=fmt):
                    m = _run(src)
                    self.assertEqual(m.get_var(2), _ref(op, x, y))

    def test_var_var(self):
        for op in OPS:
            self._check_op(op, f"a {op} b")

    def test_nested_operands(self):
        # the operands are themselves expressions (in-place and combine
        # shapes inside each other)
        for op in OPS:
            for x in VALUES:
                for y in VALUES:
                    src = ("int a\nint b\nint r\nprocedure main\n"
                           f"  {_init('a', x)}\n  {_init('b', y)}\n"
                           f"  r += (a - b) {op} (b + 1)")
                    with self.subTest(op=op, a=x, b=y):
                        self.assertEqual(_run(src).get_var(2),
                                         _ref(op, x - y, y + 1))

    def test_constant_operands(self):
        for op in OPS:
            for k in (-3, 0, 1, 2):
                ks = str(k) if k >= 0 else f"(0 - {-k})"
                for x in VALUES:
                    for src_e, want in ((f"a {op} {ks}", _ref(op, x, k)),
                                        (f"{ks} {op} a", _ref(op, k, x))):
                        src = ("int a\nint r\nprocedure main\n"
                               f"  {_init('a', x)}\n  r += {src_e}")
                        with self.subTest(e=src_e, a=x):
                            self.assertEqual(_run(src).get_var(1), want)

    def test_multiplication(self):
        for k in (-9, -1, 0, 1, 2, 3, 6, 255, -256):
            ks = str(k) if k >= 0 else f"(0 - {-k})"
            for x in (-7, -1, 0, 1, 5):
                for e in (f"a * {ks}", f"{ks} * a", f"(a + 1) * {ks}"):
                    want = x * k if "+ 1" not in e else (x + 1) * k
                    src = f"int a\nint r\nprocedure main\n  {_init('a', x)}\n  r += {e}"
                    with self.subTest(e=e, a=x):
                        self.assertEqual(_run(src).get_var(1), want)

    def test_logical_non_boolean(self):
        # Janus truth: nonzero is true; results are 0/1
        cases = [("a && b", 2, 4, 1), ("a && b", -1, 3, 1), ("a && b", 2, 0, 0),
                 ("a || b", 2, 4, 1), ("a || b", 0, -6, 1), ("a || b", 0, 0, 0),
                 ("(a && b) + (a || b)", 5, -5, 2), ("(a & b) && (a | b)", 6, 3, 1),
                 ("(a & b) && (a | b)", 4, 3, 0)]
        for e, x, y, want in cases:
            src = ("int a\nint b\nint r\nprocedure main\n"
                   f"  {_init('a', x)}\n  {_init('b', y)}\n  r += {e}")
            with self.subTest(e=e, a=x, b=y):
                self.assertEqual(_run(src).get_var(2), want)

    def test_array_reads(self):
        src = ("int x[4]\nint i\nint r\nint s\nprocedure main\n"
               "  x[0] += 3\n  x[1] -= 2\n  x[2] += 7\n"
               "  i += 2\n  r += x[i] + x[i - 1] * 3\n"
               "  s += (x[x[3]] < x[i]) && (x[1] != 0)")
        m = _run(src)
        self.assertEqual(m.get_var(5), 7 + (-2) * 3)
        self.assertEqual(m.get_var(6), 1)

    def test_random_nested_expressions(self):
        R = random.Random(7)
        leaves = ["a", "b", "c", "2", "(0 - 3)", "1"]
        for i in range(250):
            e = _rand_expr(R, 3, leaves)
            env = {"a": R.randint(-4, 4), "b": R.randint(-4, 4), "c": R.randint(-4, 4)}
            init = "\n  ".join(_init(v, env[v]) for v in "abc")
            src = f"int a\nint b\nint c\nint r\nprocedure main\n  {init}\n  r ^= {e}"
            want = _eval(parse(tokenize(src)).procs[0].body.stmts[-1].expr, env)
            with self.subTest(i=i, e=e, env=env):
                self.assertEqual(_run(src).get_var(3), want)

    def test_predicates(self):
        # if / from tests go through _as_flag; non-0/1 predicates included
        src = ("int a\nint b\nint r\nint i\nprocedure main\n  a -= 3\n  b += 5\n"
               "  if (a < 0) && (b >= 5) then r += 1 else r += 2 fi r = 1\n"
               "  if a + b then r += 10 else skip fi r > 5\n"
               "  from (i = 0) || (i > 100) do i += 1 loop skip until (i = 3) && (a != b)")
        m = _run(src)
        self.assertEqual((m.get_var(2), m.get_var(3)), (11, 3))


class TestCleanExpression(unittest.TestCase):
    """gen_expr leaves only its result register dirty, and its code run
    backwards clears that register again (no garbage, no clearing)."""

    def _machine(self, code, mem):
        prog = ([LabeledInstr(None, DATA(v)) for v in mem]
                + [LabeledInstr("start", START())] + code
                + [LabeledInstr("finish", FINISH())])
        m = PISAMachine(prog)
        m.check_clean = False
        m.run()
        return m

    def test_random(self):
        R = random.Random(11)
        leaves = ["a", "b", "x[a & 3]", "x[b & 3]", "3", "(0 - 2)"]
        ops = OPS + ["*"]
        tried = 0
        while tried < 200:
            src = _rand_expr(R, 3, leaves, ops)
            e = parse(tokenize(f"int a\nint b\nint x[4]\nint r\nprocedure main\n  r += {src}")
                      ).procs[0].body.expr
            cg = CodeGen()
            cg._var_offsets = {"a": 0, "b": 1, "x": 2, "r": 6}
            try:
                code, r = cg.gen_expr(e)
            except CodeGenError:        # var * var
                continue
            tried += 1
            self.assertEqual(cg.reg.free, CodeGen().reg.free - {r})
            self.assertEqual(_bad(code), [])
            mem = [R.randint(-5, 5), R.randint(-5, 5)] + [R.randint(-9, 9) for _ in range(4)]
            env = {"a": mem[0], "b": mem[1], "x": mem[2:6]}
            m = self._machine(code, mem)
            dirty = {g: m._read_reg(g) for g in m._GP_REGS if m._read_reg(g) and g != r}
            with self.subTest(e=src):
                self.assertEqual(dirty, {})
                self.assertEqual(m._read_reg(r), _eval(e, env))
                self.assertEqual([m.mem.get(i, 0) for i in range(6)], mem)
                m2 = self._machine(code + cg._reverse_code(code), mem)
                self.assertTrue(all(m2._read_reg(g) == 0 for g in m2._GP_REGS))

    def test_no_register_clearing_emitted(self):
        # the pattern the old garbage clearing produced
        src = ("int a\nint b\nint x[2]\nint r\nprocedure main\n"
               "  r += ((a < b) || (a = 2)) && (x[a != b] >= 1)")
        code = compile_program(parse(tokenize(src)))
        self.assertFalse(any(isinstance(li.instr, XOR) and li.instr.rd == li.instr.rs
                             for li in code))

    def test_deep_expression_fits_registers(self):
        # used to raise RegAllocError (garbage kept every operand register)
        e = "a"
        for i in range(12):
            e = f"({e} {OPS[i % len(OPS)]} (b - {i}))"
        src = f"int a\nint b\nint r\nprocedure main\n  a += 3\n  b -= 2\n  r += {e}"
        env = {"a": 3, "b": -2}
        want = _eval(parse(tokenize(src)).procs[0].body.stmts[-1].expr, env)
        self.assertEqual(_run(src, max_steps=10_000_000).get_var(2), want)


class TestRoundTrips(unittest.TestCase):

    PROGS = [
        "r += (a < b) + (a >= b) * 4 - (a = b)",
        "r ^= (a && b) | ((a || c) & 6)",
        "r -= (a != b) && (b <= c) || (c > a)",
        "x[a & 1] += (b - c) ^ (a * 3)",
        "if a != b then r += a else r += b fi (r = a) && (a != b)",
        "from i = 0 do i += 1 ; r += (i < 2) + (a > i) loop skip until (i >= 3) && (b <= 0)",
    ]

    def _src(self, body, extra=""):
        return ("int a\nint b\nint c\nint i\nint r\nint x[2]\n"
                f"procedure f\n  {body.replace(' ; ', chr(10) + '  ')}\n"
                "procedure main\n  a += 4\n  b -= 3\n  c += 1\n  x[1] += 5\n"
                f"  call f\n{extra}")

    def test_call_uncall(self):
        for body in self.PROGS:
            with self.subTest(body=body):
                m0 = _run(self._src(body, ""))
                m1 = _run(self._src(body, "  uncall f\n"))
                start = _run("int a\nint b\nint c\nint i\nint r\nint x[2]\n"
                             "procedure main\n  a += 4\n  b -= 3\n  c += 1\n  x[1] += 5")
                self.assertEqual([m1.get_var(k) for k in range(7)],
                                 [start.get_var(k) for k in range(7)])
                self.assertNotEqual([m0.get_var(k) for k in range(7)],
                                    [start.get_var(k) for k in range(7)])

    def test_inverse_program(self):
        from test_inverse import round_trip
        for body in self.PROGS:
            src = ("int a\nint b\nint c\nint i\nint r\nint x[2]\n"
                   f"procedure main\n  {body.replace(' ; ', chr(10) + '  ')}")
            with self.subTest(body=body):
                self.assertTrue(all(v == 0 for v in round_trip(src).values()))


if __name__ == "__main__":
    unittest.main()
