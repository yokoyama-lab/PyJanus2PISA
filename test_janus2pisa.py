#!/usr/bin/env python3
"""Test suite for janus2pisa compiler."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from lexer import tokenize, LexError
from parser import parse, ParseError
from syntax import *
from pisa import *
from codegen import compile_program, CodeGen, CodeGenError


class TestLexer(unittest.TestCase):
    """Test tokenizer."""

    def test_empty(self):
        tokens = tokenize("")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].kind, "EOF")

    def test_keywords(self):
        src = "procedure call uncall if then else fi from do loop until skip"
        tokens = tokenize(src)
        kinds = [t.kind for t in tokens[:-1]]  # exclude EOF
        self.assertEqual(kinds, [
            "PROCEDURE", "CALL", "UNCALL",
            "IF", "THEN", "ELSE", "FI",
            "FROM", "DO", "LOOP", "UNTIL", "SKIP",
        ])

    def test_integer(self):
        tokens = tokenize("42 0 123")
        self.assertEqual(tokens[0].kind, "INT")
        self.assertEqual(tokens[0].value, "42")
        self.assertEqual(tokens[1].value, "0")
        self.assertEqual(tokens[2].value, "123")

    def test_operators(self):
        tokens = tokenize("+= -= ^= <=> + - * = != < > <= >= && || ^ & |")
        ops = [t.value for t in tokens if t.kind == "OP"]
        self.assertEqual(ops, [
            "+=", "-=", "^=", "<=>",
            "+", "-", "*", "=", "!=", "<", ">", "<=", ">=",
            "&&", "||", "^", "&", "|",
        ])

    def test_identifiers(self):
        tokens = tokenize("x foo bar_baz n1")
        idents = [t.value for t in tokens if t.kind == "IDENT"]
        self.assertEqual(idents, ["x", "foo", "bar_baz", "n1"])

    def test_comment(self):
        tokens = tokenize("x += 1 // this is a comment\ny -= 2")
        idents = [t.value for t in tokens if t.kind == "IDENT"]
        self.assertEqual(idents, ["x", "y"])

    def test_brackets(self):
        tokens = tokenize("x[0]")
        values = [t.value for t in tokens[:-1]]
        self.assertEqual(values, ["x", "[", "0", "]"])

    def test_line_tracking(self):
        tokens = tokenize("x\ny\nz")
        self.assertEqual(tokens[0].line, 1)
        self.assertEqual(tokens[1].line, 2)
        self.assertEqual(tokens[2].line, 3)


class TestParser(unittest.TestCase):
    """Test parser."""

    def _parse(self, src):
        return parse(tokenize(src))

    def test_skip(self):
        prog = self._parse("procedure main\n  skip")
        self.assertIsInstance(prog.procs[0].body, Skip)

    def test_assign_var(self):
        prog = self._parse("int x\nprocedure main\n  x += 1")
        body = prog.procs[0].body
        self.assertIsInstance(body, AssignVar)
        self.assertEqual(body.var, "x")
        self.assertEqual(body.op, "+=")
        self.assertIsInstance(body.expr, Const)
        self.assertEqual(body.expr.value, 1)

    def test_assign_sub(self):
        prog = self._parse("int x\nprocedure main\n  x -= 5")
        body = prog.procs[0].body
        self.assertEqual(body.op, "-=")

    def test_assign_xor(self):
        prog = self._parse("int x\nprocedure main\n  x ^= 3")
        body = prog.procs[0].body
        self.assertEqual(body.op, "^=")

    def test_assign_arr(self):
        prog = self._parse("int a[10]\nprocedure main\n  a[0] += 1")
        body = prog.procs[0].body
        self.assertIsInstance(body, AssignArr)
        self.assertEqual(body.var, "a")

    def test_if(self):
        prog = self._parse(
            "int x\n"
            "procedure main\n"
            "  if x = 0 then\n"
            "    x += 1\n"
            "  else\n"
            "    skip\n"
            "  fi x != 0"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, If)
        self.assertIsInstance(body.test, BinOp)
        self.assertEqual(body.test.op, "=")

    def test_from_loop(self):
        prog = self._parse(
            "int x\n"
            "procedure main\n"
            "  from x = 0 do\n"
            "    x += 1\n"
            "  loop\n"
            "    skip\n"
            "  until x = 10"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, From)
        self.assertIsInstance(body.from_, BinOp)
        self.assertIsInstance(body.until, BinOp)

    def test_call_uncall(self):
        prog = self._parse(
            "procedure foo\n"
            "  skip\n"
            "procedure main\n"
            "  call foo\n"
            "  uncall foo"
        )
        self.assertEqual(len(prog.procs), 2)
        body = prog.procs[1].body
        self.assertIsInstance(body, Seq)
        self.assertIsInstance(body.stmts[0], Call)
        self.assertIsInstance(body.stmts[1], Uncall)

    def test_sequence(self):
        prog = self._parse(
            "int x\n"
            "procedure main\n"
            "  x += 1\n"
            "  x += 2\n"
            "  x += 3"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, Seq)
        self.assertEqual(len(body.stmts), 3)

    def test_var_decl_with_init(self):
        prog = self._parse("int x = 5\nprocedure main\n  skip")
        self.assertEqual(prog.vars[0].init, [5])

    def test_array_decl(self):
        prog = self._parse("int a[3]\nprocedure main\n  skip")
        self.assertEqual(prog.vars[0].size, 3)
        self.assertEqual(prog.vars[0].init, [0, 0, 0])

    def test_multiple_vars(self):
        prog = self._parse("int x\nint y\nint z\nprocedure main\n  skip")
        self.assertEqual(len(prog.vars), 3)

    def test_expr_precedence(self):
        prog = self._parse("int x\nprocedure main\n  x += 1 + 2 * 3")
        body = prog.procs[0].body
        # Should parse as 1 + (2 * 3)
        e = body.expr
        self.assertIsInstance(e, BinOp)
        self.assertEqual(e.op, "+")
        self.assertIsInstance(e.right, BinOp)
        self.assertEqual(e.right.op, "*")

    def test_comparison(self):
        prog = self._parse("int x\nprocedure main\n  if x < 5 then skip else skip fi x >= 5")
        body = prog.procs[0].body
        self.assertIsInstance(body, If)
        self.assertEqual(body.test.op, "<")
        self.assertEqual(body.fi.op, ">=")

    def test_swap(self):
        prog = self._parse("int x\nint y\nprocedure main\n  x <=> y")
        body = prog.procs[0].body
        self.assertIsInstance(body, Swap)
        self.assertEqual(body.lhs, "x")
        self.assertEqual(body.rhs, "y")

    def test_nested_if(self):
        prog = self._parse(
            "int x\nint y\n"
            "procedure main\n"
            "  if x = 0 then\n"
            "    if y = 0 then\n"
            "      x += 1\n"
            "    else\n"
            "      skip\n"
            "    fi y != 0\n"
            "  else\n"
            "    skip\n"
            "  fi x != 0"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, If)
        self.assertIsInstance(body.then_, If)

    def test_paren_expr(self):
        prog = self._parse("int x\nprocedure main\n  x += (1 + 2) * 3")
        body = prog.procs[0].body
        e = body.expr
        self.assertEqual(e.op, "*")
        self.assertEqual(e.left.op, "+")


class TestPISA(unittest.TestCase):
    """Test PISA instruction formatting."""

    def test_format_add(self):
        self.assertEqual(format_instr(ADD("r3", "r4")), "ADD r3 r4")

    def test_format_addi(self):
        self.assertEqual(format_instr(ADDI("r3", 5)), "ADDI r3 5")

    def test_format_bra(self):
        self.assertEqual(format_instr(BRA("main")), "BRA main")

    def test_format_beq(self):
        self.assertEqual(format_instr(BEQ("r3", "r0", "L1")), "BEQ r3 r0 L1")

    def test_format_data(self):
        self.assertEqual(format_instr(DATA(0)), "DATA 0")

    def test_format_start(self):
        self.assertEqual(format_instr(START()), "START")

    def test_labeled(self):
        prog = [
            LabeledInstr("main", ADD("r3", "r4")),
            LabeledInstr(None, SUB("r3", "r5")),
        ]
        text = print_program(prog)
        self.assertIn("main: ADD r3 r4", text)
        self.assertIn("SUB r3 r5", text)


class TestCodeGen(unittest.TestCase):
    """Test code generation."""

    def _compile(self, src):
        tokens = tokenize(src)
        prog = parse(tokens)
        return compile_program(prog)

    def test_skip(self):
        """Compiling 'skip' produces a valid PISA program."""
        code = self._compile("procedure main\n  skip")
        text = print_program(code)
        self.assertIn("START", text)
        self.assertIn("FINISH", text)
        self.assertIn("BRA main", text)

    def test_assign_const(self):
        """x += 1 generates PISA code."""
        code = self._compile("int x\nprocedure main\n  x += 1")
        text = print_program(code)
        self.assertIn("START", text)
        self.assertIn("DATA 0", text)
        # Should contain ADDI for the constant 1
        self.assertIn("ADDI", text)

    def test_assign_with_init(self):
        """Variable with initial value."""
        code = self._compile("int x = 5\nprocedure main\n  x += 1")
        text = print_program(code)
        self.assertIn("DATA 5", text)

    def test_proc_structure(self):
        """Procedure has proper entry/exit structure."""
        code = self._compile("procedure foo\n  skip\nprocedure main\n  call foo")
        text = print_program(code)
        self.assertIn("foo_top:", text)
        self.assertIn("foo_bot:", text)
        self.assertIn("SWAPBR r2", text)
        self.assertIn("BRA foo", text)

    def test_call_uncall(self):
        """Call generates BRA, uncall generates RBRA."""
        code = self._compile(
            "procedure foo\n  skip\n"
            "procedure main\n  call foo\n  uncall foo"
        )
        text = print_program(code)
        instrs = [li.instr for li in code]
        # Main body should have BRA foo and RBRA foo
        bras = [i for i in instrs if isinstance(i, BRA) and i.label == "foo"]
        rbras = [i for i in instrs if isinstance(i, RBRA) and i.label == "foo"]
        self.assertTrue(len(bras) >= 1)
        self.assertTrue(len(rbras) >= 1)

    def test_if_generates_beq(self):
        """If statement generates conditional branch."""
        code = self._compile(
            "int x\n"
            "procedure main\n"
            "  if x = 0 then\n"
            "    x += 1\n"
            "  else\n"
            "    skip\n"
            "  fi x != 0"
        )
        text = print_program(code)
        self.assertIn("BEQ", text)

    def test_from_loop_generates_labels(self):
        """From loop generates proper label structure."""
        code = self._compile(
            "int x\n"
            "procedure main\n"
            "  from x = 0 do\n"
            "    x += 1\n"
            "  loop\n"
            "    skip\n"
            "  until x = 10"
        )
        text = print_program(code)
        self.assertIn("BEQ", text)
        self.assertIn("BRA", text)

    def test_multiple_vars(self):
        """Multiple variables get separate DATA slots."""
        code = self._compile(
            "int x\nint y\nint z\n"
            "procedure main\n  x += 1\n  y += 2\n  z += 3"
        )
        text = print_program(code)
        # Should have 3 DATA declarations
        data_count = sum(1 for li in code if isinstance(li.instr, DATA))
        self.assertEqual(data_count, 3)

    def test_array_data(self):
        """Array generates multiple DATA entries."""
        code = self._compile("int a[5]\nprocedure main\n  a[0] += 1")
        data_count = sum(1 for li in code if isinstance(li.instr, DATA))
        self.assertEqual(data_count, 5)

    def test_var_xor_assign(self):
        """x ^= e generates XOR instruction."""
        code = self._compile("int x\nint y\nprocedure main\n  x ^= y")
        instrs = [li.instr for li in code]
        xors = [i for i in instrs if isinstance(i, XOR)]
        self.assertTrue(len(xors) >= 1)

    def test_swap(self):
        """Swap generates EXCH instructions."""
        code = self._compile("int x\nint y\nprocedure main\n  x <=> y")
        instrs = [li.instr for li in code]
        exchs = [i for i in instrs if isinstance(i, EXCH)]
        self.assertTrue(len(exchs) >= 2)

    def test_output_format(self):
        """Output is valid text with labels and instructions."""
        code = self._compile("procedure main\n  skip")
        text = print_program(code)
        lines = text.strip().split("\n")
        self.assertTrue(len(lines) > 0)
        # Each line should have content
        for line in lines:
            self.assertTrue(len(line.strip()) > 0)


class TestRegAlloc(unittest.TestCase):
    """Test register allocator."""

    def test_alloc_free(self):
        from regalloc import RegAlloc
        ra = RegAlloc()
        r = ra.alloc()
        self.assertTrue(r.startswith("r"))
        self.assertFalse(ra.is_free(r))
        ra.free_reg(r)
        self.assertTrue(ra.is_free(r))

    def test_commit(self):
        from regalloc import RegAlloc
        ra = RegAlloc()
        r = ra.alloc()
        ra.commit_reg(r)
        self.assertTrue(ra.is_commit(r))
        self.assertFalse(ra.is_free(r))

    def test_garbage(self):
        from regalloc import RegAlloc
        ra = RegAlloc()
        r = ra.alloc()
        ra.commit_reg(r)
        ra.to_garbage(r)
        self.assertTrue(ra.is_garbage(r))
        self.assertFalse(ra.is_commit(r))

    def test_reserved_not_allocable(self):
        from regalloc import RegAlloc
        ra = RegAlloc()
        # r0, r1, r2 should never be allocated
        for _ in range(29):  # r3..r31
            r = ra.alloc()
            self.assertNotIn(r, {"r0", "r1", "r2"})


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests: source → PISA text."""

    def _e2e(self, src):
        tokens = tokenize(src)
        prog = parse(tokens)
        code = compile_program(prog)
        return print_program(code)

    def test_empty_main(self):
        text = self._e2e("procedure main\n  skip")
        self.assertIn("start:", text)
        self.assertIn("finish:", text)

    def test_increment(self):
        text = self._e2e("int x\nprocedure main\n  x += 1")
        self.assertIn("DATA 0", text)
        self.assertIn("START", text)

    def test_fibonacci_like(self):
        """A slightly more complex program compiles without error."""
        src = """\
int x
int y
procedure main
  x += 1
  from x = 1 do
    x += y
    x <=> y
  loop
    skip
  until y = 10
"""
        text = self._e2e(src)
        self.assertIn("START", text)
        self.assertIn("FINISH", text)

    def test_multiple_procedures(self):
        src = """\
int n
procedure inc
  n += 1
procedure dec
  n -= 1
procedure main
  call inc
  call inc
  uncall inc
"""
        text = self._e2e(src)
        self.assertIn("inc_top:", text)
        self.assertIn("dec_top:", text)
        self.assertIn("main_top:", text)

    def test_nested_if(self):
        src = """\
int x
int y
procedure main
  x += 1
  if x = 1 then
    y += 10
  else
    y += 20
  fi y = 10
"""
        text = self._e2e(src)
        self.assertIn("BEQ", text)
        self.assertIn("BNE", text)


class TestLexerEdgeCases(unittest.TestCase):
    """Edge cases and error handling for the lexer."""

    def test_invalid_char_raises(self):
        with self.assertRaises(LexError):
            tokenize("@")

    def test_lex_error_location(self):
        try:
            tokenize("x\n@")
            self.fail("Expected LexError")
        except LexError as e:
            self.assertEqual(e.line, 2)
            self.assertEqual(e.col, 1)

    def test_only_whitespace(self):
        tokens = tokenize("   \t  ")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].kind, "EOF")

    def test_only_comment(self):
        tokens = tokenize("// this is a comment")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].kind, "EOF")

    def test_multiline_comment_line_tracking(self):
        tokens = tokenize("// comment\nx")
        idents = [t for t in tokens if t.kind == "IDENT"]
        self.assertEqual(idents[0].line, 2)

    def test_zero_literal(self):
        tokens = tokenize("0")
        self.assertEqual(tokens[0].kind, "INT")
        self.assertEqual(tokens[0].value, "0")

    def test_large_integer(self):
        tokens = tokenize("999999999")
        self.assertEqual(tokens[0].kind, "INT")
        self.assertEqual(tokens[0].value, "999999999")

    def test_keyword_prefix_ident(self):
        tokens = tokenize("iffy")
        self.assertEqual(tokens[0].kind, "IDENT")
        self.assertEqual(tokens[0].value, "iffy")

    def test_underscore_ident(self):
        tokens = tokenize("_x x_y_z")
        idents = [t for t in tokens if t.kind == "IDENT"]
        self.assertEqual(idents[0].value, "_x")
        self.assertEqual(idents[1].value, "x_y_z")

    def test_int_keyword_token(self):
        # 'int' is a keyword; kind becomes "INT", value is "int"
        tokens = tokenize("int")
        self.assertEqual(tokens[0].kind, "INT")
        self.assertEqual(tokens[0].value, "int")

    def test_le_one_token(self):
        toks = tokenize("<=")
        self.assertEqual(toks[0].value, "<=")
        self.assertEqual(toks[0].kind, "OP")

    def test_lt_eq_two_tokens(self):
        toks = tokenize("< =")
        self.assertEqual(toks[0].value, "<")
        self.assertEqual(toks[1].value, "=")

    def test_eof_token_present(self):
        tokens = tokenize("x")
        self.assertEqual(tokens[-1].kind, "EOF")

    def test_column_tracking(self):
        tokens = tokenize("  x")
        self.assertEqual(tokens[0].col, 3)

    def test_comment_not_tokenized(self):
        tokens = tokenize("x += 1 // comment\ny -= 2")
        ops = [t.value for t in tokens if t.kind == "OP"]
        self.assertNotIn("//", ops)

    def test_newline_increments_line(self):
        tokens = tokenize("a\nb\nc")
        idents = [t for t in tokens if t.kind == "IDENT"]
        self.assertEqual(idents[0].line, 1)
        self.assertEqual(idents[1].line, 2)
        self.assertEqual(idents[2].line, 3)

    def test_operators_are_op_kind(self):
        tokens = tokenize("&&")
        self.assertEqual(tokens[0].kind, "OP")
        self.assertEqual(tokens[0].value, "&&")

    def test_multiline_string_empty_lines(self):
        tokens = tokenize("\n\n\nx")
        idents = [t for t in tokens if t.kind == "IDENT"]
        self.assertEqual(idents[0].line, 4)


class TestParserEdgeCases(unittest.TestCase):
    """Edge cases for operator precedence, AST structure, and error handling."""

    def _parse(self, src):
        return parse(tokenize(src))

    def test_prec_and_vs_or(self):
        # a || b && c  →  a || (b && c)
        prog = self._parse(
            "int a\nint b\nint c\n"
            "procedure main\n"
            "  if a || b && c then skip else skip fi a || b && c"
        )
        test = prog.procs[0].body.test
        self.assertEqual(test.op, "||")
        self.assertEqual(test.right.op, "&&")

    def test_prec_xor_vs_bitor(self):
        # a | b ^ c  →  a | (b ^ c)
        prog = self._parse(
            "int a\nint b\nint c\n"
            "procedure main\n"
            "  if a | b ^ c then skip else skip fi a"
        )
        test = prog.procs[0].body.test
        self.assertEqual(test.op, "|")
        self.assertEqual(test.right.op, "^")

    def test_prec_compare_vs_add(self):
        # a + b = c  →  (a + b) = c
        prog = self._parse(
            "int a\nint b\nint c\n"
            "procedure main\n"
            "  if a + b = c then skip else skip fi a"
        )
        test = prog.procs[0].body.test
        self.assertEqual(test.op, "=")
        self.assertEqual(test.left.op, "+")

    def test_assoc_left_add(self):
        # x + x + x  →  (x + x) + x
        prog = self._parse(
            "int x\nprocedure main\n  x += x + x + x"
        )
        e = prog.procs[0].body.expr
        self.assertEqual(e.op, "+")
        self.assertIsInstance(e.left, BinOp)
        self.assertEqual(e.left.op, "+")

    def test_mul_higher_than_add(self):
        # 1 + 2 * 3  →  1 + (2 * 3)
        prog = self._parse("int x\nprocedure main\n  x += 1 + 2 * 3")
        e = prog.procs[0].body.expr
        self.assertEqual(e.op, "+")
        self.assertIsInstance(e.right, BinOp)
        self.assertEqual(e.right.op, "*")

    def test_unary_minus_in_expr(self):
        # -x + y is parsed as (0 - x) + y
        prog = self._parse("int x\nint y\nprocedure main\n  x += -x + y")
        e = prog.procs[0].body.expr
        self.assertEqual(e.op, "+")
        self.assertEqual(e.left.op, "-")

    def test_nested_if_in_from(self):
        prog = self._parse(
            "int x\nint y\n"
            "procedure main\n"
            "  from x = 0 do\n"
            "    if y = 0 then\n"
            "      x += 1\n"
            "    else\n"
            "      skip\n"
            "    fi y != 0\n"
            "  loop\n"
            "    skip\n"
            "  until x = 10"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, From)
        self.assertIsInstance(body.do_, If)

    def test_from_nested_in_if(self):
        prog = self._parse(
            "int x\n"
            "procedure main\n"
            "  if x = 0 then\n"
            "    from x = 0 do x += 1 loop skip until x = 5\n"
            "  else\n"
            "    skip\n"
            "  fi x != 0"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, If)
        self.assertIsInstance(body.then_, From)

    def test_array_index_expr(self):
        prog = self._parse(
            "int a[10]\nint i\n"
            "procedure main\n  a[i + i] += 1"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, AssignArr)
        self.assertIsInstance(body.idx, BinOp)
        self.assertEqual(body.idx.op, "+")

    def test_swap_array_elements(self):
        prog = self._parse(
            "int a[5]\nprocedure main\n  a[0] <=> a[1]"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, Swap)
        self.assertIsNotNone(body.lhs_idx)
        self.assertIsNotNone(body.rhs_idx)
        self.assertIsInstance(body.lhs_idx, Const)
        self.assertEqual(body.lhs_idx.value, 0)

    def test_swap_scalar_and_array(self):
        prog = self._parse(
            "int x\nint a[5]\nprocedure main\n  x <=> a[0]"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, Swap)
        self.assertIsNone(body.lhs_idx)   # x has no index
        self.assertIsNotNone(body.rhs_idx)  # a[0] has index

    def test_multiple_procs_order(self):
        prog = self._parse(
            "procedure foo\n  skip\n"
            "procedure bar\n  skip\n"
            "procedure main\n  skip"
        )
        self.assertEqual(prog.procs[0].name, "foo")
        self.assertEqual(prog.procs[1].name, "bar")
        self.assertEqual(prog.procs[2].name, "main")

    def test_missing_then_raises(self):
        with self.assertRaises(ParseError):
            parse(tokenize(
                "int x\nprocedure main\n  if x = 0 skip else skip fi x"
            ))

    def test_missing_fi_raises(self):
        with self.assertRaises(ParseError):
            parse(tokenize(
                "int x\nprocedure main\n  if x = 0 then skip else skip"
            ))

    def test_missing_until_raises(self):
        with self.assertRaises(ParseError):
            parse(tokenize(
                "int x\nprocedure main\n  from x = 0 do skip loop skip"
            ))

    def test_invalid_assign_op_raises(self):
        with self.assertRaises(ParseError):
            parse(tokenize("int x\nprocedure main\n  x *= 5"))

    def test_fi_expression_parsed(self):
        prog = self._parse(
            "int x\n"
            "procedure main\n"
            "  if x < 5 then skip else skip fi x >= 5"
        )
        body = prog.procs[0].body
        self.assertIsInstance(body, If)
        self.assertEqual(body.test.op, "<")
        self.assertEqual(body.fi.op, ">=")

    def test_var_decl_array_size(self):
        prog = self._parse("int a[7]\nprocedure main\n  skip")
        self.assertEqual(prog.vars[0].size, 7)
        self.assertEqual(len(prog.vars[0].init), 7)
        self.assertTrue(all(v == 0 for v in prog.vars[0].init))

    def test_main_proc_identified(self):
        prog = self._parse(
            "procedure helper\n  skip\n"
            "procedure main\n  skip"
        )
        self.assertEqual(prog.main_proc, "main")

    def test_no_main_uses_first_proc(self):
        prog = self._parse("procedure foo\n  skip")
        self.assertEqual(prog.main_proc, "foo")


class TestPISAAllInstructions(unittest.TestCase):
    """Test format_instr for every PISA instruction type."""

    def test_format_sub(self):
        self.assertEqual(format_instr(SUB("r3", "r4")), "SUB r3 r4")

    def test_format_neg(self):
        self.assertEqual(format_instr(NEG("r3")), "NEG r3")

    def test_format_xor(self):
        self.assertEqual(format_instr(XOR("r3", "r4")), "XOR r3 r4")

    def test_format_subi(self):
        self.assertEqual(format_instr(SUBI("r3", 5)), "SUBI r3 5")

    def test_format_xori(self):
        self.assertEqual(format_instr(XORI("r3", 1)), "XORI r3 1")

    def test_format_orx(self):
        self.assertEqual(format_instr(ORX("r3", "r4")), "ORX r3 r4")

    def test_format_andx(self):
        self.assertEqual(format_instr(ANDX("r3", "r4", "r5")), "ANDX r3 r4 r5")

    def test_format_sltx(self):
        self.assertEqual(format_instr(SLTX("r3", "r4", "r5")), "SLTX r3 r4 r5")

    def test_format_exch(self):
        self.assertEqual(format_instr(EXCH("r3", "r4")), "EXCH r3 r4")

    def test_format_rbra(self):
        self.assertEqual(format_instr(RBRA("label")), "RBRA label")

    def test_format_bne(self):
        self.assertEqual(format_instr(BNE("r3", "r0", "L1")), "BNE r3 r0 L1")

    def test_format_bgez(self):
        self.assertEqual(format_instr(BGEZ("r3", "L2")), "BGEZ r3 L2")

    def test_format_swapbr(self):
        self.assertEqual(format_instr(SWAPBR("r2")), "SWAPBR r2")

    def test_format_finish(self):
        self.assertEqual(format_instr(FINISH()), "FINISH")

    def test_format_data_nonzero(self):
        self.assertEqual(format_instr(DATA(42)), "DATA 42")

    def test_format_data_negative(self):
        self.assertEqual(format_instr(DATA(-1)), "DATA -1")

    def test_empty_program(self):
        self.assertEqual(print_program([]), "")

    def test_labeled_instr_format(self):
        prog = [LabeledInstr("foo", ADD("r3", "r4"))]
        self.assertIn("foo: ADD r3 r4", print_program(prog))

    def test_unlabeled_instr_indented(self):
        prog = [LabeledInstr(None, SUB("r3", "r4"))]
        text = print_program(prog)
        self.assertIn("SUB r3 r4", text)
        # No colon in the line (unlabeled)
        lines = text.split("\n")
        self.assertNotIn(":", lines[0])

    def test_multiple_instrs_newlines(self):
        prog = [
            LabeledInstr("L1", ADD("r3", "r4")),
            LabeledInstr(None, SUB("r5", "r6")),
            LabeledInstr("L2", NEG("r7")),
        ]
        text = print_program(prog)
        lines = text.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertIn("L1:", lines[0])
        self.assertIn("L2:", lines[2])

    def test_print_program_line_count(self):
        prog = [
            LabeledInstr("a", ADD("r3", "r4")),
            LabeledInstr("b", SUB("r3", "r4")),
        ]
        text = print_program(prog)
        self.assertEqual(text.count("\n"), 1)


class TestCodeGenExpressions(unittest.TestCase):
    """Test gen_expr for all expression types and operators."""

    def _cg(self):
        """CodeGen with x=offset 0, y=offset 1, a[3]=offset 2."""
        cg = CodeGen()
        cg._var_offsets['x'] = 0
        cg._var_offsets['y'] = 1
        cg._var_offsets['a'] = 2
        cg._var_sizes['x'] = 1
        cg._var_sizes['y'] = 1
        cg._var_sizes['a'] = 3
        return cg

    def _itypes(self, code):
        return [type(li.instr).__name__ for li in code]

    def _instrs(self, code):
        return [li.instr for li in code]

    def test_const_zero_no_addi_subi(self):
        cg = self._cg()
        code, _ = cg.gen_expr(Const(0))
        types = self._itypes(code)
        self.assertNotIn("ADDI", types)
        self.assertNotIn("SUBI", types)

    def test_const_pos_uses_addi(self):
        cg = self._cg()
        code, _ = cg.gen_expr(Const(5))
        instrs = self._instrs(code)
        addi = [i for i in instrs if isinstance(i, ADDI)]
        self.assertTrue(any(i.c == 5 for i in addi))

    def test_const_neg_uses_subi(self):
        cg = self._cg()
        code, _ = cg.gen_expr(Const(-3))
        instrs = self._instrs(code)
        subi = [i for i in instrs if isinstance(i, SUBI)]
        self.assertTrue(any(i.c == 3 for i in subi))

    def test_const_result_reg_is_register(self):
        cg = self._cg()
        _, reg = cg.gen_expr(Const(7))
        self.assertIsInstance(reg, str)
        self.assertTrue(reg.startswith("r"))
        self.assertNotIn(reg, {"r0", "r1", "r2"})

    def test_var_uses_exch(self):
        cg = self._cg()
        code, _ = cg.gen_expr(Var('x'))
        self.assertIn("EXCH", self._itypes(code))

    def test_var_x_loads_offset_0(self):
        cg = self._cg()
        code, _ = cg.gen_expr(Var('x'))
        instrs = self._instrs(code)
        addi = [i for i in instrs if isinstance(i, ADDI)]
        self.assertTrue(any(i.c == 0 for i in addi))

    def test_var_y_loads_offset_1(self):
        cg = self._cg()
        code, _ = cg.gen_expr(Var('y'))
        instrs = self._instrs(code)
        addi = [i for i in instrs if isinstance(i, ADDI)]
        self.assertTrue(any(i.c == 1 for i in addi))

    def test_var_returns_nonempty_code(self):
        cg = self._cg()
        code, _ = cg.gen_expr(Var('x'))
        self.assertGreater(len(code), 0)

    def test_array_access_uses_add_and_exch(self):
        cg = self._cg()
        code, _ = cg.gen_expr(ArrayAccess('a', Const(1)))
        types = self._itypes(code)
        self.assertIn("ADD", types)
        self.assertIn("EXCH", types)

    def test_binop_add_uses_add(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('+', Var('x'), Var('y')))
        self.assertIn("ADD", self._itypes(code))

    def test_binop_sub_uses_sub(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('-', Var('x'), Var('y')))
        self.assertIn("SUB", self._itypes(code))

    def test_binop_xor_uses_xor(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('^', Var('x'), Var('y')))
        self.assertIn("XOR", self._itypes(code))

    def test_binop_eq_uses_sltx_and_xori(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('=', Var('x'), Var('y')))
        types = self._itypes(code)
        self.assertIn("SLTX", types)
        self.assertIn("XORI", types)

    def test_binop_neq_uses_sltx_and_orx(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('!=', Var('x'), Var('y')))
        types = self._itypes(code)
        self.assertIn("SLTX", types)
        self.assertIn("ORX", types)

    def test_binop_lt_uses_sltx(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('<', Var('x'), Var('y')))
        self.assertIn("SLTX", self._itypes(code))

    def test_binop_gt_uses_sltx(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('>', Var('x'), Var('y')))
        self.assertIn("SLTX", self._itypes(code))

    def test_binop_le_uses_sltx_and_xori(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('<=', Var('x'), Var('y')))
        types = self._itypes(code)
        self.assertIn("SLTX", types)
        self.assertIn("XORI", types)

    def test_binop_ge_uses_sltx_and_xori(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('>=', Var('x'), Var('y')))
        types = self._itypes(code)
        self.assertIn("SLTX", types)
        self.assertIn("XORI", types)

    def test_binop_logical_and_uses_andx(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('&&', Var('x'), Var('y')))
        self.assertIn("ANDX", self._itypes(code))

    def test_binop_logical_or_uses_orx(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('||', Var('x'), Var('y')))
        self.assertIn("ORX", self._itypes(code))

    def test_binop_bitand_uses_andx(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('&', Var('x'), Var('y')))
        self.assertIn("ANDX", self._itypes(code))

    def test_binop_bitor_uses_orx(self):
        cg = self._cg()
        code, _ = cg.gen_expr(BinOp('|', Var('x'), Var('y')))
        self.assertIn("ORX", self._itypes(code))

    def test_mul_raises_codegen_error(self):
        cg = self._cg()
        with self.assertRaises(CodeGenError):
            cg.gen_expr(BinOp('*', Var('x'), Var('y')))

    def test_code_is_list_of_labeled_instr(self):
        cg = self._cg()
        code, _ = cg.gen_expr(Var('x'))
        self.assertIsInstance(code, list)
        self.assertTrue(all(isinstance(li, LabeledInstr) for li in code))


class TestCodeGenStatements(unittest.TestCase):
    """Test gen_stmt for all statement types."""

    def _cg(self, var_names):
        cg = CodeGen()
        for i, name in enumerate(var_names):
            cg._var_offsets[name] = i
            cg._var_sizes[name] = 1
        return cg

    def _cg_arr(self, name, size, offset=0):
        cg = CodeGen()
        cg._var_offsets[name] = offset
        cg._var_sizes[name] = size
        return cg

    def _itypes(self, code):
        return [type(li.instr).__name__ for li in code]

    def _instrs(self, code):
        return [li.instr for li in code]

    def test_skip_empty(self):
        cg = self._cg([])
        code = cg.gen_stmt(Skip())
        self.assertEqual(code, [])

    def test_assign_var_add(self):
        cg = self._cg(['x', 'y'])
        code = cg.gen_stmt(AssignVar('x', '+=', Var('y')))
        self.assertIn("ADD", self._itypes(code))

    def test_assign_var_sub(self):
        cg = self._cg(['x', 'y'])
        code = cg.gen_stmt(AssignVar('x', '-=', Var('y')))
        self.assertIn("SUB", self._itypes(code))

    def test_assign_var_xor(self):
        cg = self._cg(['x', 'y'])
        code = cg.gen_stmt(AssignVar('x', '^=', Var('y')))
        self.assertIn("XOR", self._itypes(code))

    def test_assign_const_uses_exch(self):
        cg = self._cg(['x'])
        code = cg.gen_stmt(AssignVar('x', '+=', Const(5)))
        self.assertIn("EXCH", self._itypes(code))

    def test_assign_arr_add(self):
        cg = self._cg_arr('a', 5)
        code = cg.gen_stmt(AssignArr('a', Const(0), '+=', Const(1)))
        types = self._itypes(code)
        self.assertIn("ADD", types)
        self.assertIn("EXCH", types)

    def test_assign_arr_sub(self):
        cg = self._cg_arr('a', 5)
        code = cg.gen_stmt(AssignArr('a', Const(2), '-=', Const(1)))
        self.assertIn("SUB", self._itypes(code))

    def test_assign_arr_xor(self):
        cg = self._cg_arr('a', 5)
        code = cg.gen_stmt(AssignArr('a', Const(1), '^=', Const(3)))
        self.assertIn("XOR", self._itypes(code))

    def test_call_generates_bra(self):
        cg = self._cg([])
        code = cg.gen_stmt(Call('foo'))
        instrs = self._instrs(code)
        self.assertEqual(len(instrs), 1)
        self.assertIsInstance(instrs[0], BRA)
        self.assertEqual(instrs[0].label, 'foo')

    def test_uncall_generates_rbra(self):
        cg = self._cg([])
        code = cg.gen_stmt(Uncall('foo'))
        instrs = self._instrs(code)
        self.assertEqual(len(instrs), 1)
        self.assertIsInstance(instrs[0], RBRA)
        self.assertEqual(instrs[0].label, 'foo')

    def test_swap_scalars_uses_exch(self):
        cg = self._cg(['x', 'y'])
        code = cg.gen_stmt(Swap('x', None, 'y', None))
        exch_count = self._itypes(code).count("EXCH")
        self.assertGreaterEqual(exch_count, 2)

    def test_seq_combines_stmts(self):
        cg = self._cg(['x'])
        code = cg.gen_stmt(Seq([
            AssignVar('x', '+=', Const(1)),
            AssignVar('x', '-=', Const(1)),
        ]))
        types = self._itypes(code)
        self.assertIn("ADD", types)
        self.assertIn("SUB", types)

    def test_if_generates_beq_and_bne(self):
        cg = self._cg(['x'])
        stmt = If(
            BinOp('=', Var('x'), Const(0)),
            AssignVar('x', '+=', Const(1)),
            Skip(),
            BinOp('!=', Var('x'), Const(0)),
        )
        code = cg.gen_stmt(stmt)
        types = self._itypes(code)
        self.assertIn("BEQ", types)
        self.assertIn("BNE", types)

    def test_from_generates_bra_and_beq(self):
        cg = self._cg(['x'])
        stmt = From(
            BinOp('=', Var('x'), Const(0)),
            AssignVar('x', '+=', Const(1)),
            Skip(),
            BinOp('=', Var('x'), Const(5)),
        )
        code = cg.gen_stmt(stmt)
        types = self._itypes(code)
        self.assertIn("BRA", types)
        self.assertIn("BEQ", types)

    def test_proc_entry_has_subi_r1(self):
        """gen_proc: second instruction is SUBI r1 1."""
        cg = CodeGen()
        code = cg.gen_proc(ProcDecl('foo', Skip()))
        instrs = self._instrs(code)
        self.assertIsInstance(instrs[1], SUBI)
        self.assertEqual(instrs[1].rd, 'r1')
        self.assertEqual(instrs[1].c, 1)

    def test_proc_has_swapbr_r2(self):
        cg = CodeGen()
        code = cg.gen_proc(ProcDecl('foo', Skip()))
        instrs = self._instrs(code)
        swapbr = [i for i in instrs if isinstance(i, SWAPBR)]
        self.assertGreaterEqual(len(swapbr), 1)
        self.assertEqual(swapbr[0].rd, 'r2')

    def test_proc_has_neg_r2(self):
        cg = CodeGen()
        code = cg.gen_proc(ProcDecl('foo', Skip()))
        instrs = self._instrs(code)
        negs = [i for i in instrs if isinstance(i, NEG)]
        self.assertGreaterEqual(len(negs), 1)
        self.assertEqual(negs[0].rd, 'r2')

    def test_proc_top_bot_labels(self):
        cg = CodeGen()
        code = cg.gen_proc(ProcDecl('foo', Skip()))
        labels = [li.label for li in code if li.label]
        self.assertIn('foo_top', labels)
        self.assertIn('foo_bot', labels)
        self.assertIn('foo', labels)

    def test_proc_has_addi_r1_1(self):
        """Procedure entry sequence contains ADDI r1 1."""
        cg = CodeGen()
        code = cg.gen_proc(ProcDecl('foo', Skip()))
        instrs = self._instrs(code)
        addi_r1 = [i for i in instrs if isinstance(i, ADDI) and i.rd == 'r1']
        self.assertTrue(any(i.c == 1 for i in addi_r1))


class TestCodeGenStructural(unittest.TestCase):
    """Structural tests of generated PISA programs."""

    @staticmethod
    def _compile(src):
        return compile_program(parse(tokenize(src)))

    @staticmethod
    def _instrs(code):
        return [li.instr for li in code]

    @staticmethod
    def _labels(code):
        return [li.label for li in code if li.label]

    @staticmethod
    def _itypes(code):
        return [type(li.instr).__name__ for li in code]

    @staticmethod
    def _label_map(code):
        return {li.label: i for i, li in enumerate(code) if li.label}

    def test_program_with_vars_starts_with_data(self):
        code = self._compile("int x\nprocedure main\n  skip")
        self.assertIsInstance(code[0].instr, DATA)

    def test_program_no_vars_no_data(self):
        code = self._compile("procedure main\n  skip")
        data = [i for i in self._instrs(code) if isinstance(i, DATA)]
        self.assertEqual(len(data), 0)

    def test_exactly_one_start(self):
        code = self._compile("procedure main\n  skip")
        count = sum(1 for i in self._instrs(code) if isinstance(i, START))
        self.assertEqual(count, 1)

    def test_exactly_one_finish(self):
        code = self._compile("procedure main\n  skip")
        count = sum(1 for i in self._instrs(code) if isinstance(i, FINISH))
        self.assertEqual(count, 1)

    def test_start_label_present(self):
        code = self._compile("procedure main\n  skip")
        self.assertIn("start", self._labels(code))

    def test_finish_label_present(self):
        code = self._compile("procedure main\n  skip")
        self.assertIn("finish", self._labels(code))

    def test_bra_main_present(self):
        code = self._compile("procedure main\n  skip")
        bra_main = [i for i in self._instrs(code)
                    if isinstance(i, BRA) and i.label == "main"]
        self.assertGreaterEqual(len(bra_main), 1)

    def test_proc_top_bot_labels_present(self):
        code = self._compile("procedure main\n  skip")
        labels = self._labels(code)
        self.assertIn("main_top", labels)
        self.assertIn("main_bot", labels)

    def test_data_count_matches_var_sizes(self):
        code = self._compile(
            "int x\nint y\nint a[3]\nprocedure main\n  skip"
        )
        data_count = sum(1 for i in self._instrs(code) if isinstance(i, DATA))
        self.assertEqual(data_count, 5)  # x=1, y=1, a=3

    def test_stack_offset_positive(self):
        code = self._compile("procedure main\n  skip")
        addi_r1 = [i for i in self._instrs(code)
                   if isinstance(i, ADDI) and i.rd == 'r1']
        self.assertTrue(any(i.c > 0 for i in addi_r1))

    def test_if_has_beq_and_bne(self):
        code = self._compile(
            "int x\nprocedure main\n"
            "  if x = 0 then skip else skip fi x != 0"
        )
        types = self._itypes(code)
        self.assertIn("BEQ", types)
        self.assertIn("BNE", types)

    def test_from_has_multiple_bra(self):
        code = self._compile(
            "int x\nprocedure main\n"
            "  from x = 0 do x += 1 loop skip until x = 5"
        )
        bra_count = sum(1 for i in self._instrs(code) if isinstance(i, BRA))
        self.assertGreater(bra_count, 1)

    def test_no_branch_to_undefined_labels(self):
        """All BRA/RBRA/BEQ/BNE targets must exist as labels."""
        code = self._compile(
            "int x\nprocedure main\n"
            "  if x = 0 then skip else skip fi x != 0"
        )
        label_map = self._label_map(code)
        for li in code:
            i = li.instr
            if isinstance(i, (BRA, RBRA)):
                self.assertIn(i.label, label_map,
                              f"Undefined label: {i.label}")
            elif isinstance(i, (BEQ, BNE)):
                self.assertIn(i.label, label_map,
                              f"Undefined label: {i.label}")

    def test_call_uncall_bra_rbra(self):
        code = self._compile(
            "procedure foo\n  skip\n"
            "procedure main\n  call foo\n  uncall foo"
        )
        instrs = self._instrs(code)
        bra_foo = [i for i in instrs if isinstance(i, BRA) and i.label == "foo"]
        rbra_foo = [i for i in instrs if isinstance(i, RBRA) and i.label == "foo"]
        self.assertGreaterEqual(len(bra_foo), 1)
        self.assertGreaterEqual(len(rbra_foo), 1)

    def test_single_non_main_proc_used_as_entry(self):
        code = self._compile("procedure foo\n  skip")
        instrs = self._instrs(code)
        bra_foo = [i for i in instrs if isinstance(i, BRA) and i.label == "foo"]
        self.assertGreaterEqual(len(bra_foo), 1)


class TestCodeGenErrors(unittest.TestCase):
    """Error conditions in code generation."""

    def _compile(self, src):
        return compile_program(parse(tokenize(src)))

    def test_undefined_variable_raises(self):
        with self.assertRaises(CodeGenError):
            self._compile("procedure main\n  x += 1")

    def test_multiplication_raises(self):
        with self.assertRaises(CodeGenError):
            self._compile("int x\nint y\nprocedure main\n  x += x * y")

    def test_division_raises(self):
        with self.assertRaises(CodeGenError):
            self._compile("int x\nint y\nprocedure main\n  x += x / y")

    def test_modulo_raises(self):
        with self.assertRaises(CodeGenError):
            self._compile("int x\nint y\nprocedure main\n  x += x % y")

    def test_empty_program_no_error(self):
        code = compile_program(parse(tokenize("")))
        self.assertIsInstance(code, list)

    def test_single_proc_not_named_main(self):
        code = self._compile("procedure foo\n  skip")
        text = print_program(code)
        self.assertIn("BRA foo", text)


class TestEndToEndIntegration(unittest.TestCase):
    """Integration tests for the full source → PISA pipeline."""

    def _e2e(self, src):
        return compile_program(parse(tokenize(src)))

    def _text(self, src):
        return print_program(self._e2e(src))

    def test_array_sum_compiles(self):
        src = """\
int a[5]
int s
procedure main
  a[0] += 1
  a[1] += 2
  s += a[0]
  s += a[1]
"""
        text = self._text(src)
        self.assertIn("START", text)
        self.assertIn("FINISH", text)

    def test_mutual_call_chain(self):
        src = """\
procedure foo
  skip
procedure bar
  call foo
procedure main
  call bar
  uncall bar
"""
        text = self._text(src)
        self.assertIn("BRA bar", text)
        self.assertIn("BRA foo", text)

    def test_array_swap_compiles(self):
        src = """\
int a[5]
procedure main
  a[0] += 1
  a[1] += 2
  a[0] <=> a[1]
"""
        code = self._e2e(src)
        text = print_program(code)
        self.assertIn("START", text)
        self.assertIn("FINISH", text)

    def test_nested_if_from_compiles(self):
        src = """\
int x
int y
procedure main
  from x = 0 do
    if y = 0 then
      x += 1
    else
      skip
    fi y = 0
  loop
    skip
  until x = 5
"""
        text = self._text(src)
        self.assertIn("BEQ", text)
        self.assertIn("BRA", text)

    def test_all_assign_ops_compiles(self):
        src = """\
int x
int y
procedure main
  x += y
  x -= y
  x ^= y
"""
        text = self._text(src)
        self.assertIn("ADD", text)
        self.assertIn("SUB", text)
        self.assertIn("XOR", text)

    def test_compare_eq_neq_compiles(self):
        src = """\
int x
procedure main
  if x = 0 then skip else skip fi x != 0
"""
        text = self._text(src)
        self.assertIn("SLTX", text)

    def test_compare_lt_ge_compiles(self):
        src = """\
int x
int y
procedure main
  if x < y then skip else skip fi x >= y
"""
        text = self._text(src)
        self.assertIn("SLTX", text)

    def test_bitwise_ops_compiles(self):
        src = """\
int x
int y
procedure main
  if x & y then skip else skip fi x | y
"""
        text = self._text(src)
        self.assertIn("ANDX", text)
        self.assertIn("ORX", text)

    def test_large_program_compiles(self):
        vars_decl = "\n".join(f"int v{i}" for i in range(8))
        procs = "\n".join(f"procedure p{i}\n  skip" for i in range(4))
        src = f"{vars_decl}\n{procs}\nprocedure main\n  call p0\n  call p1"
        text = self._text(src)
        self.assertIn("START", text)

    def test_print_stmt_compiles(self):
        src = "int x\nprocedure main\n  x += 1\n  print(x)"
        text = self._text(src)
        self.assertIn("START", text)

    def test_var_init_values_in_data(self):
        src = "int x = 10\nint y = 20\nprocedure main\n  skip"
        code = self._e2e(src)
        data_vals = [i.value for i in [li.instr for li in code]
                     if isinstance(i, DATA)]
        self.assertIn(10, data_vals)
        self.assertIn(20, data_vals)

    def test_uncall_is_rbra(self):
        src = """\
procedure foo
  skip
procedure main
  call foo
  uncall foo
"""
        code = self._e2e(src)
        instrs = [li.instr for li in code]
        rbras = [i for i in instrs if isinstance(i, RBRA) and i.label == "foo"]
        self.assertGreaterEqual(len(rbras), 1)

    def test_swap_scalar_produces_exch(self):
        src = "int x\nint y\nprocedure main\n  x += 1\n  y += 2\n  x <=> y"
        code = self._e2e(src)
        exchs = [li.instr for li in code if isinstance(li.instr, EXCH)]
        self.assertGreaterEqual(len(exchs), 2)


if __name__ == "__main__":
    unittest.main()
