"""Janus parser (recursive descent with Pratt-style expression parsing)."""

from typing import List, Optional
from syntax import (
    Expr, Const, Var, ArrayAccess, BinOp,
    Stmt, Skip, AssignVar, AssignArr, Swap, Call, Uncall,
    If, From, Seq, Print,
    VarDecl, ProcDecl, Program,
)
from lexer import Token, LexError


class ParseError(Exception):
    def __init__(self, msg, token=None):
        if token:
            super().__init__(f"Parse error at line {token.line}, col {token.col}: {msg}")
        else:
            super().__init__(f"Parse error: {msg}")
        self.token = token


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str, value: str = None) -> Token:
        tok = self.peek()
        if tok.kind != kind:
            raise ParseError(f"Expected {kind} but got {tok.kind} ({tok.value!r})", tok)
        if value is not None and tok.value != value:
            raise ParseError(f"Expected {value!r} but got {tok.value!r}", tok)
        return self.advance()

    def match(self, kind: str, value: str = None) -> Optional[Token]:
        tok = self.peek()
        if tok.kind == kind and (value is None or tok.value == value):
            return self.advance()
        return None

    # --- Expression parsing (Pratt-style) ---

    def _prefix_bp(self, op: str):
        if op == '-':
            return 90  # unary minus
        return None

    def _infix_bp(self, op: str):
        """Return (left_bp, right_bp) or None."""
        prec = {
            '||': (10, 11),
            '&&': (20, 21),
            '|':  (25, 26),
            '^':  (27, 28),
            '&':  (29, 30),
            '=':  (40, 41), '!=': (40, 41),
            '<':  (50, 51), '>':  (50, 51),
            '<=': (50, 51), '>=': (50, 51),
            '+':  (60, 61), '-':  (60, 61),
            '*':  (70, 71), '/':  (70, 71), '%': (70, 71),
        }
        return prec.get(op)

    def parse_expr(self, min_bp: int = 0) -> Expr:
        tok = self.peek()

        # Prefix: unary minus
        if tok.kind == "OP" and tok.value == '-':
            self.advance()
            rbp = self._prefix_bp('-')
            operand = self.parse_expr(rbp)
            lhs = BinOp('-', Const(0), operand)
        elif tok.kind == "INT":
            self.advance()
            lhs = Const(int(tok.value))
        elif tok.kind == "IDENT":
            self.advance()
            name = tok.value
            if self.match("OP", "["):
                idx = self.parse_expr()
                self.expect("OP", "]")
                lhs = ArrayAccess(name, idx)
            else:
                lhs = Var(name)
        elif tok.kind == "OP" and tok.value == "(":
            self.advance()
            lhs = self.parse_expr()
            self.expect("OP", ")")
        else:
            raise ParseError(f"Expected expression, got {tok.value!r}", tok)

        # Infix
        while True:
            tok = self.peek()
            if tok.kind != "OP":
                break
            bp = self._infix_bp(tok.value)
            if bp is None:
                break
            l_bp, r_bp = bp
            if l_bp < min_bp:
                break
            op = self.advance().value
            rhs = self.parse_expr(r_bp)
            lhs = BinOp(op, lhs, rhs)

        return lhs

    # --- Statement parsing ---

    def parse_stmt(self) -> Stmt:
        """Parse a single statement."""
        tok = self.peek()

        if tok.kind == "SKIP":
            self.advance()
            return Skip()

        if tok.kind == "CALL":
            self.advance()
            name = self.expect("IDENT").value
            return Call(name)

        if tok.kind == "UNCALL":
            self.advance()
            name = self.expect("IDENT").value
            return Uncall(name)

        if tok.kind == "PRINT":
            self.advance()
            self.expect("OP", "(")
            name = self.expect("IDENT").value
            self.expect("OP", ")")
            return Print(name)

        if tok.kind == "IF":
            return self.parse_if()

        if tok.kind == "FROM":
            return self.parse_from()

        if tok.kind == "IDENT":
            return self.parse_assignment_or_swap()

        raise ParseError(f"Expected statement, got {tok.kind} ({tok.value!r})", tok)

    def parse_assignment_or_swap(self) -> Stmt:
        name = self.expect("IDENT").value

        # Array access?
        idx = None
        if self.match("OP", "["):
            idx = self.parse_expr()
            self.expect("OP", "]")

        tok = self.peek()

        # Swap: x <=> y  or  x[i] <=> y[j]
        if tok.kind == "OP" and tok.value == "<=>":
            self.advance()
            rhs_name = self.expect("IDENT").value
            rhs_idx = None
            if self.match("OP", "["):
                rhs_idx = self.parse_expr()
                self.expect("OP", "]")
            return Swap(name, idx, rhs_name, rhs_idx)

        # Assignment: x ⊕= e  or  x[i] ⊕= e
        if tok.kind == "OP" and tok.value in ("+=", "-=", "^="):
            op = self.advance().value
            expr = self.parse_expr()
            if idx is not None:
                return AssignArr(name, idx, op, expr)
            else:
                return AssignVar(name, op, expr)

        raise ParseError(f"Expected assignment operator or <=>, got {tok.value!r}", tok)

    def parse_if(self) -> Stmt:
        self.expect("IF")
        test = self.parse_expr()
        self.expect("THEN")
        then_ = self.parse_stmt_list("ELSE")
        self.expect("ELSE")
        else_ = self.parse_stmt_list("FI")
        self.expect("FI")
        fi = self.parse_expr()
        return If(test, then_, else_, fi)

    def parse_from(self) -> Stmt:
        self.expect("FROM")
        from_ = self.parse_expr()
        self.expect("DO")
        do_ = self.parse_stmt_list("LOOP")
        self.expect("LOOP")
        loop_ = self.parse_stmt_list("UNTIL")
        self.expect("UNTIL")
        until = self.parse_expr()
        return From(from_, do_, loop_, until)

    def parse_stmt_list(self, *terminators: str) -> Stmt:
        """Parse statements until one of the terminator keywords is seen."""
        stmts = []
        while self.peek().kind not in terminators:
            stmts.append(self.parse_stmt())
        if len(stmts) == 0:
            return Skip()
        if len(stmts) == 1:
            return stmts[0]
        return Seq(stmts)

    # --- Declarations ---

    def parse_var_decl(self) -> VarDecl:
        """Parse: int x  or  int x[10]  or  int x = 5"""
        self.expect("INT")
        name = self.expect("IDENT").value
        if self.match("OP", "["):
            size_tok = self.expect("INT")
            size = int(size_tok.value)
            self.expect("OP", "]")
            return VarDecl(name, size, [0] * size)
        elif self.match("OP", "="):
            val = int(self.expect("INT").value)
            return VarDecl(name, 1, [val])
        else:
            return VarDecl(name, 1, [0])

    def parse_proc_decl(self) -> ProcDecl:
        self.expect("PROCEDURE")
        name = self.expect("IDENT").value
        body = self.parse_stmt_list("PROCEDURE", "EOF")
        return ProcDecl(name, body)

    def parse_program(self) -> Program:
        vars_ = []
        procs = []

        # Parse variable declarations (at top)
        while self.peek().kind == "INT":
            vars_.append(self.parse_var_decl())

        # Parse procedure declarations
        while self.peek().kind == "PROCEDURE":
            procs.append(self.parse_proc_decl())

        self.expect("EOF")

        # Find main procedure
        main_name = "main"
        for p in procs:
            if p.name == "main":
                break
        else:
            if procs:
                main_name = procs[0].name

        return Program(vars_, procs, main_name)


def parse(tokens: List[Token]) -> Program:
    """Parse a token list into a Program AST."""
    parser = Parser(tokens)
    return parser.parse_program()
