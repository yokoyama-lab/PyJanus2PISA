"""Code generator: Janus AST → PISA instructions.

Implements the translation from Section 4 of:
  "Clean Translation of an Imperative Reversible Programming Language"
  (Axelsen, CC 2011)
"""

from typing import List, Dict, Tuple, Optional
from syntax import (
    Expr, Const, Var, ArrayAccess, BinOp,
    Stmt, Skip, AssignVar, AssignArr, Swap, Call, Uncall,
    If, From, Seq, Print,
    VarDecl, ProcDecl, Program,
)
from pisa import (
    Instr, LabeledInstr,
    ADD, SUB, NEG, XOR, ADDI, SUBI, XORI,
    ORX, ANDX, SLTX,
    EXCH, BRA, RBRA, BEQ, BNE, BGEZ, SWAPBR,
    DATA, START, FINISH,
    is_wf, format_instr,
)
from regalloc import RegAlloc
from inverse import invert_stmt


_COMPARISONS = ('=', '!=', '<', '>', '<=', '>=')
_LOGICAL = ('&&', '||')

# `+ - ^`: result combined into the left operand's register, rl op= rr
# (CodeGen._gen_inplace).
_INPLACE = {'+': ADD, '-': SUB, '^': XOR}

# The other binary operators: re ^= f(rl, rr) into a fresh zero register re,
# with re, rl, rr pairwise distinct (CodeGen._gen_combine).  Each block is
# self-inverse and only reads rl, rr.  For `!=`, (rl < rr) and (rr < rl) are
# never both 1, so XOR-ing the two bits is their OR.  `&&` / `||` receive 0/1
# operands (_logical_operands), on which bitwise ANDX / ORX are logical.
_COMBINE = {
    '<':  lambda re, a, b: [SLTX(re, a, b)],
    '>':  lambda re, a, b: [SLTX(re, b, a)],
    '<=': lambda re, a, b: [SLTX(re, b, a), XORI(re, 1)],
    '>=': lambda re, a, b: [SLTX(re, a, b), XORI(re, 1)],
    '!=': lambda re, a, b: [SLTX(re, a, b), SLTX(re, b, a)],
    '=':  lambda re, a, b: [SLTX(re, a, b), SLTX(re, b, a), XORI(re, 1)],
    '&&': lambda re, a, b: [ANDX(re, a, b)],
    '||': lambda re, a, b: [ORX(re, a, b)],
    '&':  lambda re, a, b: [ANDX(re, a, b)],
    '|':  lambda re, a, b: [ORX(re, a, b)],
}


def _expr_vars(e: Expr) -> set:
    """Names of the variables and arrays read by e (indices included)."""
    if isinstance(e, Var):
        return {e.name}
    if isinstance(e, ArrayAccess):
        return {e.name} | _expr_vars(e.index)
    if isinstance(e, BinOp):
        return _expr_vars(e.left) | _expr_vars(e.right)
    return set()

# Jump target for a violated `fi` or loop assertion (the FINISH of gen_program).
ASSERT_FAIL_LABEL = "finish"


def _as_flag(e: Expr) -> Expr:
    """Return an expression equal to 1 if e is true (nonzero) and 0 otherwise.

    _gen_if and _gen_from XOR their predicates into a 0/1 path flag, which is
    only sound for 0/1-valued expressions.  Comparisons, `&&`/`||` and the
    constants 0/1 already are; anything else (e.g. `if 5 then ... fi 7`, valid Janus)
    is normalised by `e != 0`.
    """
    if isinstance(e, BinOp) and (e.op in _COMPARISONS or e.op in _LOGICAL):
        return e
    if isinstance(e, Const) and e.value in (0, 1):
        return e
    return BinOp('!=', e, Const(0))


def _logical_operands(e: BinOp) -> BinOp:
    """Normalise the operands of `&&` / `||` to 0/1 (see _as_flag).

    Janus `&&`/`||` are logical: operands are true when nonzero and the
    result is 0/1.  They are lowered to the bitwise ANDX / ORX, which agree
    only on 0/1 operands (`1 && 2` used to give 1 & 2 = 0, `1 || 2` gave 3).
    Operands that are already 0/1 (comparisons, logical ops, 0/1 constants),
    the usual case, are kept.  The others become `operand != 0`, which
    _gen_nonzero evaluates and uncomputes on the spot, so the extra flag
    costs code but not registers.
    """
    if e.op in _LOGICAL:
        return BinOp(e.op, _as_flag(e.left), _as_flag(e.right))
    return e


def _is_nonzero_test(e: BinOp) -> bool:
    """`e != 0`, the shape _as_flag produces."""
    return e.op == '!=' and isinstance(e.right, Const) and e.right.value == 0


def _proc_label(name: str) -> str:
    """Entry label of procedure `name`.

    Procedure labels share one namespace with the labels the compiler makes
    up (`f_top`/`f_bot`, the companion `f_inv`, `if_false_1`, ...), and Janus
    identifiers may contain `_`, so e.g. a user procedure `f_inv` used to be
    the same label as the inverted companion of `f`.  Doubling every `_` of
    the source name makes the map injective and disjoint from generated
    names: a user-derived label has only even runs of `_`, while every
    generated suffix (`_top`, `_bot`, `_inv`, `if_false_<n>`, ...) adds a
    single `_`.  Names without `_` (the common case) are unchanged.
    """
    return name.replace("_", "__")


def _inv_proc_name(name: str) -> str:
    """Label of the inverted companion of procedure `name`."""
    return _proc_label(name) + "_inv"



class CodeGenError(Exception):
    pass


class CodeGen:
    def __init__(self):
        self.reg = RegAlloc()
        self._label_counter = 0
        self._var_offsets: Dict[str, int] = {}  # variable name → DATA offset
        self._var_sizes: Dict[str, int] = {}    # variable name → size
        self._total_vars = 0
        self._proc_names: List[str] = []
        # Procedures to be inlined.  Value is (ProcDecl, can_uncall).
        # can_uncall=True means the body is branch-free and safe to reverse.
        self._inline_procs: Dict[str, Tuple['ProcDecl', bool]] = {}

    def fresh_label(self, prefix: str = "L") -> str:
        self._label_counter += 1
        return f"{prefix}_{self._label_counter}"

    # --- Helpers ---

    def _emit(self, instr: Instr, label: str = None) -> LabeledInstr:
        return LabeledInstr(label, instr)

    def _var_offset(self, name: str) -> int:
        if name not in self._var_offsets:
            raise CodeGenError(f"Undefined variable: {name}")
        return self._var_offsets[name]

    # --- Expression evaluation (Section 4.5) ---
    #
    # Every expression is compiled *cleanly* (Axelsen, CC 2011, Sec. 4.5):
    #
    #   gen_expr(e) = (code, r)   with r a register allocated for the result.
    #
    #   Pre:  r and every free register hold 0.
    #   Post: r holds the value of e; every other register and all of memory
    #         are as before.  All temporaries are back in the free pool.
    #
    # The value is removed again by `_uneval(code, r)`, i.e. by running the
    # same code backwards (`_reverse_code`), never by clearing a register.
    # The caller must keep the memory cells e reads unchanged in between
    # (Janus: in `x op= e`, x must not occur in e).
    #
    # Every emitted instruction satisfies pisa.is_wf (locally invertible).
    # The lowering of each operator is listed in docs/EXPR_LOWERING.md, which
    # is the specification for the Rocq model (rocq/Compile.v).

    def gen_expr(self, expr: Expr) -> Tuple[List[LabeledInstr], str]:
        """Evaluate expression into a register (clean; see above).

        Returns (code, result_register).  The result register is committed;
        it is released by `_uneval(code, result_register)`.
        """
        free_before = set(self.reg.free)
        code, r = self._gen_expr(expr)
        leaked = free_before - self.reg.free - {r}
        if leaked or r not in free_before:          # pragma: no cover
            raise CodeGenError(
                f"internal: expression code leaked registers {sorted(leaked)}")
        return code, r

    def _uneval(self, code: List[LabeledInstr], r: str) -> List[LabeledInstr]:
        """Uncompute an expression: `code` run backwards clears `r`."""
        inv = self._reverse_code(code)
        self.reg.free_reg(r)
        return inv

    def _gen_expr(self, expr: Expr) -> Tuple[List[LabeledInstr], str]:
        if isinstance(expr, Const):
            return self._gen_const(expr)
        if isinstance(expr, Var):
            return self._gen_var(expr)
        if isinstance(expr, ArrayAccess):
            return self._gen_array_access(expr)
        if isinstance(expr, BinOp):
            return self._gen_binop(expr)
        raise CodeGenError(f"Unknown expression type: {type(expr)}")

    def _gen_const(self, expr: Const) -> Tuple[List[LabeledInstr], str]:
        """Load constant into a fresh register."""
        rd = self.reg.alloc()
        code = []
        if expr.value != 0:
            if expr.value > 0:
                code.append(self._emit(ADDI(rd, expr.value)))
            else:
                code.append(self._emit(SUBI(rd, -expr.value)))
        self.reg.commit_reg(rd)
        return code, rd

    def _gen_var(self, expr: Var) -> Tuple[List[LabeledInstr], str]:
        """Load variable value into a fresh register via EXCH-XOR-EXCH pattern."""
        return self._gen_var_copy(expr.name)

    def _gen_var_copy(self, name: str) -> Tuple[List[LabeledInstr], str]:
        """Copy variable value to a fresh register using EXCH-XOR-EXCH pattern."""
        offset = self._var_offset(name)
        ra = self.reg.alloc()
        rd = self.reg.alloc()
        rv = self.reg.alloc()  # temporary for the value
        code = [
            self._emit(ADDI(ra, offset)),   # ra = address
            self._emit(EXCH(rv, ra)),        # rv = mem[ra], mem[ra] = 0
            self._emit(XOR(rd, rv)),         # rd ^= rv (rd was 0, so rd = value)
            self._emit(EXCH(rv, ra)),        # mem[ra] = rv (restore memory)
            self._emit(SUBI(ra, offset)),    # clear ra
        ]
        self.reg.free_reg(ra)
        self.reg.free_reg(rv)
        self.reg.commit_reg(rd)
        return code, rd

    def _gen_array_access(self, expr: ArrayAccess) -> Tuple[List[LabeledInstr], str]:
        """Read array element x[e] into a fresh register rd.

        rd is allocated before the index is evaluated, so that it is none of
        the index code's temporaries: the index is uncomputed (its code run
        backwards) after the read, and needs those registers to be 0.
        """
        base_offset = self._var_offset(expr.name)
        rd = self.reg.alloc()
        idx_code, ri = self.gen_expr(expr.index)
        ra = self.reg.alloc()
        rv = self.reg.alloc()
        code = list(idx_code)
        code += [
            self._emit(ADDI(ra, base_offset)),
            self._emit(ADD(ra, ri)),         # ra = base + index
            self._emit(EXCH(rv, ra)),        # rv = mem[ra]
            self._emit(XOR(rd, rv)),         # copy value to rd
            self._emit(EXCH(rv, ra)),        # restore mem
            self._emit(SUB(ra, ri)),
            self._emit(SUBI(ra, base_offset)),
        ]
        self.reg.free_reg(ra)
        self.reg.free_reg(rv)
        code += self._uneval(idx_code, ri)   # index register back to 0
        self.reg.commit_reg(rd)
        return code, rd

    # Largest constant multiplier accepted, as a bit width.  The shift-and-add
    # chain costs one register and two instructions per bit, so an unbounded
    # multiplier would exhaust the 29 general-purpose registers.
    MUL_MAX_BITS = 16

    @staticmethod
    def _const_value(expr: Expr) -> Optional[int]:
        """Evaluate expr at compile time, or return None if it is not constant.

        Needed because the parser desugars unary minus to `0 - e`, so a literal
        like -3 reaches codegen as a BinOp rather than a Const.
        """
        if isinstance(expr, Const):
            return expr.value
        if isinstance(expr, BinOp):
            lv = CodeGen._const_value(expr.left)
            rv = CodeGen._const_value(expr.right)
            if lv is None or rv is None:
                return None
            op = expr.op
            if op == '+': return lv + rv
            if op == '-': return lv - rv
            if op == '*': return lv * rv
            if op == '^': return lv ^ rv
            if op == '&': return lv & rv
            if op == '|': return lv | rv
        return None

    def _gen_binop(self, expr: BinOp) -> Tuple[List[LabeledInstr], str]:
        """Generate code for binary operation (clean; see docs/EXPR_LOWERING.md)."""
        # Constant folding: evaluate at compile time when both operands are constants
        if isinstance(expr.left, Const) and isinstance(expr.right, Const):
            lv, rv, op = expr.left.value, expr.right.value, expr.op
            folded = None
            if op == '+':   folded = lv + rv
            elif op == '-': folded = lv - rv
            elif op == '^': folded = lv ^ rv
            elif op == '=':  folded = int(lv == rv)
            elif op == '!=': folded = int(lv != rv)
            elif op == '<':  folded = int(lv < rv)
            elif op == '>':  folded = int(lv > rv)
            elif op == '<=': folded = int(lv <= rv)
            elif op == '>=': folded = int(lv >= rv)
            elif op == '&&': folded = int(bool(lv) and bool(rv))
            elif op == '||': folded = int(bool(lv) or bool(rv))
            elif op == '&':  folded = lv & rv
            elif op == '|':  folded = lv | rv
            elif op == '*':  folded = lv * rv
            if folded is not None:
                return self._gen_const(Const(folded))

        expr = _logical_operands(expr)
        if _is_nonzero_test(expr):
            return self._gen_nonzero(expr.left)

        op = expr.op
        if op in _INPLACE:
            return self._gen_inplace(expr)
        if op == '*':
            return self._gen_mul(expr)
        if op in _COMBINE:
            return self._gen_combine(expr)
        raise CodeGenError(f"Unknown operator: {op}")

    def _gen_inplace(self, expr: BinOp) -> Tuple[List[LabeledInstr], str]:
        """`+`, `-`, `^`: combine into the left operand's register.

        These operators are injective in the left operand for a fixed right
        operand, so the right operand can be uncomputed at once and the left
        register simply keeps the result (Axelsen's `rl op= rr`):

            <l -> rl> ; <r -> rr> ; OP rl rr ; <r -> rr>^-1        (rl != rr)

        A compile-time constant right operand k needs no register:

            <l -> rl> ; ADDI/SUBI/XORI rl k       (nothing when k = 0)
        """
        code, rl = self.gen_expr(expr.left)
        k = self._const_value(expr.right)
        if k is not None:
            code.extend(self._inplace_imm(expr.op, rl, k))
            return code, rl
        right_code, rr = self.gen_expr(expr.right)
        code.extend(right_code)
        code.append(self._emit(_INPLACE[expr.op](rl, rr)))
        code.extend(self._uneval(right_code, rr))
        return code, rl

    def _inplace_imm(self, op: str, r: str, k: int) -> List[LabeledInstr]:
        """r := r op k for a constant k (no instruction when k = 0)."""
        if k == 0:
            return []
        if op == '^':
            return [self._emit(XORI(r, k))]
        if op == '-':
            k = -k
        return [self._emit(ADDI(r, k) if k > 0 else SUBI(r, -k))]

    def _gen_mul(self, expr: BinOp) -> Tuple[List[LabeledInstr], str]:
        """`e * k` / `k * e` for a compile-time constant k (shift and add).

        PISA has no MUL instruction, and a data-dependent multiply loop cannot
        be inverted by the straight-line machinery, so one operand must be a
        compile-time constant.  With n = |k| and p0 = rv, p(i+1) = 2 p(i):

            re                                  ; fresh, allocated first
            <e -> rv>
            ADD q1 rv ; ADD q1 rv               ; q1 = 2 rv   (fresh, q1 != rv)
            ADD q2 q1 ; ADD q2 q1               ; q2 = 4 rv   ... up to the top bit
            ADD re p(i)       for every bit i set in n
            NEG re            if k < 0
            (the doubling chain)^-1             ; q's back to 0
            <e -> rv>^-1

        `ADD p p` would double in one instruction but is not invertible.
        k = 0 emits nothing (re is already 0).
        """
        k = self._const_value(expr.right)
        side = expr.left
        if k is None:
            k = self._const_value(expr.left)
            side = expr.right
        if k is None:
            raise CodeGenError(
                "Multiplication requires a constant operand; "
                "variable * variable is not supported in PISA codegen"
            )
        if abs(k) >= (1 << self.MUL_MAX_BITS):
            raise CodeGenError(
                f"Multiplier {k} exceeds {self.MUL_MAX_BITS}-bit limit "
                f"for constant multiplication"
            )
        re = self.reg.alloc()
        self.reg.commit_reg(re)
        if k == 0:
            return [], re
        n = abs(k)
        val_code, rv = self.gen_expr(side)
        chain: List[LabeledInstr] = []
        powers = [rv]
        for _ in range(n.bit_length() - 1):
            q = self.reg.alloc()
            chain.append(self._emit(ADD(q, powers[-1])))
            chain.append(self._emit(ADD(q, powers[-1])))   # q = 2 * previous
            powers.append(q)
        code = list(val_code) + chain
        for i, p in enumerate(powers):
            if (n >> i) & 1:
                code.append(self._emit(ADD(re, p)))
        if k < 0:
            code.append(self._emit(NEG(re)))
        code.extend(self._reverse_code(chain))
        for q in powers[1:]:
            self.reg.free_reg(q)
        code.extend(self._uneval(val_code, rv))
        return code, re

    def _gen_combine(self, expr: BinOp) -> Tuple[List[LabeledInstr], str]:
        """Comparisons, `&&`, `||`, `&`, `|`: combine into a fresh register.

            re                                  ; fresh, allocated first
            <l -> rl> ; <r -> rr>
            <combine re rl rr>                  ; re ^= f(rl, rr), see _COMBINE
            <r -> rr>^-1 ; <l -> rl>^-1

        re is allocated before the operands so that it is none of their
        temporaries (their code is run backwards afterwards and needs those
        registers to be 0); re, rl, rr are pairwise distinct.  The operands
        of `&&` / `||` are 0/1 here (`_logical_operands`).
        """
        re = self.reg.alloc()
        left_code, rl = self.gen_expr(expr.left)
        right_code, rr = self.gen_expr(expr.right)
        code = list(left_code) + list(right_code)
        code.extend(self._emit(i) for i in _COMBINE[expr.op](re, rl, rr))
        code.extend(self._uneval(right_code, rr))
        code.extend(self._uneval(left_code, rl))
        self.reg.commit_reg(re)
        return code, re

    def _gen_nonzero(self, e: Expr) -> Tuple[List[LabeledInstr], str]:
        """Evaluate (e != 0) into a fresh register holding only the 0/1 flag.

            rf                                  ; fresh, allocated first
            <e -> rv>
            SLTX rf rv r0                       ; rf ^= (v < 0)
            SLTX rf r0 rv                       ; rf ^= (0 < v)
            <e -> rv>^-1

        (v < 0) and (0 < v) are mutually exclusive, so XOR-ing both bits adds
        exactly (v != 0); no NEG, so INT_MIN is fine under fixed width.  r0 as
        an SLTX operand is special-cased by the PAL expansion
        (tools/pisa2pal.py), which never writes it.
        """
        rf = self.reg.alloc()
        val_code, rv = self.gen_expr(e)
        code = list(val_code)
        code.append(self._emit(SLTX(rf, rv, "r0")))
        code.append(self._emit(SLTX(rf, "r0", rv)))
        code.extend(self._uneval(val_code, rv))
        self.reg.commit_reg(rf)
        return code, rf

    def _reverse_code(self, code: List[LabeledInstr]) -> List[LabeledInstr]:
        """Reverse a sequence of instructions (run backwards, invert each)."""
        result = []
        for li in reversed(code):
            inv = self._invert_instr(li.instr)
            result.append(LabeledInstr(None, inv))
        return result

    def _invert_instr(self, instr: Instr) -> Instr:
        """Invert a single PISA instruction."""
        if isinstance(instr, ADD):
            return SUB(instr.rd, instr.rs)
        if isinstance(instr, SUB):
            return ADD(instr.rd, instr.rs)
        if isinstance(instr, NEG):
            return NEG(instr.rd)
        if isinstance(instr, XOR):
            return XOR(instr.rd, instr.rs)  # self-inverse
        if isinstance(instr, ADDI):
            return SUBI(instr.rd, instr.c)
        if isinstance(instr, SUBI):
            return ADDI(instr.rd, instr.c)
        if isinstance(instr, XORI):
            return XORI(instr.rd, instr.c)  # self-inverse
        if isinstance(instr, EXCH):
            return EXCH(instr.rd, instr.rs)  # self-inverse
        if isinstance(instr, ORX):
            return ORX(instr.rd, instr.rs, instr.rt)    # self-inverse (is_wf)
        if isinstance(instr, ANDX):
            return ANDX(instr.rd, instr.rs, instr.rt)   # self-inverse (is_wf)
        if isinstance(instr, SLTX):
            return SLTX(instr.rd, instr.rs, instr.rt)   # self-inverse (is_wf)
        if isinstance(instr, BRA):
            return RBRA(instr.label)
        if isinstance(instr, RBRA):
            return BRA(instr.label)
        if isinstance(instr, SWAPBR):
            return SWAPBR(instr.rd)
        raise CodeGenError(f"Cannot invert instruction: {type(instr).__name__}")

    # --- Statement code generation ---

    def gen_stmt(self, stmt: Stmt) -> List[LabeledInstr]:
        """Generate PISA code for a statement."""
        if isinstance(stmt, Skip):
            return []

        if isinstance(stmt, Seq):
            code = []
            for s in stmt.stmts:
                code.extend(self.gen_stmt(s))
            return code

        if isinstance(stmt, AssignVar):
            return self._gen_assign_var(stmt)

        if isinstance(stmt, AssignArr):
            return self._gen_assign_arr(stmt)

        if isinstance(stmt, Swap):
            return self._gen_swap(stmt)

        if isinstance(stmt, Call):
            return self._gen_call(stmt)

        if isinstance(stmt, Uncall):
            return self._gen_uncall(stmt)

        if isinstance(stmt, If):
            return self._gen_if(stmt)

        if isinstance(stmt, From):
            return self._gen_from(stmt)

        if isinstance(stmt, Print):
            return []  # Print is not supported in PISA

        raise CodeGenError(f"Unknown statement type: {type(stmt)}")

    def _gen_assign_var(self, stmt: AssignVar) -> List[LabeledInstr]:
        """Generate code for x ⊕= e (Fig. 6)."""
        offset = self._var_offset(stmt.var)

        # Janus forbids the assigned variable in its own right-hand side:
        # `x -= x` would zero x (not invertible), and e must still have the
        # same value when it is uncomputed after the update.  (An `x op= x`
        # special case used to emit `ADD rd rd` / `XOR rd rd`, which are not
        # locally invertible.)
        if stmt.var in _expr_vars(stmt.expr):
            raise CodeGenError(
                f"`{stmt.var} {stmt.op} ...`: the assigned variable must not "
                f"occur in the right-hand side (Janus)")

        # Fast path: constant RHS avoids a register and two instructions.
        # x += k  →  EXCH rd ra; ADDI rd k; EXCH rd ra  (no re needed)
        if isinstance(stmt.expr, Const):
            k = stmt.expr.value
            ra = self.reg.alloc()
            rd = self.reg.alloc()
            code = []
            code.append(self._emit(ADDI(ra, offset)))
            code.append(self._emit(EXCH(rd, ra)))
            if k != 0:
                if stmt.op == '+=':
                    code.append(self._emit(ADDI(rd, k)))
                elif stmt.op == '-=':
                    code.append(self._emit(SUBI(rd, k)))
                elif stmt.op == '^=':
                    code.append(self._emit(XORI(rd, k)))
                else:
                    raise CodeGenError(f"Unknown assign op: {stmt.op}")
            code.append(self._emit(EXCH(rd, ra)))
            code.append(self._emit(SUBI(ra, offset)))
            self.reg.free_reg(ra)
            self.reg.free_reg(rd)
            return code

        # General path: evaluate e → re, apply, unevaluate.
        # 1. Evaluate e → re
        eval_code, re = self.gen_expr(stmt.expr)

        # 2-3. Load x into rd via address
        ra = self.reg.alloc()
        rd = self.reg.alloc()
        code = list(eval_code)
        code.append(self._emit(ADDI(ra, offset)))     # ra = &x
        code.append(self._emit(EXCH(rd, ra)))          # rd = x, mem[ra] = 0

        # 4. Apply operation
        if stmt.op == '+=':
            code.append(self._emit(ADD(rd, re)))
        elif stmt.op == '-=':
            code.append(self._emit(SUB(rd, re)))
        elif stmt.op == '^=':
            code.append(self._emit(XOR(rd, re)))
        else:
            raise CodeGenError(f"Unknown assign op: {stmt.op}")

        # 5. Store back
        code.append(self._emit(EXCH(rd, ra)))          # mem[ra] = rd (updated)

        # 6. Clear address register
        code.append(self._emit(SUBI(ra, offset)))
        self.reg.free_reg(ra)
        self.reg.free_reg(rd)

        # 7. Unevaluate e: run its code backwards (clears re)
        code.extend(self._uneval(eval_code, re))

        return code

    def _gen_assign_arr(self, stmt: AssignArr) -> List[LabeledInstr]:
        """Generate code for x[e1] ⊕= e2 (Fig. 6)."""
        base_offset = self._var_offset(stmt.var)

        # 1. Evaluate index e1 → ra
        idx_code, ri = self.gen_expr(stmt.idx)

        # 2. Compute address: base + index
        ra = self.reg.alloc()
        code = list(idx_code)
        code.append(self._emit(ADDI(ra, base_offset)))
        code.append(self._emit(ADD(ra, ri)))

        # 3. Evaluate e2 → re
        eval_code, re = self.gen_expr(stmt.expr)
        code.extend(eval_code)

        # 4. Load array element
        rd = self.reg.alloc()
        code.append(self._emit(EXCH(rd, ra)))

        # 5. Apply operation
        if stmt.op == '+=':
            code.append(self._emit(ADD(rd, re)))
        elif stmt.op == '-=':
            code.append(self._emit(SUB(rd, re)))
        elif stmt.op == '^=':
            code.append(self._emit(XOR(rd, re)))

        # 6. Store back
        code.append(self._emit(EXCH(rd, ra)))
        self.reg.free_reg(rd)

        # 7. Unevaluate e2 (its code backwards; rd is 0 again)
        code.extend(self._uneval(eval_code, re))

        # 8-9. Clear address and unevaluate index
        code.append(self._emit(SUB(ra, ri)))
        code.append(self._emit(SUBI(ra, base_offset)))
        self.reg.free_reg(ra)

        code.extend(self._uneval(idx_code, ri))

        return code

    def _gen_swap(self, stmt: Swap) -> List[LabeledInstr]:
        """Generate code for x <=> y."""
        code = []

        # Get addresses of both sides
        def get_addr(name, idx_expr):
            offset = self._var_offset(name)
            ra = self.reg.alloc()
            addr_code = [self._emit(ADDI(ra, offset))]
            ri = None
            idx_code: List[LabeledInstr] = []
            if idx_expr is not None:
                idx_code, ri = self.gen_expr(idx_expr)
                addr_code = list(idx_code) + addr_code
                addr_code.append(self._emit(ADD(ra, ri)))
            return addr_code, ra, ri, idx_code

        lhs_code, la, li, lhs_idx_code = get_addr(stmt.lhs, stmt.lhs_idx)
        rhs_code, ra2, ri2, rhs_idx_code = get_addr(stmt.rhs, stmt.rhs_idx)

        code.extend(lhs_code)
        code.extend(rhs_code)

        # Load both values
        t1 = self.reg.alloc()
        t2 = self.reg.alloc()
        code.append(self._emit(EXCH(t1, la)))    # t1 = lhs value
        code.append(self._emit(EXCH(t2, ra2)))   # t2 = rhs value

        # Swap and store back
        code.append(self._emit(EXCH(t1, ra2)))   # mem[rhs] = old lhs
        code.append(self._emit(EXCH(t2, la)))    # mem[lhs] = old rhs

        self.reg.free_reg(t1)
        self.reg.free_reg(t2)

        # Clear addresses (reverse order)
        if ri2 is not None:
            code.append(self._emit(SUB(ra2, ri2)))
        rhs_offset = self._var_offset(stmt.rhs)
        code.append(self._emit(SUBI(ra2, rhs_offset)))
        self.reg.free_reg(ra2)
        if ri2 is not None:
            code.extend(self._uneval(rhs_idx_code, ri2))

        if li is not None:
            code.append(self._emit(SUB(la, li)))
        lhs_offset = self._var_offset(stmt.lhs)
        code.append(self._emit(SUBI(la, lhs_offset)))
        self.reg.free_reg(la)
        if li is not None:
            code.extend(self._uneval(lhs_idx_code, li))

        return code

    def _gen_call(self, stmt: Call) -> List[LabeledInstr]:
        """Generate procedure call (Fig. 5): BRA f, or inline the body."""
        if stmt.proc in self._inline_procs:
            proc, _can_uncall = self._inline_procs[stmt.proc]
            return self.gen_stmt(proc.body)
        return [self._emit(BRA(_proc_label(stmt.proc)))]

    def _gen_uncall(self, stmt: Uncall) -> List[LabeledInstr]:
        """Generate procedure uncall: run f backwards.

        A plain `RBRA f` does not work here: this interpreter has no Pendulum
        direction bit, so it would execute f's body FORWARD, making `uncall f`
        behave exactly like `call f`.  Instead we call an inverted companion
        procedure `f_inv`, whose body is invert_stmt(f.body); gen_program emits
        one for every uncalled procedure.  Inlined bodies are inverted in place.
        """
        if stmt.proc in self._inline_procs:
            proc, can_uncall = self._inline_procs[stmt.proc]
            if can_uncall:
                return self.gen_stmt(invert_stmt(proc.body))
        return [self._emit(BRA(_inv_proc_name(stmt.proc)))]

    def _gen_if(self, stmt: If) -> List[LabeledInstr]:
        """Generate if-then-else (Fig. 11).

        if e1 then S1 else S2 fi e2

                    <rt ^= e1>              ; e1, e2 normalised to 0/1 by _as_flag
        test:       BEQ rt r0 false
                    XORI rt 1               ; then path: rt = 0
                    <S1>
                    XORI rt 1               ; then path: rt = 1
        assert:     <rt ^= e2>              ; both paths join here: rt = 0 iff e2
                                            ; matches the path taken
        assert_true: BRA end
        false:      BRA test                ; paired with the BEQ
                    <S2>                    ; else path: rt = 0
                    BRA assert
        end:        BRA assert_true         ; paired; falls through below
                    BNE rt r0 finish        ; violated -> halt with rt = 1
        """
        test_false = self.fresh_label("if_false")
        test_label = self.fresh_label("if_test")
        assert_label = self.fresh_label("if_assert")
        assert_true = self.fresh_label("if_assert_true")
        end_label = self.fresh_label("if_end")

        code = []

        # --- Test e1 ---
        rt = self.reg.alloc()
        self.reg.commit_reg(rt)

        code.extend(self._gen_flag_xor(rt, _as_flag(stmt.test)))

        # Branch
        code.append(self._emit(BEQ(rt, "r0", test_false), test_label))
        code.append(self._emit(XORI(rt, 1)))

        # --- Then branch (S1) ---
        then_code = self.gen_stmt(stmt.then_)
        code.extend(then_code)

        # --- Assert e2 (both paths) ---
        #
        # rt is the path flag here: 1 on the then path, 0 on the else path.
        # XORing eval(e2) into it leaves 0 exactly when the assertion holds
        # (then requires e2 true, else requires e2 false), so a *correct*
        # program leaves rt clean and a violated assertion leaves garbage,
        # which the interpreter reports at FINISH.
        #
        # e2 is evaluated on BOTH paths.  It used to be skipped on the then
        # path by a `BNE rt r0 assert_true`, which meant the assertion was
        # never checked at all and reversibility was silently lost.
        code.append(self._emit(XORI(rt, 1)))

        # Evaluate assertion.  The else path joins at assert_label, the first
        # instruction after the XORI.  When e2 needs no code (`fi 0`) that is
        # the XOR itself; the label used to be dropped there, leaving
        # `BRA if_assert` dangling.
        assert_code = self._gen_flag_xor(rt, _as_flag(stmt.fi))
        assert_code[0] = LabeledInstr(assert_label, assert_code[0].instr)
        code.extend(assert_code)

        code.append(self._emit(BRA(end_label), assert_true))

        # --- False branch entry ---
        code.append(self._emit(BRA(test_label), test_false))

        # --- Else branch (S2) ---
        else_code = self.gen_stmt(stmt.else_)
        code.extend(else_code)

        # --- Assert e2 (false path) ---
        # (same assertion structure for the false branch)
        code.append(self._emit(BRA(assert_label)))

        # --- End ---
        # The Pendulum pair (assert_true: BRA end_label / end_label: BRA assert_true)
        # exits via pc = end_label + 1.  Both paths leave rt = flag XOR eval(e2),
        # which is 0 for a correct program — so rt needs no clearing, and any
        # nonzero value left here is a genuine assertion violation rather than
        # something to be wiped.  It is reported at once: rt goes back to the
        # free pool, and the next statement that XORs a value into it (every
        # allocation assumes a zero register) could cancel the 1 and hide it.
        code.append(self._emit(BRA(assert_true), end_label))  # Pendulum pair second
        code.append(self._emit(BNE(rt, "r0", ASSERT_FAIL_LABEL)))
        self.reg.free_reg(rt)

        return code

    def _gen_from(self, stmt: From) -> List[LabeledInstr]:
        """Generate from-do-loop-until (Fig. 12).

        from e1 do S1 loop S2 until e2

                <rt ^= e1>           ; entry assertion: e1 must hold
                XORI rt 1            ; rt = 0 iff e1
                BNE rt r0 finish     ; violated -> halt with rt = 1
        do:     <S1 (do body)>       ; rt = 0
        test:   <rt ^= e2>           ; exit test
                BEQ rt r0 loop       ; e2 false: rt is already 0
                XORI rt 1            ; e2 true: rt 1 -> 0
                BRA exit
        loop:   ADDI r0 0            ; landing NOP (removed by remove_nops)
                <S2 (loop body)>     ; rt = 0
                <rt ^= e1>           ; re-entry assertion: rt = e1
                BNE rt r0 finish     ; e1 must NOT hold; violated -> rt = 1
                BRA do               ; rt = 0
        exit:   ADDI r0 0            ; rt = 0

        rt is 0 whenever S1 or S2 runs, and at `do`, `loop` and `exit`.  This
        matters beyond tidiness: a `call` in S1 or S2 runs a callee whose body
        was compiled assuming the registers from r3 up are clean, and rt is one
        of them.  S2 used to run with rt = 1 (`loop: XORI rt 1 ; <S2> ; <rt ^=
        e1> ; XORI rt 1 ; ...`), so in `from i = 0 do i += 1 loop call f until
        i = 3; call f` with `f: c += 5` f's body ran on a dirty r3 and the
        program ended with c = 5 instead of Janus's 15.  After `BEQ rt r0
        loop` is taken rt *is* 0, so the two XORIs around S2 were dropped; `loop:`
        heads a labeled NOP (instead of S2's first line, as `do:` does for
        S1) so that the layout does not depend on S2's first instruction.

        e1 and e2 are normalised to 0/1 by _as_flag, so every flag update is
        an XOR with a known bit and rt is restored by XORI instead of being
        wiped (`XOR rt rt`, which discarded the assertion and is not
        reversible).  A violated assertion cannot be left in rt as `_gen_if`
        does, because rt steers the loop test: a stale 1 at `test` would flip
        the exit decision and could clear itself.  So the violation jumps to
        `finish`, where the interpreter reports the nonzero rt as garbage.  A
        correct program never takes these branches.
        """
        test_label = self.fresh_label("from_test")
        loop_body = self.fresh_label("from_loop")
        exit_label = self.fresh_label("from_exit")
        entry_do = self.fresh_label("from_do")
        from_ = _as_flag(stmt.from_)
        until = _as_flag(stmt.until)

        code = []
        rt = self.reg.alloc()
        self.reg.commit_reg(rt)

        # --- Entry assertion (e1 must be true) ---
        code.extend(self._gen_flag_xor(rt, from_))
        code.append(self._emit(XORI(rt, 1)))
        code.append(self._emit(BNE(rt, "r0", ASSERT_FAIL_LABEL)))

        # entry_do label: start of do body
        code_do = self.gen_stmt(stmt.do_)
        if code_do:
            code_do[0] = LabeledInstr(entry_do, code_do[0].instr)
        else:
            code_do = [self._emit(ADDI("r0", 0), entry_do)]  # NOP with label

        code.extend(code_do)

        # --- Test e2 (exit condition) ---
        test_code = self._gen_flag_xor(rt, until)
        test_code[0] = LabeledInstr(test_label, test_code[0].instr)
        code.extend(test_code)

        # Branch on result
        code.append(self._emit(BEQ(rt, "r0", loop_body)))
        code.append(self._emit(XORI(rt, 1)))   # rt was 1 (e2 true)
        code.append(self._emit(BRA(exit_label)))

        # --- Loop body ---
        # The BEQ above is taken only when rt = 0, so S2 runs with a clean
        # flag (a callee in S2 relies on it).  The label sits on a NOP.
        loop_code = [self._emit(ADDI("r0", 0), loop_body)]  # NOP with label

        body_code = self.gen_stmt(stmt.loop_)
        loop_code.extend(body_code)

        # Re-entry assertion: rt = e1, which must be 0 (e1 false).
        loop_code.extend(self._gen_flag_xor(rt, from_))
        loop_code.append(self._emit(BNE(rt, "r0", ASSERT_FAIL_LABEL)))

        loop_code.append(self._emit(BRA(entry_do)))
        code.extend(loop_code)

        # --- Exit ---
        self.reg.free_reg(rt)
        # Add exit label
        code.append(self._emit(ADDI("r0", 0), exit_label))  # NOP with label

        return code

    def _gen_flag_xor(self, rt: str, e: Expr) -> List[LabeledInstr]:
        """rt ^= e for a 0/1 expression e, leaving no other register dirty.

        Always returns at least one instruction (the XOR), so the caller can
        label its first element.
        """
        eval_code, re = self.gen_expr(e)
        code = list(eval_code)
        code.append(self._emit(XOR(rt, re)))
        code.extend(self._uneval(eval_code, re))
        return code

    # --- Procedure code generation (Fig. 5) ---

    def gen_proc(self, proc: ProcDecl, label: Optional[str] = None) -> List[LabeledInstr]:
        """Generate code for a procedure definition.

        f_top: BRA f_bot
        f:     SUBI r1 1
               EXCH r2 r1
               SWAPBR r2
               NEG r2
               EXCH r2 r1
               ADDI r1 1
               <code for f body>
        f_bot: BRA f_top
        """
        if label is None:
            label = _proc_label(proc.name)
        f_top = f"{label}_top"
        f_bot = f"{label}_bot"

        code = [
            self._emit(BRA(f_bot), f_top),
            self._emit(SUBI("r1", 1), label),
            self._emit(EXCH("r2", "r1")),
            self._emit(SWAPBR("r2")),
            self._emit(NEG("r2")),
            self._emit(EXCH("r2", "r1")),
            self._emit(ADDI("r1", 1)),
        ]

        body_code = self.gen_stmt(proc.body)
        code.extend(body_code)

        code.append(self._emit(BRA(f_top), f_bot))

        return code

    # --- Program code generation (Fig. 4) ---

    def gen_program(self, prog: Program) -> List[LabeledInstr]:
        """Generate complete PISA program.

        <variable DATA declarations>
        <procedure code for each procedure>
        start: START
               ADDI r1 <stack_offset>
               BRA main
        finish: FINISH
                SUBI r1 <stack_offset>
        """
        code = []

        # 1. Variable DATA declarations
        offset = 0
        for vd in prog.vars:
            self._var_offsets[vd.name] = offset
            self._var_sizes[vd.name] = vd.size
            for val in vd.init:
                code.append(self._emit(DATA(val)))
            offset += vd.size
        self._total_vars = offset

        # Stack starts after variables; size based on max call depth
        call_depth = _compute_call_depth(prog)
        stack_offset = offset + max(call_depth * 2, 4)

        # 2. Identify inlinable procedures before generating any code.
        #
        #    A procedure f is inlinable when the call overhead (9 instructions
        #    for f_top/f_bot + prologue) can be eliminated:
        #      - call-only inline:  call_count ≤ 1 AND uncall_count == 0
        #        (body may contain any statements; no reversal needed)
        #      - full inline:       call_count ≤ 1 AND uncall_count ≤ 1
        #                           AND body is branch-free (safe to reverse)
        #
        #    main is never inlined (it is the entry point).
        reachable_fwd, reachable_inv = _needed_procs(prog)
        reachable = reachable_fwd | reachable_inv
        total_call_counts: Dict[str, Tuple[int, int]] = {}
        for proc in prog.procs:
            for name, (c, u) in _count_all_calls(proc.body).items():
                tc, tu = total_call_counts.get(name, (0, 0))
                total_call_counts[name] = (tc + c, tu + u)

        proc_map = {p.name: p for p in prog.procs}
        for name in reachable:
            if name == prog.main_proc or name not in proc_map:
                continue
            proc = proc_map[name]
            c, u = total_call_counts.get(name, (0, 0))
            if c <= 1 and u == 0:
                # Call-only inline: body executed forward only.
                self._inline_procs[name] = (proc, False)
            elif (c <= 1 and u <= 1
                  and _is_simple_for_inline(proc.body)
                  and _estimate_inline_cost(proc.body) <= _FULL_INLINE_LIMIT):
                # Full inline: body executed forward for both call and uncall.
                # Size limit prevents code growth when the body is large enough
                # that emitting it twice costs more than the call overhead.
                self._inline_procs[name] = (proc, True)

        # A procedure's name is its entry label, so it must not be one of the
        # program labels: `finish` in particular is where a violated `fi` or
        # loop assertion jumps (ASSERT_FAIL_LABEL).
        for proc in prog.procs:
            if proc.name in ("start", ASSERT_FAIL_LABEL):
                raise CodeGenError(
                    f"procedure name `{proc.name}` is reserved (program label)")

        # 3. Procedure code (skip dead and inlined procedures)
        for proc in prog.procs:
            if proc.name in reachable_fwd and proc.name not in self._inline_procs:
                proc_code = self.gen_proc(proc)
                code.extend(proc_code)

        # 3b. Inverted companion procedures, one per uncalled procedure.
        #     `uncall f` compiles to `BRA f_inv`, so f_inv must exist whenever
        #     the uncall is not inlined away.
        for name in sorted(reachable_inv):
            if name not in proc_map:
                continue
            inlined = self._inline_procs.get(name)
            if inlined is not None and inlined[1]:
                continue          # uncall of this one is inlined; no branch target needed
            inv_proc = ProcDecl(name, invert_stmt(proc_map[name].body))
            code.extend(self.gen_proc(inv_proc, _inv_proc_name(name)))

        # 3. Entry/exit
        code.append(self._emit(START(), "start"))
        code.append(self._emit(ADDI("r1", stack_offset)))
        code.append(self._emit(BRA(_proc_label(prog.main_proc))))

        code.append(self._emit(FINISH(), "finish"))
        code.append(self._emit(SUBI("r1", stack_offset)))

        return code


def _cancels(a: Instr, b: Instr) -> bool:
    """Return True if instructions a and b are mutual inverses and cancel.

    Pairs detected:
      XORI rd c  ; XORI rd c    (XOR with same constant, self-inverse)
      ADDI rd c  ; SUBI rd c    (add then subtract same constant)
      SUBI rd c  ; ADDI rd c    (subtract then add same constant)
      ADD  rd rs ; SUB  rd rs   (add then subtract same register)
      SUB  rd rs ; ADD  rd rs   (subtract then add same register)
      XOR  rd rs ; XOR  rd rs   (XOR same register, self-inverse)
      NEG  rd    ; NEG  rd      (negate twice)
      EXCH rd rs ; EXCH rd rs   (swap is self-inverse)
    """
    if isinstance(a, XORI) and isinstance(b, XORI):
        return a.rd == b.rd and a.c == b.c
    if isinstance(a, ADDI) and isinstance(b, SUBI):
        return a.rd == b.rd and a.c == b.c
    if isinstance(a, SUBI) and isinstance(b, ADDI):
        return a.rd == b.rd and a.c == b.c
    # Register pairs cancel only with DISTINCT operands: XOR r r clears r
    # (its pair's net effect is r := 0, not identity), ADD r r doubles, and
    # EXCH r r is not self-inverse.  See rocq/Opt.v (`cancels_undo`).
    if isinstance(a, ADD) and isinstance(b, SUB):
        return a.rd == b.rd and a.rs == b.rs and a.rd != a.rs
    if isinstance(a, SUB) and isinstance(b, ADD):
        return a.rd == b.rd and a.rs == b.rs and a.rd != a.rs
    if isinstance(a, XOR) and isinstance(b, XOR):
        return a.rd == b.rd and a.rs == b.rs and a.rd != a.rs
    if isinstance(a, NEG) and isinstance(b, NEG):
        return a.rd == b.rd
    if isinstance(a, EXCH) and isinstance(b, EXCH):
        return a.rd == b.rd and a.rs == b.rs and a.rd != a.rs
    return False


def _peephole_pass(code: List[LabeledInstr]) -> List[LabeledInstr]:
    """Single pass of the peephole optimiser."""
    result = []
    i = 0
    while i < len(code):
        cur = code[i]
        if (i + 1 < len(code)
                and _cancels(cur.instr, code[i + 1].instr)
                and code[i + 1].label is None
                # a labeled pair at the very end has nowhere to carry its
                # label; cancelling it would leave any branch to it dangling
                and not (cur.label is not None and i + 2 >= len(code))):
            # Both cancel: skip them; forward the first's label if any
            if cur.label is not None and i + 2 < len(code):
                next_li = code[i + 2]
                if next_li.label is None:
                    code[i + 2] = LabeledInstr(cur.label, next_li.instr)
                else:
                    # Both have labels; keep a NOP to carry the first label
                    result.append(LabeledInstr(cur.label, XORI(cur.instr.rd if hasattr(cur.instr, 'rd') else 'r0', 0)))
            i += 2
            continue
        result.append(cur)
        i += 1
    return result


def peephole(code: List[LabeledInstr]) -> List[LabeledInstr]:
    """Remove adjacent cancelling instruction pairs, iterated to fixed point.

    Iterating is necessary so that cancelling a pair can expose a new pair:
    e.g.  SUBI ra c; ADDI ra c  cancels, then reveals  EXCH rd ra; EXCH rd ra
    which enables store-block fusion for consecutive same-variable assignments.
    """
    while True:
        new_code = _peephole_pass(code)
        if len(new_code) == len(code):
            break
        code = new_code
    return code


def _collect_calls(stmt) -> set:
    """Recursively collect all procedure names called from a statement."""
    from syntax import Call, Uncall, Seq, If, From
    if isinstance(stmt, (Call, Uncall)):
        return {stmt.proc}
    if isinstance(stmt, Seq):
        result = set()
        for s in stmt.stmts:
            result |= _collect_calls(s)
        return result
    if isinstance(stmt, If):
        return _collect_calls(stmt.then_) | _collect_calls(stmt.else_)
    if isinstance(stmt, From):
        return _collect_calls(stmt.do_) | _collect_calls(stmt.loop_)
    return set()


def _reachable_procs(prog: Program) -> set:
    """Return the set of procedure names reachable from main (including main)."""
    call_graph = {p.name: _collect_calls(p.body) for p in prog.procs}
    reachable = set()

    def visit(name: str):
        if name in reachable:
            return
        reachable.add(name)
        for callee in call_graph.get(name, set()):
            visit(callee)

    visit(prog.main_proc)
    return reachable


def _count_all_calls(stmt) -> Dict[str, Tuple[int, int]]:
    """Return {proc: (call_count, uncall_count)} for every Call/Uncall in stmt."""
    counts: Dict[str, Tuple[int, int]] = {}

    def merge(name: str, dc: int, du: int) -> None:
        c, u = counts.get(name, (0, 0))
        counts[name] = (c + dc, u + du)

    def visit(s) -> None:
        from syntax import Call, Uncall, Seq, If, From
        if isinstance(s, Call):
            merge(s.proc, 1, 0)
        elif isinstance(s, Uncall):
            merge(s.proc, 0, 1)
        elif isinstance(s, Seq):
            for sub in s.stmts:
                visit(sub)
        elif isinstance(s, If):
            visit(s.then_)
            visit(s.else_)
        elif isinstance(s, From):
            visit(s.do_)
            visit(s.loop_)

    visit(stmt)
    return counts


_FULL_INLINE_LIMIT = 10
"""Max estimated instruction count for the full-inline (call+uncall) case.

Without inlining: body_size + 11 instructions (9 overhead + 1 BRA + 1 RBRA).
With inlining:    2 * body_size instructions (body emitted twice).
Break-even at body_size = 11; inlining wins when body_size <= 10.
"""


def _estimate_inline_cost(stmt) -> int:
    """Estimate the instruction count of a statement body for inlining decisions.

    Uses conservative (over) estimates to avoid unexpected code growth.
    """
    if isinstance(stmt, (Skip, Print)):
        return 0
    if isinstance(stmt, AssignVar):
        if isinstance(stmt.expr, Var) and stmt.expr.name == stmt.var:
            return 3   # self-reference fast path: EXCH + op + EXCH
        if isinstance(stmt.expr, Const):
            return 3   # const fast path: EXCH + ADDI/SUBI/XORI + EXCH
        return 14      # general: eval + op + uneval
    if isinstance(stmt, AssignArr):
        return 15
    if isinstance(stmt, Swap):
        return 8
    if isinstance(stmt, Seq):
        return sum(_estimate_inline_cost(s) for s in stmt.stmts)
    return 9999        # If, From, Call, Uncall: large; _is_simple_for_inline already rejects them


def _is_simple_for_inline(stmt) -> bool:
    """Return True if stmt contains no control flow or calls.

    A 'simple' body contains only Skip, AssignVar, AssignArr, Swap, Print
    (and Seq thereof).  Safe to inline for both call and uncall because
    only linear data instructions are generated (no branch labels).
    """
    from syntax import Skip, AssignVar, AssignArr, Swap, Print, Seq
    if isinstance(stmt, (Skip, AssignVar, AssignArr, Swap, Print)):
        return True
    if isinstance(stmt, Seq):
        return all(_is_simple_for_inline(s) for s in stmt.stmts)
    return False  # If, From, Call, Uncall → not safe to reverse naively


def _needed_procs(prog: Program) -> Tuple[set, set]:
    """Return (F, I): procedures needing forward code, and needing an inverse.

    Inside an *inverted* body a `call g` has become `uncall g` and vice versa,
    so the two sets are mutually recursive.  Computing them together also keeps
    dead code out: a procedure that is only ever uncalled gets `f_inv` emitted
    and no forward copy.
    """
    proc_map = {p.name: p for p in prog.procs}
    fwd, inv, work = set(), set(), []

    def add(name: str, inverted: bool) -> None:
        if name not in proc_map:
            return
        target = inv if inverted else fwd
        if name not in target:
            target.add(name)
            work.append((name, inverted))

    add(prog.main_proc, False)
    while work:
        name, inverted = work.pop()
        for g, (calls, uncalls) in _count_all_calls(proc_map[name].body).items():
            if calls:
                add(g, inverted)
            if uncalls:
                add(g, not inverted)
    return fwd, inv


def _compute_call_depth(prog: Program) -> int:
    """Compute maximum procedure call nesting depth by DFS on the call graph."""
    call_graph = {p.name: _collect_calls(p.body) for p in prog.procs}

    def depth(name: str, visited: frozenset) -> int:
        if name in visited:
            return 0  # cycle guard (recursion not supported in Janus, but be safe)
        callees = call_graph.get(name, set())
        if not callees:
            return 0
        return 1 + max(depth(c, visited | {name}) for c in callees)

    return depth(prog.main_proc, frozenset())


def _get_branch_target(instr: Instr) -> Optional[str]:
    if isinstance(instr, (BRA, RBRA)):
        return instr.label
    if isinstance(instr, (BEQ, BNE)):
        return instr.label
    if isinstance(instr, BGEZ):
        return instr.label
    return None


def _remap_branch(instr: Instr, alias: Dict[str, str]) -> Instr:
    """Return instruction with branch target substituted via alias map."""
    target = _get_branch_target(instr)
    if target is None or target not in alias:
        return instr
    new_t = alias[target]
    if isinstance(instr, BRA):   return BRA(new_t)
    if isinstance(instr, RBRA):  return RBRA(new_t)
    if isinstance(instr, BEQ):   return BEQ(instr.rd, instr.rs, new_t)
    if isinstance(instr, BNE):   return BNE(instr.rd, instr.rs, new_t)
    if isinstance(instr, BGEZ):  return BGEZ(instr.rd, new_t)
    return instr


def _is_nop_instr(instr: Instr) -> bool:
    """Return True if instr is a semantic no-op (zero-constant ADDI/SUBI/XORI)."""
    return (
        (isinstance(instr, (ADDI, SUBI)) and instr.c == 0) or
        (isinstance(instr, XORI) and instr.c == 0)
    )


def remove_nops(code: List[LabeledInstr]) -> List[LabeledInstr]:
    """Remove NOP instructions (zero-constant ADDI/SUBI/XORI).

    Unlabeled NOPs: simply dropped.
    Labeled NOPs: label is forwarded to the next instruction.
      - If successor has no label: NOP's label moves to successor.
      - If successor has a label: branch targets are aliased and NOP is dropped.
      - NOP at end of program: left in place (label cannot be discarded).
    """
    alias_map: Dict[str, str] = {}   # NOP_label → successor_label
    forward_set: set = set()          # NOP_labels whose successor has no label

    for i, li in enumerate(code):
        if not (_is_nop_instr(li.instr) and li.label is not None):
            continue
        if i + 1 < len(code):
            nxt = code[i + 1]
            if nxt.label is not None:
                alias_map[li.label] = nxt.label
            else:
                forward_set.add(li.label)
        # NOP at end of code: can't remove safely, leave it

    result = []
    carry_label: Optional[str] = None

    for li in code:
        if _is_nop_instr(li.instr):
            if li.label is None:
                # Unlabeled NOP: just drop it
                continue
            # Labeled NOP: drop if we can forward its label
            if li.label in alias_map or li.label in forward_set:
                if li.label in forward_set:
                    carry_label = li.label  # attach to next instruction
                continue

        label = li.label
        if carry_label is not None:
            label = carry_label
            carry_label = None

        instr = _remap_branch(li.instr, alias_map)
        result.append(LabeledInstr(label, instr))

    return result


def program_stats(code: List[LabeledInstr]) -> Dict[str, int]:
    """Return statistics about a compiled PISA program.

    Keys: total, data_words, code_instructions, labeled_instructions,
          registers_used.
    """
    from pisa import DATA, START, FINISH
    instrs = [li.instr for li in code]
    data_count = sum(1 for i in instrs if isinstance(i, DATA))
    # START and FINISH are not real code, subtract them
    code_count = sum(1 for i in instrs
                     if not isinstance(i, (DATA, START, FINISH)))
    labels_count = sum(1 for li in code if li.label is not None)

    regs: set = set()
    for li in code:
        for attr in vars(li.instr).values():
            if isinstance(attr, str) and len(attr) > 1 and attr[0] == 'r':
                tail = attr[1:]
                if tail.isdigit() and int(tail) >= 3:
                    regs.add(attr)

    return {
        'total': len(code),
        'data_words': data_count,
        'code_instructions': code_count,
        'labeled_instructions': labels_count,
        'registers_used': len(regs),
    }


def remove_unused_labels(code: List[LabeledInstr]) -> List[LabeledInstr]:
    """Strip labels not referenced by any branch instruction.

    Unreferenced labels have no effect on execution.  The 'start' label is
    always preserved because the PISA runtime uses it to locate the entry
    point.  Procedure-related labels (name, name_top, name_bot) are kept
    naturally because they are referenced by BRA/RBRA instructions.
    """
    _ALWAYS_KEEP = frozenset({'start', 'finish'})

    referenced: set = set()
    for li in code:
        t = _get_branch_target(li.instr)
        if t:
            referenced.add(t)

    return [
        LabeledInstr(
            li.label if (li.label is None
                         or li.label in referenced
                         or li.label in _ALWAYS_KEEP)
                     else None,
            li.instr,
        )
        for li in code
    ]


def check_wf(code: List[LabeledInstr], stage: str = "") -> None:
    """Raise CodeGenError unless every instruction satisfies pisa.is_wf.

    compile_program calls this on the unoptimised and the optimised code, so
    a lowering that emits a non-invertible instruction (`XOR r r`, `ADD r r`,
    `SLTX r r s`, ...) fails at compile time instead of silently producing a
    program that is reversible only as a whole, or not at all.
    """
    bad = [(i, format_instr(li.instr)) for i, li in enumerate(code)
           if not is_wf(li.instr)]
    if bad:
        shown = ", ".join(f"#{i} {t}" for i, t in bad[:5])
        raise CodeGenError(
            f"internal: {len(bad)} instruction(s) not locally invertible"
            f"{' (' + stage + ')' if stage else ''}: {shown}")


def compile_program(prog: Program) -> List[LabeledInstr]:
    """Compile a Janus Program AST to PISA instructions."""
    cg = CodeGen()
    code = cg.gen_program(prog)
    check_wf(code, "unoptimised")
    code = peephole(code)
    code = remove_nops(code)
    code = remove_unused_labels(code)
    check_wf(code, "optimised")
    return code
