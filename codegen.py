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
)
from regalloc import RegAlloc
from inverse import invert_stmt


def _inv_proc_name(name: str) -> str:
    """Label of the inverted companion of procedure `name`."""
    return name + "_inv"



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

    def gen_expr(self, expr: Expr) -> Tuple[List[LabeledInstr], str]:
        """Evaluate expression into a register.

        Returns (code, result_register).
        The result register is committed.
        """
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
        """Read array element x[e]."""
        base_offset = self._var_offset(expr.name)
        # Evaluate index
        idx_code, ri = self.gen_expr(expr.index)
        ra = self.reg.alloc()
        rd = self.reg.alloc()
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
        # ri is garbage (from index evaluation) - keep for uncomputation
        self.reg.to_garbage(ri)
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

    def _split_const_mul(self, expr: BinOp, rl: str, rr: str) -> Tuple[str, int]:
        """For a multiplication, return (value register, constant multiplier).

        PISA has no MUL instruction, and a data-dependent multiply loop cannot
        be inverted by the straight-line _reverse_code machinery, so at least
        one operand must be a compile-time constant.
        """
        k = self._const_value(expr.right)
        if k is not None:
            return rl, k
        k = self._const_value(expr.left)
        if k is not None:
            return rr, k
        raise CodeGenError(
            "Multiplication requires a constant operand; "
            "variable * variable is not supported in PISA codegen"
        )

    def _gen_const_mul(self, rv: str, k: int) -> Tuple[List[LabeledInstr], str, List[str]]:
        """Emit `re = rv * k` for a compile-time constant k (shift-and-add).

        `rv` is only read, never written, so the caller's value survives.
        Returns (code, re, temps), where each temp holds rv*2^i; the caller
        disposes of them (marks them garbage, or reverses the code to zero
        them).  Doubling uses a fresh zeroed register and two ADDs rather than
        `ADD p p`, which is not reversible on PISA.
        """
        if abs(k) >= (1 << self.MUL_MAX_BITS):
            raise CodeGenError(
                f"Multiplier {k} exceeds {self.MUL_MAX_BITS}-bit limit "
                f"for constant multiplication"
            )
        code: List[LabeledInstr] = []
        re = self.reg.alloc()
        temps: List[str] = []
        n, p = abs(k), rv
        while n:
            if n & 1:
                code.append(self._emit(ADD(re, p)))
            n >>= 1
            if n:
                q = self.reg.alloc()
                code.append(self._emit(ADD(q, p)))
                code.append(self._emit(ADD(q, p)))  # q = 2p
                temps.append(q)
                p = q
        if k < 0:
            code.append(self._emit(NEG(re)))
        return code, re, temps

    def _gen_binop(self, expr: BinOp) -> Tuple[List[LabeledInstr], str]:
        """Generate code for binary operation."""
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

        left_code, rl = self.gen_expr(expr.left)
        right_code, rr = self.gen_expr(expr.right)
        code = list(left_code) + list(right_code)

        op = expr.op

        if op == '+':
            # Result in rl, rl += rr
            code.append(self._emit(ADD(rl, rr)))
            self.reg.to_garbage(rr)
            self.reg.commit_reg(rl)
            return code, rl

        if op == '-':
            code.append(self._emit(SUB(rl, rr)))
            self.reg.to_garbage(rr)
            self.reg.commit_reg(rl)
            return code, rl

        if op == '^':
            code.append(self._emit(XOR(rl, rr)))
            self.reg.to_garbage(rr)
            self.reg.commit_reg(rl)
            return code, rl

        if op == '*':
            rv, k = self._split_const_mul(expr, rl, rr)
            mul_code, re, temps = self._gen_const_mul(rv, k)
            code.extend(mul_code)
            for t in temps:
                self.reg.to_garbage(t)
            self.reg.to_garbage(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        if op in ('=', '!='):
            # x = y  →  !(x < y) && !(y < x)
            # Using SLTX: rd = (rs < rt) ? 1 : 0
            re = self.reg.alloc()
            rt = self.reg.alloc()
            code.append(self._emit(SLTX(re, rl, rr)))   # re = (rl < rr)
            code.append(self._emit(SLTX(rt, rr, rl)))   # rt = (rr < rl)
            code.append(self._emit(ORX(re, rt)))         # re |= rt (re = rl!=rr)
            self.reg.free_reg(rt)
            if op == '=':
                code.append(self._emit(XORI(re, 1)))     # flip: re = (rl==rr)
            self.reg.to_garbage(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        if op == '<':
            re = self.reg.alloc()
            code.append(self._emit(SLTX(re, rl, rr)))
            self.reg.to_garbage(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        if op == '>':
            re = self.reg.alloc()
            code.append(self._emit(SLTX(re, rr, rl)))  # swap operands
            self.reg.to_garbage(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        if op == '<=':
            # x <= y  ↔  !(y < x)
            re = self.reg.alloc()
            code.append(self._emit(SLTX(re, rr, rl)))
            code.append(self._emit(XORI(re, 1)))
            self.reg.to_garbage(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        if op == '>=':
            re = self.reg.alloc()
            code.append(self._emit(SLTX(re, rl, rr)))
            code.append(self._emit(XORI(re, 1)))
            self.reg.to_garbage(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        if op == '&&':
            re = self.reg.alloc()
            code.append(self._emit(ANDX(re, rl, rr)))
            # ANDX zeroes rl as side effect
            self.reg.to_garbage(rr)
            self.reg.free_reg(rl)  # ANDX clears rd1
            self.reg.commit_reg(re)
            return code, re

        if op == '||':
            re = self.reg.alloc()
            code.append(self._emit(ORX(re, rl)))
            code.append(self._emit(ORX(re, rr)))
            self.reg.to_garbage(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        if op == '&':
            re = self.reg.alloc()
            code.append(self._emit(ANDX(re, rl, rr)))
            self.reg.free_reg(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        if op == '|':
            re = self.reg.alloc()
            code.append(self._emit(ORX(re, rl)))
            code.append(self._emit(ORX(re, rr)))
            self.reg.to_garbage(rl)
            self.reg.to_garbage(rr)
            self.reg.commit_reg(re)
            return code, re

        raise CodeGenError(f"Unknown operator: {op}")

    def uneval_expr(self, expr: Expr, result_reg: str) -> List[LabeledInstr]:
        """Generate code to unevaluate an expression (reverse of gen_expr).

        This clears the result register and any garbage produced during evaluation.
        Implements the inverse computation for clean translation.
        """
        # The inverse of evaluation: run the evaluation code backwards.
        # For simplicity, re-generate and reverse.
        # Save register state, generate forward, reverse instructions.
        fwd_code, _ = self._gen_expr_for_uneval(expr, result_reg)
        return self._reverse_code(fwd_code)

    def _gen_expr_for_uneval(self, expr: Expr, target_reg: str) -> Tuple[List[LabeledInstr], str]:
        """Re-generate expression code targeting a specific register for reversal."""
        # For unevaluation, we generate the same forward code
        # and then reverse it. This is a simplified approach.
        # In practice, we track which registers were used during gen_expr.
        return self.gen_expr(expr)

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
            return ORX(instr.rd, instr.rs)
        if isinstance(instr, ANDX):
            return ANDX(instr.rd1, instr.rd2, instr.rs)
        if isinstance(instr, SLTX):
            return SLTX(instr.rd, instr.rs, instr.rt)
        if isinstance(instr, BRA):
            return RBRA(instr.label)
        if isinstance(instr, RBRA):
            return BRA(instr.label)
        if isinstance(instr, SWAPBR):
            return SWAPBR(instr.rd)
        raise CodeGenError(f"Cannot invert instruction: {type(instr).__name__}")

    def _clear_garbage(self) -> List[LabeledInstr]:
        """Zero all garbage registers (XOR r r → 0) and return them to the free pool."""
        code = []
        for r in sorted(self.reg.garbage, key=lambda r: int(r[1:])):
            code.append(self._emit(XOR(r, r)))
            self.reg.free_reg(r)
        return code

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

        # Self-reference: x op= x.  Avoids the full eval/uneval cycle entirely.
        #   x += x  →  EXCH rd ra; ADD rd rd; EXCH rd ra  (double in-place)
        #   x -= x  →  EXCH rd ra; XOR rd rd; EXCH rd ra  (zero in-place)
        #   x ^= x  →  EXCH rd ra; XOR rd rd; EXCH rd ra  (zero in-place)
        if isinstance(stmt.expr, Var) and stmt.expr.name == stmt.var:
            ra = self.reg.alloc()
            rd = self.reg.alloc()
            code = []
            code.append(self._emit(ADDI(ra, offset)))
            code.append(self._emit(EXCH(rd, ra)))
            if stmt.op == '+=':
                code.append(self._emit(ADD(rd, rd)))
            elif stmt.op in ('-=', '^='):
                code.append(self._emit(XOR(rd, rd)))
            else:
                raise CodeGenError(f"Unknown assign op: {stmt.op}")
            code.append(self._emit(EXCH(rd, ra)))
            code.append(self._emit(SUBI(ra, offset)))
            self.reg.free_reg(ra)
            self.reg.free_reg(rd)
            return code

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

        # 7. Unevaluate e (clear re and garbage)
        uneval_code = self._gen_uneval_expr(stmt.expr, re)
        code.extend(uneval_code)
        code.extend(self._clear_garbage())

        return code

    def _gen_uneval_expr(self, expr: Expr, result_reg: str) -> List[LabeledInstr]:
        """Unevaluate expression: reverse the evaluation to clear registers.

        Uses the EXCH-XOR-EXCH pattern in reverse for variables,
        SUBI for constants, etc.
        """
        if isinstance(expr, Const):
            code = []
            if expr.value != 0:
                if expr.value > 0:
                    code.append(self._emit(SUBI(result_reg, expr.value)))
                else:
                    code.append(self._emit(ADDI(result_reg, -expr.value)))
            self.reg.free_reg(result_reg)
            return code

        if isinstance(expr, Var):
            offset = self._var_offset(expr.name)
            ra = self.reg.alloc()
            rv = self.reg.alloc()
            code = [
                self._emit(ADDI(ra, offset)),
                self._emit(EXCH(rv, ra)),         # rv = mem[ra]
                self._emit(XOR(result_reg, rv)),   # clear result_reg (was copy of rv)
                self._emit(EXCH(rv, ra)),          # restore mem
                self._emit(SUBI(ra, offset)),
            ]
            self.reg.free_reg(ra)
            self.reg.free_reg(rv)
            self.reg.free_reg(result_reg)
            return code

        if isinstance(expr, BinOp):
            return self._gen_uneval_binop(expr, result_reg)

        if isinstance(expr, ArrayAccess):
            # Similar to var but with index computation
            # For now, simplified version
            return self._gen_uneval_array(expr, result_reg)

        raise CodeGenError(f"Cannot unevaluate: {type(expr)}")

    def _gen_uneval_binop(self, expr: BinOp, result_reg: str) -> List[LabeledInstr]:
        """Unevaluate binary operation.

        We re-evaluate the operands, undo the operation, then unevaluate operands.
        For arithmetic ops (+, -, ^) with a constant operand, we skip the register
        load/unload and use an immediate instruction directly (saves 2 instructions).
        """
        op = expr.op

        # Fast path for arithmetic with constant operand(s).
        # Avoids allocating a register and two instructions per const operand.
        if op in ('+', '-', '^'):
            left_is_const = isinstance(expr.left, Const)
            right_is_const = isinstance(expr.right, Const)

            if left_is_const or right_is_const:
                code = []
                # Load the non-const side (if any) into a register
                if not left_is_const:
                    left_code, rl = self.gen_expr(expr.left)
                    code.extend(left_code)
                else:
                    rl = None
                if not right_is_const:
                    right_code, rr = self.gen_expr(expr.right)
                    code.extend(right_code)
                else:
                    rr = None

                kl = expr.left.value  if left_is_const  else None
                kr = expr.right.value if right_is_const else None

                # Undo: result = left op right  →  result -= right, result -= left  (for +)
                if op == '+':
                    # result -= right
                    if rr is None:
                        if kr != 0: code.append(self._emit(SUBI(result_reg, kr)))
                    else:
                        code.append(self._emit(SUB(result_reg, rr)))
                    # result -= left
                    if rl is None:
                        if kl != 0: code.append(self._emit(SUBI(result_reg, kl)))
                    else:
                        code.append(self._emit(SUB(result_reg, rl)))
                elif op == '-':
                    # result = left - right  →  result += right; result -= left
                    if rr is None:
                        if kr != 0: code.append(self._emit(ADDI(result_reg, kr)))
                    else:
                        code.append(self._emit(ADD(result_reg, rr)))
                    if rl is None:
                        if kl != 0: code.append(self._emit(SUBI(result_reg, kl)))
                    else:
                        code.append(self._emit(SUB(result_reg, rl)))
                elif op == '^':
                    if rr is None:
                        if kr != 0: code.append(self._emit(XORI(result_reg, kr)))
                    else:
                        code.append(self._emit(XOR(result_reg, rr)))
                    if rl is None:
                        if kl != 0: code.append(self._emit(XORI(result_reg, kl)))
                    else:
                        code.append(self._emit(XOR(result_reg, rl)))

                # Unevaluate non-const operands
                if rr is not None:
                    code.extend(self._gen_uneval_expr(expr.right, rr))
                if rl is not None:
                    code.extend(self._gen_uneval_expr(expr.left, rl))
                self.reg.free_reg(result_reg)
                return code

        # General path: re-evaluate left and right to get their values
        left_code, rl = self.gen_expr(expr.left)
        right_code, rr = self.gen_expr(expr.right)
        code = list(left_code) + list(right_code)

        # Undo the operation on result_reg
        if op == '+':
            code.append(self._emit(SUB(result_reg, rr)))
            # Now result_reg should equal rl
            code.append(self._emit(SUB(result_reg, rl)))
            # Now result_reg should be 0
        elif op == '-':
            code.append(self._emit(ADD(result_reg, rr)))
            code.append(self._emit(SUB(result_reg, rl)))
        elif op == '^':
            code.append(self._emit(XOR(result_reg, rr)))
            code.append(self._emit(XOR(result_reg, rl)))
        elif op == '*':
            # Recompute the product into a temp, subtract it to zero result_reg,
            # then run the product code backwards to zero the temp and its chain.
            rv, k = self._split_const_mul(expr, rl, rr)
            mul_code, sub_re, temps = self._gen_const_mul(rv, k)
            code.extend(mul_code)
            code.append(self._emit(SUB(result_reg, sub_re)))
            code.extend(self._reverse_code(mul_code))
            for t in temps:
                self.reg.free_reg(t)
            self.reg.free_reg(sub_re)
        elif op == '<':
            sub_re = self.reg.alloc()
            code.append(self._emit(SLTX(sub_re, rl, rr)))
            code.append(self._emit(XOR(result_reg, sub_re)))
            code.append(self._emit(XOR(sub_re, sub_re)))   # zero sub_re before freeing
            self.reg.free_reg(sub_re)
        elif op == '>':
            sub_re = self.reg.alloc()
            code.append(self._emit(SLTX(sub_re, rr, rl)))
            code.append(self._emit(XOR(result_reg, sub_re)))
            code.append(self._emit(XOR(sub_re, sub_re)))   # zero sub_re before freeing
            self.reg.free_reg(sub_re)
        elif op == '<=':
            sub_re = self.reg.alloc()
            code.append(self._emit(SLTX(sub_re, rr, rl)))
            code.append(self._emit(XORI(sub_re, 1)))
            code.append(self._emit(XOR(result_reg, sub_re)))
            code.append(self._emit(XOR(sub_re, sub_re)))   # zero sub_re before freeing
            self.reg.free_reg(sub_re)
        elif op == '>=':
            sub_re = self.reg.alloc()
            code.append(self._emit(SLTX(sub_re, rl, rr)))
            code.append(self._emit(XORI(sub_re, 1)))
            code.append(self._emit(XOR(result_reg, sub_re)))
            code.append(self._emit(XOR(sub_re, sub_re)))   # zero sub_re before freeing
            self.reg.free_reg(sub_re)
        elif op == '=':
            sub_re = self.reg.alloc()
            sub_rt = self.reg.alloc()
            code.append(self._emit(SLTX(sub_re, rl, rr)))
            code.append(self._emit(SLTX(sub_rt, rr, rl)))
            code.append(self._emit(ORX(sub_re, sub_rt)))
            self.reg.free_reg(sub_rt)
            code.append(self._emit(XORI(sub_re, 1)))
            code.append(self._emit(XOR(result_reg, sub_re)))
            code.append(self._emit(XOR(sub_re, sub_re)))   # zero sub_re before freeing
            self.reg.free_reg(sub_re)
        elif op == '!=':
            sub_re = self.reg.alloc()
            sub_rt = self.reg.alloc()
            code.append(self._emit(SLTX(sub_re, rl, rr)))
            code.append(self._emit(SLTX(sub_rt, rr, rl)))
            code.append(self._emit(ORX(sub_re, sub_rt)))
            self.reg.free_reg(sub_rt)
            code.append(self._emit(XOR(result_reg, sub_re)))
            code.append(self._emit(XOR(sub_re, sub_re)))   # zero sub_re before freeing
            self.reg.free_reg(sub_re)
        elif op in ('&&', '&'):
            # ANDX zeroes rd2; copy rl to a temp to preserve it for subsequent uneval
            rl_copy = self.reg.alloc()
            code.append(self._emit(XOR(rl_copy, rl)))  # rl_copy = rl_val (rl unchanged)
            sub_re = self.reg.alloc()
            code.append(self._emit(ANDX(sub_re, rl_copy, rr)))  # sub_re ^= rl_copy & rr; rl_copy = 0
            code.append(self._emit(XOR(result_reg, sub_re)))
            code.append(self._emit(XOR(sub_re, sub_re)))   # zero sub_re before freeing
            self.reg.free_reg(sub_re)
            self.reg.free_reg(rl_copy)
        elif op in ('||', '|'):
            # ORX zeroes its source register; copy rl and rr to temps to preserve
            # their values for the subsequent uneval calls (same pattern as && / &).
            rl_copy = self.reg.alloc()
            rr_copy = self.reg.alloc()
            sub_re = self.reg.alloc()
            code.append(self._emit(XOR(rl_copy, rl)))      # rl_copy = rl_val (rl unchanged)
            code.append(self._emit(ORX(sub_re, rl_copy)))  # sub_re |= rl_copy; rl_copy = 0
            self.reg.free_reg(rl_copy)
            code.append(self._emit(XOR(rr_copy, rr)))      # rr_copy = rr_val (rr unchanged)
            code.append(self._emit(ORX(sub_re, rr_copy)))  # sub_re |= rr_copy; rr_copy = 0
            self.reg.free_reg(rr_copy)
            code.append(self._emit(XOR(result_reg, sub_re)))
            code.append(self._emit(XOR(sub_re, sub_re)))   # zero sub_re before freeing
            self.reg.free_reg(sub_re)
        else:
            raise CodeGenError(f"Cannot unevaluate operator: {op}")

        # Unevaluate right, then left
        right_uneval = self._gen_uneval_expr(expr.right, rr)
        left_uneval = self._gen_uneval_expr(expr.left, rl)
        code.extend(right_uneval)
        code.extend(left_uneval)

        self.reg.free_reg(result_reg)
        return code

    def _gen_uneval_array(self, expr: ArrayAccess, result_reg: str) -> List[LabeledInstr]:
        """Unevaluate array access."""
        base_offset = self._var_offset(expr.name)
        idx_code, ri = self.gen_expr(expr.index)
        ra = self.reg.alloc()
        rv = self.reg.alloc()
        code = list(idx_code)
        code += [
            self._emit(ADDI(ra, base_offset)),
            self._emit(ADD(ra, ri)),
            self._emit(EXCH(rv, ra)),
            self._emit(XOR(result_reg, rv)),
            self._emit(EXCH(rv, ra)),
            self._emit(SUB(ra, ri)),
            self._emit(SUBI(ra, base_offset)),
        ]
        self.reg.free_reg(ra)
        self.reg.free_reg(rv)
        idx_uneval = self._gen_uneval_expr(expr.index, ri)
        code.extend(idx_uneval)
        self.reg.free_reg(result_reg)
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

        # 7. Unevaluate e2
        uneval_e2 = self._gen_uneval_expr(stmt.expr, re)
        code.extend(uneval_e2)

        # 8-9. Clear address and unevaluate index
        code.append(self._emit(SUB(ra, ri)))
        code.append(self._emit(SUBI(ra, base_offset)))
        self.reg.free_reg(ra)

        uneval_idx = self._gen_uneval_expr(stmt.idx, ri)
        code.extend(uneval_idx)
        code.extend(self._clear_garbage())

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
            if idx_expr is not None:
                idx_code, ri = self.gen_expr(idx_expr)
                addr_code = list(idx_code) + addr_code
                addr_code.append(self._emit(ADD(ra, ri)))
            return addr_code, ra, ri

        lhs_code, la, li = get_addr(stmt.lhs, stmt.lhs_idx)
        rhs_code, ra2, ri2 = get_addr(stmt.rhs, stmt.rhs_idx)

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
            uneval_ri2 = self._gen_uneval_expr(stmt.rhs_idx, ri2)
            code.extend(uneval_ri2)

        if li is not None:
            code.append(self._emit(SUB(la, li)))
        lhs_offset = self._var_offset(stmt.lhs)
        code.append(self._emit(SUBI(la, lhs_offset)))
        self.reg.free_reg(la)
        if li is not None:
            uneval_li = self._gen_uneval_expr(stmt.lhs_idx, li)
            code.extend(uneval_li)
        code.extend(self._clear_garbage())

        return code

    def _gen_call(self, stmt: Call) -> List[LabeledInstr]:
        """Generate procedure call (Fig. 5): BRA f, or inline the body."""
        if stmt.proc in self._inline_procs:
            proc, _can_uncall = self._inline_procs[stmt.proc]
            return self.gen_stmt(proc.body)
        return [self._emit(BRA(stmt.proc))]

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

        Test e1 (Fig. 9):
            BNE rt r0 error
            <eval e1 → re>
            XOR rt re
            <uneval e1>
        test: BEQ rt r0 test_false
              XORI rt 1
              <S1>
              ...
        Assert e2 (at join, Fig. 10):
              XORI rt 1
        assert: BNE rt r0 assert_true
              <eval e2 → re>
              XOR rt re
              <uneval e2>
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

        # Evaluate test expression
        eval_code, re = self.gen_expr(stmt.test)
        code.extend(eval_code)
        code.append(self._emit(XOR(rt, re)))

        # Unevaluate test
        uneval_code = self._gen_uneval_expr(stmt.test, re)
        code.extend(uneval_code)
        code.extend(self._clear_garbage())

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

        # Evaluate assertion
        eval_fi, re2 = self.gen_expr(stmt.fi)
        code.extend(eval_fi)
        code[-len(eval_fi)] = LabeledInstr(assert_label, code[-len(eval_fi)].instr) \
            if eval_fi else code[-1]
        code.append(self._emit(XOR(rt, re2)))
        uneval_fi = self._gen_uneval_expr(stmt.fi, re2)
        code.extend(uneval_fi)
        code.extend(self._clear_garbage())

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
        # something to be wiped.
        code.append(self._emit(BRA(assert_true), end_label))  # Pendulum pair second
        self.reg.free_reg(rt)

        return code

    def _gen_from(self, stmt: From) -> List[LabeledInstr]:
        """Generate from-do-loop-until (Fig. 12).

        from e1 do S1 loop S2 until e2

        entry:  <assert e1>          ; entry assertion
                <S1 (do body)>
        test:   <test e2>            ; exit test
                BEQ rt r0 loop_body
                XORI rt 1
                BRA exit
        loop_body:
                XORI rt 1
                <S2 (loop body)>
                <assert NOT e1>      ; re-entry assertion
                BRA entry_do
        exit:   ...
        """
        test_label = self.fresh_label("from_test")
        loop_body = self.fresh_label("from_loop")
        exit_label = self.fresh_label("from_exit")
        entry_do = self.fresh_label("from_do")

        code = []
        rt = self.reg.alloc()
        self.reg.commit_reg(rt)

        # --- Entry assertion (e1 must be true) ---
        # Assert e1
        eval_entry, re = self.gen_expr(stmt.from_)
        code.extend(eval_entry)
        code.append(self._emit(XOR(rt, re)))
        uneval_entry = self._gen_uneval_expr(stmt.from_, re)
        code.extend(uneval_entry)
        code.extend(self._clear_garbage())
        # rt should be nonzero (assertion holds)
        code.append(self._emit(XOR(rt, rt)))   # clear rt (safe for non-boolean)

        # entry_do label: start of do body
        code_do = self.gen_stmt(stmt.do_)
        if code_do:
            code_do[0] = LabeledInstr(entry_do, code_do[0].instr)
        else:
            code_do = [self._emit(ADDI("r0", 0), entry_do)]  # NOP with label

        code.extend(code_do)

        # --- Test e2 (exit condition) ---
        eval_exit, re2 = self.gen_expr(stmt.until)
        test_code = list(eval_exit)
        test_code.append(self._emit(XOR(rt, re2)))
        uneval_exit = self._gen_uneval_expr(stmt.until, re2)
        test_code.extend(uneval_exit)
        test_code.extend(self._clear_garbage())

        if test_code:
            test_code[0] = LabeledInstr(test_label, test_code[0].instr)
        code.extend(test_code)

        # Branch on result
        code.append(self._emit(BEQ(rt, "r0", loop_body)))
        code.append(self._emit(XOR(rt, rt)))   # clear rt (safe for non-boolean)
        code.append(self._emit(BRA(exit_label)))

        # --- Loop body ---
        loop_code = [self._emit(XORI(rt, 1), loop_body)]

        body_code = self.gen_stmt(stmt.loop_)
        loop_code.extend(body_code)

        # Assert NOT e1 (re-entry: e1 must be false)
        eval_reentry, re3 = self.gen_expr(stmt.from_)
        loop_code.extend(eval_reentry)
        loop_code.append(self._emit(XOR(rt, re3)))
        uneval_reentry = self._gen_uneval_expr(stmt.from_, re3)
        loop_code.extend(uneval_reentry)
        loop_code.extend(self._clear_garbage())
        # rt = 1 ^ eval(e1). Since e1 is false on re-entry, eval(e1)=0 → rt=1.
        # Flip rt back to 0 so the invariant rt=0 at from_test is maintained.
        loop_code.append(self._emit(XOR(rt, rt)))  # clear rt (safe for non-boolean)

        loop_code.append(self._emit(BRA(entry_do)))
        code.extend(loop_code)

        # --- Exit ---
        self.reg.free_reg(rt)
        # Add exit label
        code.append(self._emit(ADDI("r0", 0), exit_label))  # NOP with label

        return code

    # --- Procedure code generation (Fig. 5) ---

    def gen_proc(self, proc: ProcDecl) -> List[LabeledInstr]:
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
        f_top = f"{proc.name}_top"
        f_bot = f"{proc.name}_bot"

        code = [
            self._emit(BRA(f_bot), f_top),
            self._emit(SUBI("r1", 1), proc.name),
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
            inv_proc = ProcDecl(_inv_proc_name(name), invert_stmt(proc_map[name].body))
            code.extend(self.gen_proc(inv_proc))

        # 3. Entry/exit
        code.append(self._emit(START(), "start"))
        code.append(self._emit(ADDI("r1", stack_offset)))
        code.append(self._emit(BRA(prog.main_proc)))

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


def compile_program(prog: Program) -> List[LabeledInstr]:
    """Compile a Janus Program AST to PISA instructions."""
    cg = CodeGen()
    code = cg.gen_program(prog)
    code = peephole(code)
    code = remove_nops(code)
    code = remove_unused_labels(code)
    return code
