# Reversible expression lowering

This is the specification of how `codegen.py` compiles Janus expressions
since branch `fix/reversible-expr` (2026-09).  It is written so that the Rocq
model (`rocq/Compile.v`, `CompileLoop.v`, `CompileProc.v`) can be brought in
line with it; the Rocq model still describes the previous lowering (see
"What changed" below).

## 1. Well-formedness: `pisa.is_wf`

Every instruction `codegen.py` emits must be *locally invertible*: injective
on machine states, so that the program is reversible instruction by
instruction and not merely as a whole.  `pisa.is_wf(instr)` is:

| instruction | condition | why |
|---|---|---|
| `ADD rd rs`, `SUB rd rs`, `XOR rd rs` | `rd != rs` | `XOR r r` clears r, `SUB r r` too, `ADD r r` doubles (not injective mod 2^w) |
| `ANDX rd rs rt`, `ORX rd rs rt`, `SLTX rd rs rt` | `rd ∉ {rs, rt}` | `rd ^= f(rs, rt)` is then self-inverse |
| `EXCH rd ra` | `rd != ra`, `rd != r0` | `EXCH r r` loses the register; `EXCH r0 ra` loses the memory word |
| `SWAPBR rd` | `rd != r0` | r0 would lose `br` |
| `NEG`, `ADDI`, `SUBI`, `XORI`, `BRA`, `RBRA`, `BEQ`, `BNE`, `BGEZ`, `DATA`, `START`, `FINISH` | always | |

A write to r0 by any other instruction is discarded, i.e. a no-op, and so
invertible (`ADDI r0 0` is the compiler's NOP).  `rs = rt` is allowed.

`codegen.compile_program` checks `is_wf` on the unoptimised and the optimised
code (`codegen.check_wf`) and raises `CodeGenError` otherwise;
`conftest.py` checks it again, independently, for every program the test suite
compiles, and `test_reversible_expr.py` for the corpus and the cross-check
programs.  `pisa_interp.py` still *executes* ill-formed instructions.

`ORX` / `ANDX` have Pendulum's (and phpisa's) 3-operand semantics:

    ORX  rd rs rt:   rd ^= rs | rt
    ANDX rd rs rt:   rd ^= rs & rt

(previously `ORX rd rs: rd |= rs; rs := 0` and `ANDX rd1 rd2 rs:
rd1 ^= rd2 & rs; rd2 := 0`, both non-injective; rocq/PISA.v
`orx_not_injective` / `andx_not_injective`).

## 2. The clean-expression contract

`gen_expr(e)` returns `(code, r)`:

* **Pre:** `r` and every free register hold 0.
* **Post:** `r` holds the value of `e`; every other register and all of memory
  are as before; every temporary is back in the free pool (checked at compile
  time: `gen_expr` raises if a register leaked).
* **Uncompute:** `_uneval(code, r)` = `code` run backwards (`_reverse_code`:
  reversed order, each instruction inverted: `ADD↔SUB`, `ADDI↔SUBI`, the
  others self-inverse).  It clears `r`.  Registers are **never** cleared by
  `XOR r r`, and there is no "garbage" state any more.

Uncomputing by reversal requires that the registers `code` uses are 0 and the
memory cells it reads are unchanged when the reverse runs.  Hence:

* A register that must survive the uncomputation of an operand (a result
  register) is allocated **before** that operand's code is generated, so it
  is none of its temporaries.  Temporaries are always the lowest free
  registers (`RegAlloc.alloc`).
* `x op= e` with `x` occurring in `e` (including in an index, e.g.
  `x += a[x]`) is rejected with `CodeGenError` (it is not Janus; the old
  special case `x += x` / `x -= x` / `x ^= x` emitted `ADD rd rd` /
  `XOR rd rd`).  Array updates `a[i] op= e` where `e` reads `a` are compiled
  as before (correct when the cells do not alias).

Notation below: `<e → r>` is the code of `gen_expr(e)` with result `r`;
`<e → r>⁻¹` its reverse.  Registers are listed in allocation order.

## 3. Lowering per expression form

### Constant `k`

    r                       ; fresh
    ADDI r k                ; k > 0   (SUBI r -k if k < 0; nothing if k = 0)

`e1 op e2` with both operands literal constants is folded to a constant
first (all operators, as before).

### Variable `x` (offset `o`)

    ra, r, rv               ; fresh, in this order
    ADDI ra o
    EXCH rv ra              ; rv = x, cell = 0
    XOR  r rv               ; r = x
    EXCH rv ra              ; cell = x, rv = 0
    SUBI ra o

(unchanged; `ADDI ra 0` / `SUBI ra 0` disappear in `remove_nops`).

### Array element `x[i]` (base offset `b`)   — changed

    r                       ; fresh, allocated FIRST
    <i → ri>
    ra, rv                  ; fresh
    ADDI ra b
    ADD  ra ri              ; ra = b + i
    EXCH rv ra
    XOR  r rv               ; r = x[i]
    EXCH rv ra
    SUB  ra ri
    SUBI ra b
    <i → ri>⁻¹              ; index uncomputed at once (was: kept as garbage)

### `e1 + e2`, `e1 - e2`, `e1 ^ e2` — in place   — changed

The operator is injective in the left operand, so the left register keeps
the result and only the right operand is uncomputed (Axelsen, CC 2011):

    <e1 → rl>
    <e2 → rr>
    ADD rl rr               ; resp. SUB rl rr, XOR rl rr   (rl != rr)
    <e2 → rr>⁻¹
                            ; result in rl

If `e2` is a compile-time constant `k` (`CodeGen._const_value`, which also
sees `0 - 3` etc.), no register is used:

    <e1 → rl>
    ADDI rl k               ; +: ADDI k / SUBI -k;  -: SUBI k / ADDI -k;  ^: XORI k
                            ; nothing when k = 0

### `e * k`, `k * e` (k a compile-time constant, |k| < 2^16)   — changed

With `n = |k|`, `p0 = rv`, `p(j+1) = 2·p(j)`:

    re                      ; fresh, allocated FIRST
    <e → rv>
    q1: ADD q1 rv ; ADD q1 rv       ; q1 = 2v    (q fresh; ADD q q would not be wf)
    q2: ADD q2 q1 ; ADD q2 q1       ; q2 = 4v    ... up to bit_length(n) - 1
    ADD re p(j)             ; for every bit j set in n, ascending
    NEG re                  ; if k < 0
    (doubling chain)⁻¹      ; SUB q q' pairs in reverse: q's back to 0
    <e → rv>⁻¹

`k = 0` emits nothing (re stays 0).  The right operand is tried first as the
constant.  `var * var` raises `CodeGenError`.

### Comparisons and `&`, `|`, `&&`, `||` — fresh result register   — changed

    re                      ; fresh, allocated FIRST
    <e1 → rl>
    <e2 → rr>
    <combine re rl rr>      ; re ^= f(rl, rr); re, rl, rr pairwise distinct
    <e2 → rr>⁻¹
    <e1 → rl>⁻¹

| op | combine (re = 0 before) | value |
|---|---|---|
| `a < b`  | `SLTX re rl rr` | a < b |
| `a > b`  | `SLTX re rr rl` | b < a |
| `a <= b` | `SLTX re rr rl ; XORI re 1` | ¬(b < a) |
| `a >= b` | `SLTX re rl rr ; XORI re 1` | ¬(a < b) |
| `a != b` | `SLTX re rl rr ; SLTX re rr rl` | (a<b) ⊕ (b<a) = (a<b) ∨ (b<a): never both 1 |
| `a = b`  | `SLTX re rl rr ; SLTX re rr rl ; XORI re 1` | ¬(a != b) |
| `a & b`  | `ANDX re rl rr` | bitwise and |
| `a \| b` | `ORX re rl rr` | bitwise or |
| `a && b` | `ANDX re rl rr` | on 0/1 operands (below) |
| `a \|\| b` | `ORX re rl rr` | on 0/1 operands (below) |

The previous `!=` was `SLTX re rl rr; SLTX rt rr rl; ORX re rt` with the
clearing ORX; the extra register `rt` is gone.

**`&&` / `||` operands** go through `_logical_operands` first: an operand that
is already 0/1 (a comparison, `&&`/`||`, or the literal 0/1) is kept, any other
operand `e` becomes `e != 0` (below).  Janus truth is "nonzero", results are
0/1, so e.g. `2 && 4 = 1`, `(0 - 1) || 0 = 1`.

### `e != 0` — the nonzero flag (`_gen_nonzero`, also used by `_as_flag`)

    rf                      ; fresh, allocated FIRST
    <e → rv>
    SLTX rf rv r0           ; rf ^= (v < 0)
    SLTX rf r0 rv           ; rf ^= (0 < v)       (exclusive, so rf = (v != 0))
    <e → rv>⁻¹

(The uncomputation now runs these two SLTX in the opposite order; previously
the same two lines were re-emitted in the same order.)

## 4. Statement contexts (layouts unchanged)

Only the expression code inside the statements changed; the statement
layouts (`_gen_if`, `_gen_from`, call/uncall, procedure wrappers) are as
before.  Every uneval is the reverse of the *same* code object:

    x op= e      <e → re> ; ra, rd ; ADDI ra o ; EXCH rd ra ; OP rd re ;
                 EXCH rd ra ; SUBI ra o ; <e → re>⁻¹
                 (x op= k for a literal k: ADDI ra o ; EXCH rd ra ;
                  ADDI/SUBI/XORI rd k ; EXCH rd ra ; SUBI ra o — unchanged)
    x[i] op= e   <i → ri> ; ra ; ADDI ra b ; ADD ra ri ; <e → re> ; rd ;
                 EXCH rd ra ; OP rd re ; EXCH rd ra ; <e → re>⁻¹ ;
                 SUB ra ri ; SUBI ra b ; <i → ri>⁻¹
    x[i] <=> y[j]  as before, with each index uncomputed by <i → ri>⁻¹
    rt ^= e      (if / from tests and assertions, `_gen_flag_xor`, e already
                 normalised by `_as_flag`):  <e → re> ; XOR rt re ; <e → re>⁻¹

The trailing `XOR r r` block (`_clear_garbage`) that used to end every
assignment, swap and flag update no longer exists.

## 5. Example (unoptimised), `int a; int b; int r;  r += a < b`

    ADDI r4 0 ; EXCH r6 r4 ; XOR r5 r6 ; EXCH r6 r4 ; SUBI r4 0   ; <a → r5>   (re = r3 first)
    ADDI r4 1 ; EXCH r7 r4 ; XOR r6 r7 ; EXCH r7 r4 ; SUBI r4 1   ; <b → r6>
    SLTX r3 r5 r6                                                 ; r3 = a < b
    ADDI r4 1 ; EXCH r7 r4 ; XOR r6 r7 ; EXCH r7 r4 ; SUBI r4 1   ; <b → r6>⁻¹
    ADDI r4 0 ; EXCH r6 r4 ; XOR r5 r6 ; EXCH r6 r4 ; SUBI r4 0   ; <a → r5>⁻¹
    ADDI r4 2 ; EXCH r5 r4 ; ADD r5 r3 ; EXCH r5 r4 ; SUBI r4 2   ; r += r3
    (the first five lines' reverse, i.e. the whole <a < b → r3> backwards)

## 6. What changed relative to the Rocq model

`rocq/Compile.v` (`gen_expr` / `ungen_expr`, `cmp_fwd`) still models the old
lowering; `tools/rocq_*` replay its legacy `IOrx` / `IAndx` through
`tools/rocq_legacy.py`.  Differences a model update has to take in:

1. `IOrx`/`IAndx` → 3-operand XOR-accumulating forms (`PISA.v`).
2. Comparisons, `&`, `|`, `&&`, `||`: operands uncomputed right after the
   combine (no `XOR r r`); `!=`/`=` without the extra register and ORX.
3. `ungen_expr e = invert_code (gen_expr e)` for **every** expression (not
   only arithmetic).  In particular the nonzero test's uneval is the reversed
   SLTX pair.
4. `+ - ^` with a constant right operand use an immediate.
5. `*` (not in the Rocq fragment) and array reads (index uncomputed at once).
6. Registers: the result register of a combine / multiplication / nonzero /
   array read is allocated before its operands — the same shape as Compile.v's
   target-register-first scheme (`gen_expr e rt` with operands above `rt`).
   For `+ - ^` the result is the left operand's register.  Register numbers
   are still `RegAlloc`'s lowest-free choice, so e.g. the address register of
   a variable load sits between the target and the value register.

Cost: on the random expressions of `.claude/handoff/stress.py` (1000 per
depth), `RegAllocError` drops from 0 / 36.0 / 76.8 / 82.7 % (depth 2 / 3 / 4 / 5)
to 0 % at every depth, the peak register count from ≤ 29 to ≤ 15, while the
code of the programs both versions compile grows by a median factor of
1.40 (depth 2) / 1.76 (depth 3) (max 2.09 / 3.16): every nesting level
emits its operands' code twice (compute and uncompute), so code size grows
exponentially with the nesting depth, where the old re-evaluating uneval
kept every intermediate in a register instead.  On the difftest corpus
(shallow expressions, mostly loop tests like `i = 3`) the optimised code got
slightly *smaller*: 2597 → 2525 instructions over the 30 programs both
versions compile (s8-compare-negative 207 → 177, loop 87 → 78; j1-fib
138 → 139 is the only growth), because the old uneval re-evaluated both
operands of a comparison and then cleared them.
