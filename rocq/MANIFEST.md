# Machine-checked semantic preservation for Janus → PISA

Rocq Prover 9.1.1. Build with `make` (regenerate with `rocq makefile -f _CoqProject -o Makefile`).
No `Admitted`, no `admit`, no local `Axiom`.

This directory answers the question "is the translation this repository implements
actually correct?" for the **straight-line fragment** of Janus plus **`if`** and
**`from/loop/until`** (PISACtl.v / CompileIf.v / CompileLoop.v: a PC-based machine
with the Pendulum paired-branch mechanism, and the `_gen_if` and `_gen_from`
layouts of `codegen.py` **as of main a1088e2** (PR #4 test normalisation, PR #6
assertion checks) proved correct on it — see "Control flow" below; violated
`fi` / entry / re-entry assertions are proved to reach `finish` with a dirty
flag — see "Loops"). It is the PISA counterpart of the "whole-translator
semantic preservation" that `RevLowering.v` in the PyJanus development
explicitly leaves open.

## Modules

| File | Contents |
|---|---|
| PISA.v | straight-line machine: registers, memory, `step`/`run`, local inverses (incl. `SLTX`) |
| Src.v | source fragment `Skip`/`Assign`/`Swap`/`Seq`, `exec`, `invert` |
| Compile.v | the straight-line compiler and `compile_spec` |
| Opt.v | `peephole` / `remove_nops` preserve `run` |
| LOpt.v | first labeled-code model (direct branches); `remove_unused_labels`, label forwarding |
| PISACtl.v | **control-flow machine**: labeled program, pc, `br`, paired branches, `cstep`/`steps`/`exec_fuel` |
| CompileIf.v | **`If`**: base-parametrised body compiler, `_as_flag` (`flag_block`), `exec_c`, `compile_c`, `compile_c_spec` |
| CompileLoop.v | **`from/loop/until`**: `lstmt` (subsumes `cstmt`), `exec_l`/`lp_l`, `compile_l` = `_gen_if`/`_gen_from`'s layouts, `compile_l_spec`, the three violation theorems |
| Test.v | executable checks and `Print Assumptions` |
| Extract.v | OCaml extraction of the straight-line compiler (driven by `driver.ml`) |

## What is proved

| Theorem | File | Statement |
|---|---|---|
| `step_invert` | PISA.v | every well-formed instruction is undone by `invert_instr` |
| `run_invert_code` | PISA.v | `run (invert_code c) (run c s) = s` for well-formed `c` |
| `exec_rev` | Src.v | `exec st a b → exec (invert st) b a` (the fragment is reversible) |
| `wf_compile` | Compile.v | every instruction the compiler emits has distinct operand registers |
| `gen_expr_spec` | Compile.v | expression code is correct **and clean** (see below) |
| **`compile_spec`** | Compile.v | **semantic preservation** — see below |
| `compile_reversible` | Compile.v | `run (invert_code (compile st)) (run (compile st) s) = s` |
| `peephole_run`, `remove_nops_run`, `optimize_run` | Opt.v | the optimizer passes preserve `run`, for all straight-line code |
| `cancels_undo` | Opt.v | a cancelling pair is exactly a well-formed instruction followed by its inverse |
| `strip_exec` | LOpt.v | `remove_unused_labels` preserves execution, on a PC-based labeled-code machine (axiom-free) |
| `delete_cancelling_pair_fwd` | LOpt.v | deleting a cancelling pair with label forwarding preserves every terminating run (forward simulation with a pc map) |
| `steps_ops` | PISACtl.v | a straight-line segment of the labeled machine behaves like `PISA.run` |
| `compile_at_spec`, `compile_at_scratch` | CompileIf.v | the straight-line compiler with a scratch *base* is correct, and at base `scratch` it is literally `compile` |
| `nz_block_spec`, `flag_block_spec` | CompileIf.v | `rf ^= (e != 0)` via two `SLTX`, and `rt ^= _as_flag(e)`, as exact state equations (value `truth (eval σ e)`) |
| `wf_flag_block` | CompileIf.v | every instruction of the flag code is well-formed (no `XOR r r`) |
| `exec_c_rev` | CompileIf.v | the source with `If` is reversible (`invert_c` swaps test and assertion) |
| `compile_c_labels` | CompileIf.v | every label the compiler emits lies in `[n, n')` (freshness) |
| **`compile_c_spec`** | CompileIf.v | **semantic preservation + cleanliness for `If`** on the paired-branch machine, for a fragment embedded anywhere — see "Control flow" |
| `compile_c_program` | CompileIf.v | closed corollary: the fuel executor halts at the end with the right memory and the same registers |
| `if_true_violation`, `if_false_violation` | CompileIf.v | (layout level) a violated `fi` reaches `finish` with the flag at 1 |
| `exec_l_rev` | CompileLoop.v | the source with `If` and loops is reversible (the `Janus.exec_rev` proof, via `opn_l`) |
| `steps_relabel` | CompileLoop.v | putting a fresh label on an unlabeled data line preserves every run (axiom-free) |
| `compile_l_labels`, `compile_l_head` | CompileLoop.v | labels lie in `[n, n')`; compiled code is empty or starts with an unlabeled data line |
| **`compile_l_spec`** | CompileLoop.v | **semantic preservation + cleanliness for loops and `If`**, the exact `_gen_if`/`_gen_from` layouts of main — see "Loops" |
| `compile_l_program` | CompileLoop.v | closed program with a `finish:` line: the fuel executor halts at its end with the right memory and the same registers |
| **`compile_l_if_violation`** | CompileLoop.v | a violated `fi` assertion (either path) jumps to `finish` with the flag = 1 |
| **`compile_l_loop_entry_violation`** | CompileLoop.v | a false entry assertion jumps to `finish` with the flag = 1, memory untouched |
| **`compile_l_loop_reentry_violation`** | CompileLoop.v | a true re-entry assertion after the first round jumps to `finish` with the flag = 1 |
| `compile_l_lift` | CompileLoop.v | on `If`-only programs `compile_l` *is* `compile_c` |
| `entry_violation_no_exec`, `reentry_violation_no_exec`, `if_violation_no_exec` | CompileLoop.v | the three example programs Janus rejects have no execution in `exec_l`; `ex_*_violation_detected` show the code ends with r3 = 1 |

### The main theorem

```coq
Theorem compile_spec : forall st σ σ' m,
  exec st σ σ' -> wf_stmt st ->
  models m σ -> clean_above scratch m ->
  models (run (compile st) m) σ' /\ regs (run (compile st) m) = regs m.
```

`models m σ` says memory cell `Z.of_nat x` holds `σ x` (the layout `codegen.py`
emits as `DATA` words). The conjunction states two things at once:

1. **Semantic preservation** — if the source statement takes store `σ` to `σ'`,
   the compiled code takes a memory representing `σ` to one representing `σ'`.
2. **Cleanliness** — the register file afterwards is *equal* to the one before.
   No scratch register is left dirty. This is the "clean" in Axelsen's clean
   translation, and it is what makes `compile_reversible` follow.

`gen_expr_spec` is the same idea for expressions, as an exact state equation:

```coq
run (gen_expr e rt) s = mkState (rupd rt (eval σ e) (regs s)) (mem s)
```

— the target register gains the value, *everything else is untouched*, including
memory. The proof of the `Bin` case is where clean translation actually happens:
the right operand's code is run, used, and then cancelled by `run_invert_code`.

## Scope and side conditions

- **Fragment**: `Skip`, `x op= e`, `x <=> y`, `S1; S2`, with `op ∈ {+=, -=, ^=}`
  and expressions over `+`, `-`, `^`. Source definitions are kept identical in
  shape to `Janus.v` of the PyJanus development so results transfer.
- **`occurs x e = false`** on assignment — carried as a premise of `E_Assign`,
  exactly as in `Janus.v`. It is what makes unevaluation after the store sound.
- **`x <> y`** on swap (`wf_stmt`). Semantically `sw s x x = s`, but every
  reversible lowering of a swap destroys the cell when the operands alias. The
  same restriction PyJanus and vjanus impose; `RevLowering.v` proves the XOR-triple
  version of this collapse.
- **Registers**: `r0`–`r2` reserved, scratch from `r3`, matching `regalloc.py`.
  The model has an unbounded register file, so register *exhaustion* (the
  `RegAllocError` of `regalloc.py`) is out of scope.

## Control flow: the machine model and `If` (PISACtl.v, CompileIf.v)

### The machine (PISACtl.v)

`cstate = {cpc : nat; cbr : Z; cst : PISA.state}` over a labeled program
`lprog = list (option label * cinstr)`, with `cinstr = COp i | CBra l | CRbra l
| CBeq rd rs l | CBne rd rs l | CBgez rd l | CSwapbr rd`. `cstep : lprog -> cstate
-> option cstate` is a function (the machine is deterministic), `steps` its
reflexive–transitive closure, `exec_fuel` the fuel executor
(`steps_exec_fuel` ties them). Data instructions delegate to `PISA.step`.

The semantics is that of `pisa_interp.py`, checked line by line:

| `pisa_interp.py` | model |
|---|---|
| `_detect_paired_branches`: A is paired iff `code[A]` is BRA/RBRA → B and `code[B]` branches back to A (unconditionally, or conditionally via its label) | `paired p a`, computed from the same static data |
| paired BRA: `br += t - pc; pc += br` if `br ≠ 0` else `pc += 1` | `bra_step`, `paired = true` branch |
| unpaired BRA/RBRA: `pc = t` | `bra_step`, `paired = false` branch |
| BEQ/BNE/BGEZ with `br = 0`: direct conditional jump | `cond_step`, `br =? 0` branch |
| … with `br ≠ 0`: taken → `br += t - pc`; `br = 0` → `pc = t + 1`, else `pc += br`; not taken → `pc += br` | `cond_step`, other branch |
| RBRA executed exactly like BRA (the direction bit is tracked but never used) | `CRbra` shares `bra_step` |
| SWAPBR: exchange register and `br`, `pc += 1` | `CSwapbr` |
| `pc = pc + br` negative → `PC out of range` | `jump` returns `None` (stuck) |
| `r0` reads 0 | **not modelled** (PISA.v's register file is uniform); the theorems assume `regs ms 0 = 0`, which compiled code preserves |
| `finish: FINISH` as the target of `BNE rt r0 finish` | a label `fin` the enclosing program defines; `with_finish` appends it as a NOP line and the fuel executor halts by falling off the end |
| garbage check at `FINISH` | stated, not executed: the violation theorems conclude "at `fin` with the flag register = 1" |
| software call stack, procedure detection (`f_top`/`f_bot`), `START`/`DATA` | **not modelled** — milestone 2 |
| direction bit / reverse execution | **not modelled** — `pisa_interp.py` does not implement it either |

So the model covers every control instruction the interpreter executes inside a
procedure body, including how a branch landing on its partner is consumed
(`cstep_bra_paired_cancel`, `cstep_beq_cancel`). It was validated by running
the verified compiler's output for three programs (then path, else path,
nested) on `pisa_interp.py`: memory, `br = 0` and clean registers agree with
`exec_fuel` (`ex_then`, `ex_else`, `ex_nested` in CompileIf.v).

### The source and the compiler (CompileIf.v)

```coq
Inductive cstmt := CBase (s : stmt) | CSeq (a b : cstmt)
                 | CIf (e1 : expr) (a b : cstmt) (e2 : expr).
| EC_IfTrue  : eval σ e1 <> 0 -> exec_c a σ σ' -> eval σ' e2 <> 0 -> exec_c (CIf e1 a b e2) σ σ'
| EC_IfFalse : eval σ e1 =  0 -> exec_c b σ σ' -> eval σ' e2 =  0 -> exec_c (CIf e1 a b e2) σ σ'
```

These are `Janus.exec`'s rules (true = nonzero). Until 2026-09-25 they read
`= 1` / `= 0`, because `_gen_if` XOR-ed raw test values into the flag; PR #4's
normalisation removed the reason.

**Test normalisation (`_as_flag`).** `flag_block e rt` is
`_gen_flag_xor(rt, _as_flag(e))`. Src.expr has no comparisons or logical
operators, so `_as_flag` keeps exactly the constants 0 and 1
(`is_bool_const`; code `xor_block`: evaluate into `S rt`, `XOR rt (S rt)`,
unevaluate) and turns everything else into `e != 0`, which `_gen_nonzero` /
`_nonzero_into` compile as

```
nz_block e rf = <eval e -> S rf> ; SLTX rf (S rf) r0 ; SLTX rf r0 (S rf) ; <uneval e>
flag_block e rt = nz_block e (S rt) ; XOR rt (S rt) ; nz_block e (S rt)     (e not 0/1)
```

(`v < 0` and `0 < v` are exclusive, so the two SLTX bits XOR to `v ≠ 0`; the
second `nz_block` clears `rf` again.) `SLTX` was added to PISA.v's `instr`
(`rd ^= (rs < rt)`, self-inverse, well-formed when `rd ∉ {rs, rt}`;
`step_invert` extended). `flag_block_spec`:

```coq
run (flag_block e rt) s = mkState (rupd rt (Z.lxor (regs s rt) (truth (eval σ e))) (regs s)) (mem s)
```

with `truth v = if v =? 0 then 0 else 1`, under `models s σ`, `clean_above (S rt) s`,
`regs s 0 = 0`, `rt <> 0`.

`compile_c fin st b n : lprog * label` compiles with scratch base `b`, labels
from `n`, and `fin` the label of `finish`, returning the next free label. `If`
is `if_code`, the exact `_gen_if` layout of main:

```
          <rt ^= _as_flag(e1)>
test:     BEQ rt r0 false ; XORI rt 1 ; <S1> ; XORI rt 1
assert:   <rt ^= _as_flag(e2)>
true:     BRA end
false:    BRA test ; <S2> ; BRA assert
end:      BRA true
          BNE rt r0 finish
```

with `rt = b`, the five labels `n..n+4` in `_gen_if`'s allocation order, bodies
at base `S b` (`_gen_if` allocates the flag first, so bodies use `r4` upwards —
which is why Compile.v's fixed-base compiler is generalised to `compile_at b`,
`compile_at scratch = compile`). The compiler never defines `fin`.

### The theorem

```coq
Theorem compile_c_spec : forall fin st σ σ' b n p n' ms pre post,
  exec_c st σ σ' -> wf_cstmt st ->
  compile_c fin st b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  exists ms',
    steps (pre ++ p ++ post) (mkC (length pre) 0 ms) (mkC (length pre + length p) 0 ms')
    /\ models ms' σ' /\ regs ms' = regs ms.
```

Started at its first line with `br = 0`, the code reaches the line after its
last with `br = 0`, memory representing `σ'` and the *same* register file.
Nothing is assumed about `fin`: on a valid execution the flag is 0 at
`BNE rt r0 finish`, the branch falls through and its label is never looked up.
The `pre`/`post` quantification is what makes it compose: bodies are such
fragments (nested `If` works), and so is the whole statement inside a larger
program whose labels avoid `[n, n')` (`compile_c_labels`). `compile_c_program`
specialises to `pre = post = []`, `b = scratch`, `n = 0` and the fuel executor.

Section `IfLayout` also proves the violation side at the layout level
(`if_true_violation`, `if_false_violation`): when the exit assertion
disagrees with the path, the flag is `1` at the `BNE`, which jumps to `fin`.
The statement-level theorem is `compile_l_if_violation` in CompileLoop.v.

Side conditions, all deliberate:

- **`b <> 0` and `regs ms 0 = 0`** stand for the hard-wired zero register that
  `BEQ rt r0`, `BNE rt r0` and the `SLTX … r0` of `e != 0` read.
- **Bodies** are straight-line statements or nested `If`s (`wf_cstmt` is
  `wf_stmt` on the leaves). Loops are in CompileLoop.v — see "Loops" below.
- **Expressions** are Src.expr (`+ - ^`); so the only tests `_as_flag` leaves
  alone are the constants 0/1. Comparisons and `&&`/`||` (which it also leaves
  alone, and which are 0/1-valued) are milestone 3.

`Print Assumptions` (recorded at build time at the end of CompileIf.v):

```
compile_c_spec, compile_c_program, flag_block_spec : functional_extensionality_dep
exec_c_rev                                         : functional_extensionality_dep   (via Src.exec_rev)
compile_at_scratch                                 : Closed under the global context
```

### Loops: `from e1 do S1 loop S2 until e2` (CompileLoop.v)

**Source.** `lstmt = LBase | LSeq | LIf | LLoop e1 a c e2`, with `exec_l` / `lp_l`
mutually inductive and shaped exactly like `Janus.v`'s `E_Loop` / `L_one` /
`L_more` (true = nonzero; the `= 1` restriction of the previous version is gone):

```coq
| EL_Loop : eval σ e1 <> 0 -> lp_l e1 a c e2 σ σ' -> exec_l (LLoop e1 a c e2) σ σ'
| LP_One  : exec_l a σ σ' -> eval σ' e2 <> 0 -> lp_l e1 a c e2 σ σ'
| LP_More : exec_l a σ σ1 -> eval σ1 e2 = 0 -> exec_l c σ1 σ2 -> eval σ2 e1 = 0 ->
            lp_l e1 a c e2 σ2 σ' -> lp_l e1 a c e2 σ σ'
```

`lift : cstmt -> lstmt` embeds the `If` language (`exec_c_lift`), and
`compile_l_lift` shows the compiler is unchanged on it, so nothing of
CompileIf.v is lost; its `If` layout lemmas are reused verbatim.

**Layout.** `loop_code fin b n e1 e2 pa pb` is `_gen_from` of main line for
line (labels `from_test` = n, `from_loop` = n+1, `from_exit` = n+2,
`from_do` = n+3 in its allocation order, bodies at base `S b` with labels
from n+4):

```
          <rt ^= _as_flag(e1)> ; XORI rt 1 ; BNE rt r0 finish     -- entry assertion
do:       <S1>                 (label on S1's first line; a labeled ADDI r0 0 if S1 is empty)
test:     <rt ^= _as_flag(e2)> ; BEQ rt r0 loop ; XORI rt 1 ; BRA exit
loop:     XORI rt 1 ; <S2> ; <rt ^= _as_flag(e1)> ; XORI rt 1 ; BNE rt r0 finish ; BRA do
exit:     ADDI r0 0
```

No branch here is paired, so only direct jumps are exercised. `_gen_from`
*overwrites* the label of S1's first line; `compile_l_head` shows that line
never carries one, and `steps_relabel` transfers S1's run across the added
label.

**Theorem.**

```coq
Theorem compile_l_spec : forall fin st σ σ', exec_l st σ σ' ->
  forall b n p n' ms pre post,
  wf_lstmt st -> compile_l fin st b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  exists ms',
    steps (pre ++ p ++ post) (mkC (length pre) 0 ms) (mkC (length pre + length p) 0 ms')
    /\ models ms' σ' /\ regs ms' = regs ms.
```

The proof is a mutual induction on `exec_l`/`lp_l` (`exec_l_mut`); the
invariant at `do` is "`br = 0`, the same register file (so `rt = 0`), memory
representing the current store", and each round is `do_steps` (S1) ·
`iter_steps` (test fails, `rt := 1`) · `s2_steps` · `back_steps` (re-entry
assertion false: `rt := 1 xor 0`, `XORI` back to 0, `BNE` falls through,
`BRA do`), the last round `do_steps` · `exit_steps`; the entry is
`entry_steps` (`rt := 1`, `XORI` to 0, `BNE` falls through).

The previous version (PR #5) was parametrised by the flag clear `clr` and
instantiated at `XOR rt rt`; that parameter, `compile_lg`/`compile_lg_spec`,
`clr_ok`, `compile_xori_spec` and the `clr_*` lemmas are gone, since main's
`_gen_from` no longer clears the flag.

`compile_l_program` closes it: `compile_l fin_label st scratch 1` followed by
the `finish:` line (`with_finish`, `fin_label = 0`) runs under the fuel
executor to its end with memory representing `σ'` and the same registers.

**Violated assertions (machine-checked).** The source has no execution when an
assertion fails, so `compile_l_spec` is silent there. Three theorems cover it;
each concludes that the code reaches `fin` (wherever the enclosing program
defines it: `find_label fin (pre ++ p ++ post) = Some f`) with `br = 0` and the
register file `rupd b 1 (regs ms)` — flag 1, everything else unchanged, i.e.
what `pisa_interp.py` rejects as garbage at FINISH:

```coq
Theorem compile_l_if_violation : forall fin e1 a c e2 σ σ' b n p n' ms pre post f,
  wf_lstmt a -> wf_lstmt c ->
  (eval σ e1 <> 0 /\ exec_l a σ σ' /\ eval σ' e2 = 0) \/
  (eval σ e1 = 0 /\ exec_l c σ σ' /\ eval σ' e2 <> 0) ->
  compile_l fin (LIf e1 a c e2) b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  find_label fin (pre ++ p ++ post) = Some f ->
  exists M', steps (pre ++ p ++ post) (mkC (length pre) 0 ms)
                   (mkC f 0 (mkState (rupd b 1 (regs ms)) M'))
             /\ models (mkState (regs ms) M') σ'.

Theorem compile_l_loop_entry_violation : forall fin e1 a c e2 σ b n p n' ms pre post f,
  compile_l fin (LLoop e1 a c e2) b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  eval σ e1 = 0 -> (* label hypotheses on pre/post as above *) ... ->
  find_label fin (pre ++ p ++ post) = Some f ->
  steps (pre ++ p ++ post) (mkC (length pre) 0 ms)
        (mkC f 0 (mkState (rupd b 1 (regs ms)) (mem ms))).

Theorem compile_l_loop_reentry_violation :
  forall fin e1 a c e2 σ σ1 σ2 b n p n' ms pre post f,
  wf_lstmt a -> wf_lstmt c ->
  eval σ e1 <> 0 -> exec_l a σ σ1 -> eval σ1 e2 = 0 ->
  exec_l c σ1 σ2 -> eval σ2 e1 <> 0 ->
  compile_l fin (LLoop e1 a c e2) b n = (p, n') -> (* ... as above ... *) ->
  find_label fin (pre ++ p ++ post) = Some f ->
  exists M2, steps (pre ++ p ++ post) (mkC (length pre) 0 ms)
                   (mkC f 0 (mkState (rupd b 1 (regs ms)) M2))
             /\ models (mkState (regs ms) M2) σ2.
```

The re-entry theorem covers a violation after the *first* round; a violation
after round k > 1 is the same argument composed with k − 1 `loop_round`s,
not stated.

The counterexamples of PR #5 are replaced accordingly:

| program | Janus (PyJanus) | `exec_l` | before (PR #5, `XOR rt rt`) | now (model and `pisa_interp.py`) |
|---|---|---|---|---|
| `x2 += 1; from x0 do c += 1 loop x1<=>x2 x0<=>x1 until x2` (`prog_v1`) | "Assertion failed: should be true" | no execution (`entry_violation_no_exec`) | ran to the end clean, `c = 1` (`ex_entry_violation_accepted`, removed) | jumps to `finish`, `c = 0`, r3 = 1 (`ex_entry_violation_detected`) |
| `x0 += 1; from x0 do c += 1 loop x2 += 1 until x2` (`prog_v2`) | "Assertion failed: should be false" | no execution (`reentry_violation_no_exec`) | ran to the end clean, `c = 2` (`ex_reentry_violation_accepted`, removed) | jumps to `finish` after one round, `c = 1`, r3 = 1 (`ex_reentry_violation_detected`) |
| `x2 += 5; if x2 then c += 1 else skip fi x0` (`prog_if_v`) | "Assertion failed" | no execution (`if_violation_no_exec`) | (left 1 in the flag) | jumps to `finish`, r3 = 1 (`ex_if_violation_detected`) |

`ex_xori_same`, `ex_xori_entry_violation_still_clean` and
`ex_xori_reentry_violation_dirty` (about the hypothetical `XORI` clear) are
removed as obsolete; `ex_violation_dirty` of CompileIf.v became
`ex_violation_detected` (its old assertion `y ^ 3 = 2` is *true* under
normalisation, so the example now uses `y ^ 1`). `ex_violations_branch_to_finish`
checks that the three violating programs get stuck at the `BNE` when no
`finish` line exists, and the valid ones do not.

**Cross-check.** `tools/rocq_loop_crosscheck.py` asks Rocq (`vm_compute`) for
`compile_l fin_label st scratch 1`'s code and for `run_l` on the eleven programs
of CompileLoop.v, runs that code on `pisa_interp.py` (between `start:` and
`finish:`), compiles the same Janus source with `codegen.py` and runs it, and
compares the control skeleton — labels, `BRA`/`BEQ`/`BNE`, `XORI`, `XOR r r`
on a flag register, the `SLTX` pairs (flag register and `r0` side), `ADDI r0 0` —
of `codegen.py`'s unoptimised output with the Rocq layout, up to label renaming.
**11/11 agree** (store, registers r3–r8, `br`, skeleton) against main a1088e2,
including four programs with non-0/1 tests (`prog_if5`: test 5, assertion 10;
`prog_if_else`: assertion `x0 + x1 - x1`; `prog_loop3`: token 3 through
entry, exit test and re-entry; `prog_const`: the constants 0/1 path) and the
three violating ones (both sides end at `finish` with r3 = 1; the interpreter's
garbage check is switched off to compare the dirty file). Straight-line pieces
still differ at the instruction level (`codegen.py` emits `ADDI` for constant
right-hand sides and clears binary-operator garbage with `XOR r r`;
Compile.v evaluates generally and unevaluates operands), which is why the
skeleton, not the full listing, is compared. Not in CI: it needs `rocq` and
the built `.vo` files. `ROCQ_DIR` overrides the development it reads (used for
the mutation runs below).

`Print Assumptions` (at the end of CompileLoop.v):

```
compile_l_spec, compile_l_program, exec_l_rev,
compile_l_if_violation, compile_l_loop_entry_violation,
compile_l_loop_reentry_violation                        : functional_extensionality_dep
steps_relabel, compile_l_lift                           : Closed under the global context
```

**Mutation tests** (2026-09-25; each applied, rebuilt, restored — sha256 of
CompileIf.v/CompileLoop.v checked after restoring). "Proof" is the Rocq build
of the repository; "cross-check" is the script run on a scratch copy with the
same mutation and every proof admitted, so that it measures the script alone.

| mutation | proof | cross-check |
|---|---|---|
| drop the `!= 0` wrapping (`flag_block := xor_block` always) | fails in `flag_block_spec` (false for non-0/1 values) | 10/11 DIFF (only `prog_const` agrees, as it should) |
| `BNE rt r0 finish` → `BEQ rt r0 finish` in the loop (all 12 occurrences, layout lemmas kept consistent) | fails in `entry_steps` (a valid run would jump to `finish`) | 8/11 DIFF (every program with a loop) |
| `BRA do` → `BRA test` (loop back edge, consistently in the layout lemmas) | fails in `paired_Bk` (the back edge no longer targets `do`) | 8/11 DIFF (every program with a loop) |
| drop the `BNE rt r0 finish` after `end:` in `if_code` | fails in `len_if_code` (layout pinned) | 5/11 DIFF (every program with an `if`) |

## Not covered (next milestones)

1. **Control flow** — `If` and **`Loop`** are DONE (above) for the layouts of
   main a1088e2: tests normalised by `_as_flag` (PR #4, so no Boolean
   restriction), `BNE rt r0 finish` after the `if` join and after the loop's
   entry / re-entry assertions, flag restored by `XORI rt 1` (PR #6). History:
   PR #5 proved the older `_gen_from` whose `XOR rt rt` clears discarded the
   assertions (machine-checked counterexamples, since removed — see "Loops");
   main fixed that on the Python side (2026-09-24), and the proof was redone on
   the new layout (2026-09-25). What remains of milestone 1: violations after
   round k > 1 (only the first round is stated), and `finish` as an executed
   `FINISH` with the garbage check (here: a label plus the "flag = 1"
   conclusion).
2. **Procedures** — `Call` / `Uncall`. The source-side contract is *already*
   machine-checked in `RevProc.v` of the PyJanus development (see below), in the
   more general by-reference-parameter form; what is missing is only that the
   *emitted code* meets it. The Python compiler now does, by branching to an
   inverted companion `f_inv` (the bug where `uncall` ran the body forward is
   fixed), but nothing here proves it.
3. **Arrays**, constant multiplication, comparison operators.
4. **The optimizer** — `peephole` and `remove_nops` are DONE for straight-line
   code (Opt.v: `optimize_run`; writing the proof exposed and fixed an unsound
   cancellation of aliased pairs like `XOR r r ; XOR r r` in `_cancels`), and
   `remove_unused_labels` is DONE (LOpt.v: `strip_exec`, axiom-free, on the
   first PC-based labeled-code model — direct branches only; Pendulum
   RBRA/`br`/SWAPBR remain milestone 1), as is the label *forwarding* inside
   `_peephole_pass` (LOpt.v: `delete_cancelling_pair_fwd` — deleting a
   cancelling pair, moving its label onto the next line, preserves every
   terminating run; a forward simulation with the pc map `phi`, shown for one
   deletion site, which composes over the sites of a pass). Stating it exposed
   another bug: a *labeled* pair at the very end of the code has nowhere to
   carry its label, and `_peephole_pass` silently dropped it, leaving any
   branch to it dangling — it now keeps such a pair (`mergeable` is the
   corresponding side condition in the proof). The simulation is the forward
   direction only (terminating runs of the original are reproduced); the
   converse, and procedure inlining, remain.

Before starting any of these, read "Related existing formalization" below: the
framework there may supply most of milestones 1–2 for free.

## RESUME — where to pick up

- **In sync with main a1088e2** (2026-09-25, branch `feat/rocq-sync-main`):
  `compile_l` emits `_gen_if` / `_gen_from` of main line for line
  (`compile_l_spec`, cross-check 11/11). When `codegen.py`'s control layouts
  change again: edit `if_code` / `loop_code`, then the `Let`s of `Section
  IfLayout` / `Section LoopLayout` (positions, the prefix lists `A0`…`A8`,
  `postA`/`postB`/`postD`/`postS`) — every other lemma is one line or one
  phase (`entry_check`, `entry_steps`, `do_steps`, `exit_steps`,
  `iter_steps`, `s2_steps`, `reentry_check`, `back_steps`,
  `back_violation`), so a layout change is local — and add a program to
  `tools/rocq_loop_crosscheck.py`.
- **Procedures** (milestone 2) are the next milestone of this directory; the
  labeled-fragment theorem (`pre`/`post` quantification) is the interface a
  procedure body will be proved against. `finish` is already a parameter
  (`fin`), which is how a procedure body will refer to the program's exit.
- **Comparisons in Src.expr** (milestone 3): `_as_flag` leaves comparisons and
  `&&`/`||` alone; once Src.expr has them, `is_bool_const` must grow to
  "comparison or logical operator or 0/1 constant" (and `flag_block_spec`
  needs those to evaluate to 0/1). The `ISltx` instruction needed for them is
  already in PISA.v.
- **Later-round violations**: `compile_l_loop_reentry_violation` is stated for
  the first round; the general one is `loop_round`^(k−1) followed by
  `back_violation`, i.e. an induction on `opn_l`.
- **`r0`**: either keep the `regs ms 0 = 0` premise or give PISACtl.v a
  read-through `rread` that returns 0 for register 0 — cheap, but it should
  wait for the register-width change below so PISA.v is touched once.
- **Register width**: PISACtl.v/CompileIf.v use `PISA.state` as is. When the
  fixed-width (`RevSMod`) register file lands in PISA.v, `xor_block_spec`,
  `nz_block_spec` (`sltx_pair` — note `SLTX`'s signed comparison is exactly
  what makes `v ≠ 0` wrap-safe; `_flag_of` avoids `NEG` for that reason) and
  the `compile_at_*` proofs are the places that compute on register values
  (`Z.lxor 1 1`, `rupd_zero`); the control-flow lemmas never look inside
  registers except through `regs s rd =? regs s rs`.
- **Non-Boolean tests**: DONE (2026-09-25) — `_as_flag` is modelled
  (`flag_block`), `exec_c`/`exec_l` use Janus's `<> 0`.
- **Unify the two labeled models**: LOpt.v's `binstr`/`exec_fuel` (direct
  branches only) is subsumed by PISACtl.v; re-stating `strip_exec` and
  `delete_cancelling_pair_fwd` on PISACtl.v would retire LOpt.v's machine.
- **Extraction / differential test**: `Extract.v` and `tools/rocq_diff.py`
  cover the straight-line compiler only; extracting `compile_c` and
  `exec_fuel` would let `rocq_diff.py` compare `if` programs too (the manual
  check above did this once for three programs). For loops,
  `tools/rocq_loop_crosscheck.py` does it without extraction (it reads
  `vm_compute` output), and could absorb the `if` examples as well.

## Extraction and the tie-back to the Python code

The Rocq `compile` is a re-implementation of `codegen.py`'s scheme, not extracted
from it, so the proof says nothing about the Python code on its own. `Extract.v`
extracts the verified compiler and machine to OCaml (`driver.ml` drives them),
and `../tools/rocq_diff.py` compares three things per program:

| check | what a mismatch would mean |
|---|---|
| verified instructions run on `pisa_interp.py` vs on `PISA.run` | the Python **interpreter** disagrees with the formal PISA semantics |
| `codegen.py` output vs the verified compiler's, on the same source | the Python **compiler** disagrees with the verified translation |
| scratch registers at the end | garbage — `clean_above` violated |

```bash
make -f Makefile.driver     # extract + build (needs OCaml)
cd .. && python3 tools/rocq_diff.py
```

Currently 8/8 programs agree (re-run 2026-09-25 after `ISltx` was added to
`PISA.instr`: the extracted `.ml`/`.mli` files were regenerated from
Extract.v and `driver.ml` prints `SLTX`). `Makefile.driver` used to list only
`PISA.ml Src.ml Compile.ml driver.ml`, which stopped linking (`Unbound module
"BinInt"`) once the extracted modules started to `open BinInt` etc.; it now
links every extracted `.ml`/`.mli` in `ocamldep -sort` order and builds the
`.vo` it needs itself, so it works from a clean checkout (2026-09-25).
`ExtrOcamlNatInt`/`ExtrOcamlZInt` realise `nat`
and `Z` by OCaml's native `int`, which the theorems do *not* cover — they are
about unbounded `nat`/`Z`, so the extracted code inherits them only while no
value overflows a 63-bit int.

## Related existing formalization: `yokoyama-lab/PyJanus`, `coq/`

That repository (a *separate* checkout, `github.com/yokoyama-lab/PyJanus`, 39 `.v`
files, Rocq 9.1, whole build ≈3 s) already contains a large machine-checked
development about Janus. It was surveyed in full before the milestones above were
written; this section records what is there, so that work here neither duplicates
it nor misses the reuse.

### The framework

`RevCore.v` isolates reversibility behind a module type `REV_PRIM` whose only
obligations are three local laws on the atomic primitives:

```coq
pinv_invol : pinv (pinv p) = p
pstep_det  : pstep p a b -> pstep p a b' -> b = b'
pstep_rev  : pstep p a b -> pstep (pinv p) b a
```

The functor `RevLang (P : REV_PRIM)` then builds structured control flow
(sequencing, assertion-guarded `if`, `from/loop/until`, `call`/`uncall`), the
inverter, the semantics, and proves `exec_rev` / `exec_iff` / `exec_det` /
`exec_injective` once and for all. `RevNecessity.v` shows the three laws are also
*necessary* (they force primitive injectivity; a "reset to 0" atom is provably
inadmissible), and `RevAlgebra.v` recasts the whole thing as an open algebra of
relational combinators, where each construct's reversibility is a closure lemma.

Instances with no state or primitives in common with Janus — `RevStack.v` (state
= `list Z`), `RevCA.v` (cellular automaton), `RevToy.v` (a counter) — inherit
reversibility verbatim from the functor.

### What is proved there (do not re-prove)

| Area | Where |
|---|---|
| Janus reversibility, big-step | `Janus.v`, `RevJanus.v` |
| Parameterized **by-reference procedures**, incl. `Uncall` | `RevProc.v` (concrete), `RevCoreP.v` (generic functor) |
| Arrays with a runtime aliasing test, `local`/`delocal` | `RevArr.v`, `RevExt.v` |
| Frame-stacked locals + **recursion** | `RevFrame.v` |
| Small-step semantics, **equivalent** to big-step | `RevSmallStep.v` |
| Denotational adequacy, **full abstraction**, `denote_invert` | `RevDenote.v` |
| Dagger / inverse-category structure | `RevInverse.v`, `RevCat.v` |
| **Bennett reversibilization (compute–copy–uncompute)** | `RevBennett.v` (`bennett_correct`) |
| `*=` / `/=` as a partial-injective primitive | `RevMul.v` |
| Sized ints and the `-m bits` mode (modular / signed-window cores) | `RevMod.v`, `RevExtMod.v`, `RevSMod.v`, `RevExtSMod.v` |
| Reversible I/O | `RevIO.v` |
| Verified fuel interpreters extracted to OCaml (six of them) | `RevExtract*.v` |
| Clean-reversible construction from injective specs | `RevPipeline*.v`, `RevGolomb.v`, `RevVarint.v`, `RevZigzag.v`, `RevDeltaN.v` |

`RevExtractFrame.v` backs **`vjanus`**, a standalone verified jana2014
implementation (own lexer/parser/lowering, no Python at runtime) that matches
PyJanus on 48/48 of the corpus, with `vjanus -inverse` running the verified
inverter.

### What is *not* there — where this directory adds something

- **No assembly target.** Every target in that development is a structured
  language (the frame core). PISA is unstructured: labels and branches, with the
  reversibility of a *code layout* rather than of a syntax tree.
- **No whole-translator semantic preservation.** `RevLowering.v` verifies only
  the lowering rules that carry real proof obligations (the XOR-triple swap and
  its aliased collapse, stack `push`/`pop`, the local-array bracket, injectivity
  of struct-array addressing and the Cantor fold) and says explicitly that a Coq
  model of all of `lower.ml` proved to commute with the source semantics remains
  future work. Their roadmap for it is `docs/vjanus-lowering-soundness.md`.
- **No cleanliness statement about a translation.** `RevBennett.v` verifies the
  reversibilization *construction*, and the pipeline files produce clean
  programs, but "the compiler restores every scratch register" — the second
  conjunct of `compile_spec` — is a property of a code generator and is stated
  here.

### Reuse to consider before the next milestone

1. **Make `PISA.v` a `REV_PRIM` instance.** `step_invert` is exactly `pstep_rev`;
   `pinv_invol` and `pstep_det` are immediate. `RevStack.v` / `RevCA.v` show that
   an unrelated state space is fine. Caveat, so as not to overclaim: `RevLang`
   builds *structured* control flow, so this buys the straight-line case (a `Seq`
   chain — i.e. `run_invert_code`) and the **source** side of milestones 1–2. It
   does not by itself say anything about an arbitrary PISA instruction sequence,
   which is what `compile_spec` is about.
2. **`RevProc.v` already fixes the `Uncall` contract** (`E_Uncall` is defined as
   `exec (rename (pbind p args) (invert (pbody p)))`, with `exec_rev` proved).
   Milestone 2 should therefore prove only that the *emitted code* meets that
   contract, not restate it.
3. **`RevBennett.bennett_correct`** for the compute/uncompute argument that
   `gen_expr_spec`'s `Bin` case currently makes by hand.
4. **Adopt their `audit.sh`.** It runs `Print Assumptions` on every headline
   theorem and fails the build on any axiom beyond functional extensionality or
   any `Admitted`, wired to CI as `.github/workflows/coq.yml`. `Test.v` here only
   *prints* its assumptions — nothing fails if that changes.
5. **Fixed-width registers.** `PISA.v` models registers as unbounded `Z`, which
   real PISA is not; this is an unflagged fidelity gap. `RevSMod.v`'s signed
   window `[-2^(b-1), 2^(b-1))` is the ready-made model, and it is validated
   against PyJanus's `-m 8` output. **Experiment** (branch `claude/pisa-fixed-width`): `RevSMod.v`
   (stand-in), `PISAFixed.v`, `CompileFixed.v`, `OptFixed.v`, `TestFixed.v` do exactly this; the
   per-lemma breakage table is in `FIXED_WIDTH_REPORT.md`.
6. **`harness/`** is the established pattern for differential-testing an extracted
   interpreter against PyJanus, wired into their pytest suite;
   `../tools/rocq_diff.py` and `../tools/pyjanus_crosscheck.py` re-invent it.

## Axiom footprint

`functional_extensionality_dep`, and nothing else (`Print Assumptions` in
`Test.v` and at the end of `CompileIf.v` and `CompileLoop.v` reports it at build
time; PISACtl.v's own lemmas and `steps_relabel` are axiom-free). It is used only to promote pointwise equality
of the register file and memory — both higher-order maps, `reg -> Z` and
`addr -> Z` — to Leibniz equality. Removing it would require a first-order
machine state (e.g. a bounded vector of registers). This is the same trade-off
documented in the R-CORE development for `store_ext`.
