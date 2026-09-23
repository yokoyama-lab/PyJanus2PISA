# Fixed-width registers for PISA.v — what breaks and what survives

Experiment branch `claude/pisa-fixed-width`. Everything here compiles with Rocq 9.1.1
(`rocq makefile -f _CoqProject -o Makefile && make`, 11 s, **no `Admitted`, no `admit`, no `Axiom`**;
the only assumption anywhere is `functional_extensionality_dep`, exactly as before).
The original files (`PISA.v`, `Src.v`, `Compile.v`, `Opt.v`, `LOpt.v`, `Test.v`, `Extract.v`) are untouched.

## 1. Purpose

MANIFEST.md, item 5 ("Fixed-width registers"), flags an unflagged fidelity gap: `PISA.v` models
registers and memory cells as unbounded `Z`, whereas real Pendulum/PISA (and PyJanus in `-m 8` mode,
and `pisa_interp.py` if it is ever given a word size) has `b`-bit two's-complement words.  This
experiment replaces the `Z` arithmetic of `step` by arithmetic wrapped into the signed window
`[-2^(b-1), 2^(b-1))` and records, lemma by lemma, which existing proofs still hold, which need a
changed statement, and which are simply false — so that the cost of closing the gap is known before
the real `RevSMod.v` is plugged in.

## 2. Design

### 2.1 `RevSMod.v` — a stand-in

PyJanus's `RevSMod.v` (yokoyama-lab/PyJanus, `coq/RevSMod.v`, validated against `pyjanus -m 8`) is
**not in this repository**, so `rocq/RevSMod.v` here is a minimal stand-in written for this
experiment (its header says so).  Parameterised by `b : nat` with `0 < b`:

```coq
half    := 2 ^ (Z.of_nat b - 1)            (* 2^(b-1) *)
modulus := 2 * half                        (* = 2^b, modulus_pow *)
wrap z  := (z + half) mod modulus - half   (* signed window *)
in_window z := - half <= z < half
```

Lemmas (all proved): `wrap_range`, `wrap_id` (identity on the window), `wrap_idem`, `wrap_half`
(`wrap 2^(b-1) = -2^(b-1)`), the congruence laws `wrap_add_l/r`, `wrap_sub_l/r`, `wrap_neg`,
`wrap_add_mult`, `wrap_mod`, `wrap_congr`; the cancellation laws `wrap_add_sub_cancel`,
`wrap_sub_add_cancel`, `wrap_neg_neg`; and for xor **(A)** `lxor_in_window` (xor of two in-window
values is in the window — sign-case analysis via `Z.lnot`, `Z.log2_lxor`) and **(B)** `wrap_lxor_l/r`
(`wrap` commutes with `Z.lxor` modulo `2^b` — bitwise via `Z.bits_inj'`, `Z.mod_pow2_bits_low/high`),
plus `wrap_lxor_cancel`.  Sanity: `wrap 8 128 = -128`, `wrap 8 (-129) = 127`, `wrap 3 5 = -3` by `reflexivity`.

### 2.2 `PISAFixed.v` — choice (i)

State type unchanged (`regs : reg -> Z`, `mem : addr -> Z`); `step` applies `wrap` to the result of
every ALU instruction (`IAdd ISub IXor IAddi ISubi IXori INeg`); `IExch` is a pure move and is not
wrapped (a move is the identity on words).  The representable states are the invariant

```coq
wf_state s := (forall r, in_window (regs s r)) /\ (forall a, in_window (mem s a))
```

with `step_wf : wf_state s -> wf_state (step i s)` and `run_wf`.  Choice (ii), a subset-type codomain,
was rejected because it changes the state type (hence every downstream statement) and forces the
`rupd`/`mupd` lemmas to be re-proved with proof irrelevance, for no gain in fidelity.  Everything
width-independent (maps, syntax, `invert_instr`, `wf_instr`, `xor_involutive`) is verbatim PISA.v and
lives outside the `Section Fixed` (Variable `b`, Hypothesis `Hb : 0 < b`).  Immediates stay `Z`:
the model accepts any immediate and `wrap`s it; the assembler's immediate width is not modelled.

## 3. Per-lemma results

Legend: **U** = holds unchanged (same statement, same or trivially adapted proof);
**M** = holds with modified statement (what changed); **B** = the PISA.v statement is false on the
fixed-width machine (a machine-checked counterexample is given); **N** = not attempted.

### 3.1 PISA.v → PISAFixed.v

| Lemma / Example | Status | Reason / change |
|---|---|---|
| `rupd_same/other/shadow/id`, `mupd_same/other/shadow/comm/id` | U | width-independent maps |
| `run_nil`, `run_cons`, `run_app`, `run_one` | U | never look inside `step` |
| `wf_code_app`, `xor_involutive` | U | width-independent |
| `step_wf`, `run_wf` (NEW) | — | invariant preservation; needed by everything below |
| `step_invert` | **M** | needs `wf_state s`. IAdd/ISub via `wrap_add_sub_cancel`/`wrap_sub_add_cancel` + `wrap_id`; IAddi/ISubi likewise for **any** immediate; IXor via `lxor_in_window` then `xor_involutive`; IXori via `wrap_lxor_cancel` (immediate may be outside the window, so (A) does not suffice — (B) is needed); INeg via `wrap_neg_neg`; IExch unchanged. |
| `step_invert` without `wf_state` | **B** | `step_invert_needs_wf`: `ADDI r0 0 ; SUBI r0 0` on `r0 = 2^(b-1)` yields `-2^(b-1)` (`addi_subi_bad`). |
| `run_invert_code` | **M** | needs `wf_state s`; induction threads it with `step_wf` |
| `writes_only`, `preserves_mem`, `*_app`, `writes_only_weaken` | U | frame lemmas never inspect `step` |
| `ex_addi` | **M** | needs `4 <= b` (5 must fit); proof is no longer `reflexivity`. At `b = 8` the original statement holds by `reflexivity` (`ex_addi_8`); at `b = 3` it is **false**: the register holds `-3` (`ex_addi_3_wraps`, `ex_addi_3_breaks`). |
| `ex_add_copy` | **M** | needs `4 <= b`; uses `wrap_idem` (`ex_add_copy_8` unchanged at 8 bits) |
| `ex_exch_roundtrip`, `ex_exch_stores` | **M** | need `5 <= b` (9 must fit); `_8` versions unchanged |
| `ex_invert_roundtrip` | **M** | `forall s` becomes `forall s, wf_state s -> …` |
| `ex_overflow_8`, `ex_overflow_8_undone` (NEW) | — | 127+1 = -128 at 8 bits, and SUBI undoes it |

### 3.2 Compile.v → CompileFixed.v (target = PISAFixed, source = unchanged Src.v)

| Lemma | Status | Reason / change |
|---|---|---|
| `rupd_comm`, `rupd_zero`, `of_nat_inj`, `op_instr`, `aop_instr`, `gen_*`, `compile` | U | code generator is width-independent |
| `wf_invert_instr`, `wf_invert_code`, `wf_gen_expr`, `wf_compile` | U | about operand registers only |
| `models_w` (NEW) | — | `mem s (Z.of_nat x) = wrap (σ x)`: the store is observed modulo the window |
| `expr_vars_ok` / `stmt_vars_ok` (NEW) | — | every variable address `Z.of_nat x` must be in the window: `gen_var` loads it with `ADDI`, which wraps, so a variable at address ≥ 2^(b-1) would be read from the wrong cell |
| `gen_var_spec` | **M** | `models_w`; result register gets `wrap (σ x)`; needs `in_window (Z.of_nat x)` |
| `gen_expr_spec` | **M** | needs `wf_state s` (the trailing inverse block uses `run_invert_code`), `expr_vars_ok e`; delivers `wrap (eval σ e)`. Cst case: PISA.v put `n`, now `wrap n`. Bin case: `wrap_add_l/r`, `wrap_sub_l/r`, `wrap_lxor_l/r` show wrapped operands combine to the wrapped result. |
| `models_update` | **M** | `models_w_update`, stores `wrap v` |
| `gen_assign_spec` | **M** | stored value is `wrap (adenote o (σ x) (eval σ e))`; hypotheses `wf_state`, `in_window (Z.of_nat x)`, `expr_vars_ok e` added |
| `gen_swap_spec` | **M** | addresses in the window; values move unwrapped, statement otherwise literal |
| `compile_spec` (as stated in Compile.v) | **B** | `compile_spec_unwrapped_fails`: `x += 2^(b-1)` from `x = 0` (all hypotheses of Compile.v satisfied, plus `wf_state`, plus addresses ok) — Janus gives `2^(b-1)`, the machine stores `-2^(b-1)`. Src.v's `Z` semantics does not match wrapped arithmetic. |
| `compile_spec_w` (NEW form) | **M** | `models` → `models_w`, plus `wf_state ms` and `stmt_vars_ok st`. Intermediate stores may overflow freely; residues track Janus's residues. |
| `compile_spec_in_window` (recovered original) | **M** | Compile.v's exact conclusion, under `wf_state ms`, `stmt_vars_ok st` and `forall x, in_window (σ' x)` — only the **final** store must be representable |
| `compile_reversible` | **M** | needs `wf_state s` |

### 3.3 Opt.v → OptFixed.v

| Lemma | Status | Reason / change |
|---|---|---|
| `cancels`, `cancels_spec`, `peephole_pass`, `peephole_pass_len`, `peephole_iter`, `peephole`, `is_nop`, `remove_nops`, `optimize` | U | code rewriting, width-independent |
| `cancels_undo` | **M** | needs `wf_state s` (via `step_invert`) |
| `peephole_pass_run` | **M** | needs `wf_state s`; the non-cancelling case threads `step_wf` |
| `peephole_pass_run` without `wf_state` | **B** | `peephole_pass_run_needs_wf`: the pair `ADDI r0 0 ; SUBI r0 0` is deleted but is not a no-op on `bad_state` |
| `peephole_iter_run`, `peephole_run` | **M** | needs `wf_state s` |
| `nop_step` | **M** | needs `wf_state s`: `ADDI r 0` normalises `r` |
| `remove_nops_run`, `optimize_run` | **M** | needs `wf_state s` |
| `ex_cancel_pair`, `ex_cascade`, `ex_aliased_xor_kept`, `ex_nops_removed` | U | about code |

### 3.4 Test.v → TestFixed.v

| Example | Status | Reason |
|---|---|---|
| `prog_x/y`, `prog_clean_3..6`, `gen_assign_shape`, `prog_wf` | U at `b = 8` | all values (3, 5) fit; `reflexivity` |
| `prog_vars_ok_8` (NEW) | — | the new address obligation, discharged by computation |
| `prog_reversible` | **M** | needs `wf_state 8 zero_state` (holds) |
| `prog_exec`, `prog_src_x/y` | U | source-side, unchanged in Test.v |
| `prog_x` at `b = 3` | **B** | `x + 2 = 5` wraps to `-3`; machine ends with `x = -3, y = 3` (`prog_x_3_differs`), which is exactly `wrap 3 5` (`prog_x_3_is_wrapped`); still clean and reversible (`prog_reversible_3`) |

### 3.5 LOpt.v, Extract.v

| Item | Status | Reason |
|---|---|---|
| `LOpt.v` (`strip_exec`, `find_label_*`, `exec_mono`, `delete_pair_fwd`, `delete_cancelling_pair_fwd`) | **N** (analysed, not ported) | `strip_exec` and the label lemmas never inspect `step`, so they would port unchanged. `delete_pair_fwd` takes `Hab : forall s, step b (step a s) = s` unconditionally; on the fixed-width machine only `wf_state s -> …` is available (`cancels_undo`), so the section hypothesis must be weakened and `wf_state` threaded through the fuel induction (every state reachable from a `wf_state` start is `wf_state` by `step_wf`). Estimated: mechanical, ~1 day. |
| `Extract.v` | **N** (deliberately) | the fixed files are not extracted; `ExtrOcamlZInt` would silently give a *63-bit* `wrap`, which is the wrong width — plugging in a real width needs `Z` extracted to `Zarith` or to a width-checked int. |

## 4. Admitted items

**None.** Every lemma listed above as U/M/B is closed with `Qed`; `Print Assumptions` (TestFixed.v,
OptFixed.v) reports only `functional_extensionality_dep` for `compile_spec_w`, `compile_spec_in_window`,
`compile_spec_unwrapped_fails`, `compile_reversible`, `optimize_run`, and *closed under the global
context* for `step_invert_needs_wf`.

## 5. `git diff --stat` (against `main` @ 866b8e2)

```
 rocq/CompileFixed.v | 529 ++++++++++++++++++++++++++++++++++++++++++++++++++++
 rocq/OptFixed.v     | 206 ++++++++++++++++++++
 rocq/PISAFixed.v    | 427 ++++++++++++++++++++++++++++++++++++++++++
 rocq/RevSMod.v      | 285 ++++++++++++++++++++++++++++
 rocq/TestFixed.v    | 101 ++++++++++
 rocq/_CoqProject    |   5 +
 6 files changed, 1553 insertions(+)
```
(plus this report and a pointer line in MANIFEST.md, added in the same PR.)

## 6. Discussion

The fidelity gap costs one invariant and one relation, not a redesign: every reversibility theorem
survives with `wf_state s` added (18 lemmas change statement, 0 are lost), and the change is
provably necessary (`step_invert_needs_wf`, `peephole_pass_run_needs_wf`), because a word outside the
window is normalised by the first instruction and no inverse can recover it.  The compiler side is
where the gap bites: Compile.v's `compile_spec` is false as stated (`compile_spec_unwrapped_fails`),
since `Src.v` evaluates `x += e` in `Z` while the machine evaluates it modulo `2^b`; it survives in
two forms — modulo the window (`compile_spec_w`, with `models_w`) unconditionally, and in its exact
original form (`compile_spec_in_window`) under the hypothesis that the *final* store is representable
(intermediate overflow is harmless because residues compose), plus two hypotheses PISA.v never
needed: a representable initial machine state and variable addresses below `2^(b-1)`.  A weaker but
cleaner alternative is to wrap the *source* semantics too (`adenote` through `wrap`, as PyJanus's
`-m` mode does), which would make `compile_spec` hold verbatim with `models`; that is the shape the
real PyJanus `RevSMod.v` supports, and the differential test (`tools/rocq_diff.py`) should then compare
against `pyjanus -m b`, not against the unbounded interpreter.  To plug the real `RevSMod.v` in, only
section 2.1 must be matched: `wrap`, `in_window`, and the 14 named laws above — if PyJanus states
`wrap` differently (e.g. via `Z.modulo` with a `2^b` offset or via `Zmod`), a 20-line bridge file
proving these laws from theirs is all that `PISAFixed.v` needs, since nothing downstream unfolds `wrap`.

## 7. RESUME (days 2–7)

1. **Real RevSMod.v**: obtain PyJanus `coq/RevSMod.v`; write `RevSModBridge.v` proving this file's
   14 laws from theirs (or replace `RevSMod.v` outright if the definitions coincide); re-run `make`.
2. **Wrapped source semantics**: add `SrcW.v` (or a `-m` flag on `Src.v`) with `adenote`/`denote` through
   `wrap`; prove `compile_spec` verbatim with `models`; prove `exec_rev` still holds (needs
   `wrap_add_sub_cancel` on the source side — same laws).
3. **LOpt.v port** (mechanical): weaken `Hab` to `wf_state s -> …`, thread `wf_state` through
   `exec_fuel` (`exec_fuel_wf`), re-prove `delete_pair_fwd`.
4. **Differential test**: extend `tools/rocq_diff.py` with a `-m 8` mode; compare `PISAFixed.run 8`
   (via a small `Extract` with `Z` → Zarith, not `ExtrOcamlZInt`) against `pyjanus -m 8` and against
   `pisa_interp.py` once it grows a word size. Expected disagreements: exactly the overflow cases
   (`ex_overflow_8`, `prog_x_3`).
5. **Immediates**: model the assembler's immediate width (PISA `ADDI` has a bounded immediate) as a
   `wf_instr` side condition and check `codegen.py` splits large constants accordingly.
6. **Addresses**: `stmt_vars_ok` is a real constraint on `regalloc.py`/`DATA` layout — assert it in
   `codegen.py` (variable index < 2^(b-1)) and in the Rocq compiler as a `wf_stmt` conjunct.
7. **MANIFEST.md item 5**: once (1)–(2) land, move the fixed-width files from "experiment" to the main
   table and add `PISAFixed`/`CompileFixed` to `audit.sh`-style `Print Assumptions` checks.
