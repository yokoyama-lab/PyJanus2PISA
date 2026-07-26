# Machine-checked semantic preservation for Janus → PISA

Rocq Prover 9.1.1. Build with `make` (regenerate with `rocq makefile -f _CoqProject -o Makefile`).
No `Admitted`, no `admit`, no local `Axiom`.

This directory answers the question "is the translation this repository implements
actually correct?" for the **straight-line fragment** of Janus. It is the PISA
counterpart of the "whole-translator semantic preservation" that
`RevLowering.v` in the PyJanus development explicitly leaves open.

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

## Not covered (next milestones)

1. **Control flow** — `If` / `Loop`. Needs PISA branches (`BRA`/`RBRA`/`BEQ`/
   `BGEZ`) and the paired-branch (Pendulum) mechanism `pisa_interp.py` implements,
   so the machine model must grow a program counter and a branch direction.
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
   RBRA/`br`/SWAPBR remain milestone 1). Remaining: the label *forwarding*
   inside `_peephole_pass` (a label moves onto the line after a cancelled
   pair), and procedure inlining.

Before starting any of these, read "Related existing formalization" below: the
framework there may supply most of milestones 1–2 for free.

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

Currently 8/8 programs agree. `ExtrOcamlNatInt`/`ExtrOcamlZInt` realise `nat`
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
   against PyJanus's `-m 8` output.
6. **`harness/`** is the established pattern for differential-testing an extracted
   interpreter against PyJanus, wired into their pytest suite;
   `../tools/rocq_diff.py` and `../tools/pyjanus_crosscheck.py` re-invent it.

## Axiom footprint

`functional_extensionality_dep`, and nothing else (`Print Assumptions` in
`Test.v` reports it at build time). It is used only to promote pointwise equality
of the register file and memory — both higher-order maps, `reg -> Z` and
`addr -> Z` — to Leibniz equality. Removing it would require a first-order
machine state (e.g. a bounded vector of registers). This is the same trade-off
documented in the R-CORE development for `store_ext`.
