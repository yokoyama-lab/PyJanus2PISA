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
2. **Procedures** — `Call` / `Uncall`. The contract to state and prove is
   `exec Γ (Uncall p) a b ↔ exec Γ (invert (Γ p)) a b`. The Python compiler now
   meets it by branching to an inverted companion `f_inv` (the bug where
   `uncall` ran the body forward is fixed), but nothing here proves that yet.
3. **Arrays**, constant multiplication, comparison operators.
4. **The optimizer** — `peephole`, `remove_nops`, `remove_unused_labels`,
   inlining. Each should be proved to preserve `run`.

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

## Axiom footprint

`functional_extensionality_dep`, and nothing else (`Print Assumptions` in
`Test.v` reports it at build time). It is used only to promote pointwise equality
of the register file and memory — both higher-order maps, `reg -> Z` and
`addr -> Z` — to Leibniz equality. Removing it would require a first-order
machine state (e.g. a bounded vector of registers). This is the same trade-off
documented in the R-CORE development for `store_ext`.
