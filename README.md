# PyJanus2PISA

A Python implementation of a compiler from **Janus** to **PISA** (Pendulum Instruction Set Architecture), together with a PISA interpreter and a Janus program inverter.

## Overview

This project implements the *clean translation* of Janus programs into PISA assembly as described in:

> Axelsen, H. B. "Clean Translation of an Imperative Reversible Programming Language." *CC 2011*, LNCS 6601, pp. 144–163.

Program inversion follows the rules from:

> Yokoyama, T. and Glück, R. "A Reversible Programming Language and Its Invertible Self-Interpreter." *PEPM 2007*, pp. encodes–78.

**Janus** is a reversible imperative programming language in which every program computes a bijection on the program state. **PISA** is a reversible assembly language whose every instruction has a local inverse, making the whole machine state reversible.

## Features

- Full Janus compiler: lexer → parser → code generator → PISA assembly
- PISA interpreter with Pendulum branch semantics and software call stack
- Program inverter: given P, produces P⁻¹ (the semantic inverse) at the AST level
- CLI with `--inverse`, `--ast`, and `--tokens` flags
- Optimization passes: peephole cancellation with store-block fusion (EXCH/EXCH), NOP removal, unreferenced-label removal, procedure inlining (with size limit; branch-free bodies only for `uncall` safety), and self-referencing assignment optimization
- Multiplication by a compile-time constant, compiled to a branch-free shift-and-add chain
- 334 tests (all passing)

On a representative program exercising conditionals, loops, arrays, and procedure calls, the optimization passes reduce code size from 268 to 217 instructions (≈19%); see `program_stats` in `codegen.py`.

### Limitations

- Multiplication requires one operand to be a compile-time constant, at most 16 bits wide. PISA has no `MUL` instruction, and a data-dependent multiply loop could not be inverted by the straight-line reversal used for expression unevaluation. `variable * variable` raises `CodeGenError`.
- Division (`/`) and modulo (`%`) are not supported by the code generator.
- Procedures take no parameters; all variables are global.

### How `uncall` is compiled

`uncall f` branches to an **inverted companion procedure** `f_inv`, whose body is
`invert_stmt(f.body)`; the code generator emits one for every uncalled procedure
(transitively, since an inverted body's `call g` came from an `uncall g`).

A plain `RBRA f` does not work: PISA's `RBRA` means "run backwards", but the
bundled interpreter has no Pendulum direction bit, so it executed `f` *forward*
and `uncall f` behaved exactly like `call f`. That bug was found by differential
testing against PyJanus and is fixed; `uncall bump` on `bump: x += 1` now gives
`x = -1`, agreeing with PyJanus.

Consequently `invert_program` inverts **only the main procedure's body** and
leaves the procedure environment alone — the standard rule, since `uncall f` now
genuinely runs `f` backwards. (Formally: `exec` is relative to a fixed
environment Γ and inversion keeps Γ; see `exec_rev` in `rocq/Src.v`.)

The interpreter's call/return protocol was fixed at the same time: a return is
now recognised at `f_bot` itself, and `br` is saved and cleared across a call.
Previously the return went through the paired-branch machinery, whose `br`
bookkeeping the prologue's `SWAPBR` perturbs, so the second call to any
procedure fell through to whatever instruction physically followed it — which
happened to be harmless for the old layouts and looped forever for new ones.

### `if` exit assertions are checked

For `if e1 then S1 else S2 fi e2`, Janus requires `e2` to hold on exit exactly
when `e1` held on entry. The branch flag is `1` on the then path and `0` on the
else path, and `eval(e2)` is XORed into it on **both** paths, so the flag ends at
`0` precisely when the assertion holds. A correct program therefore leaves no
garbage; a violated assertion leaves the discrepancy in the flag, and the
interpreter reports it at `FINISH`:

```
PISAError: garbage left in registers at FINISH (r3=1); the program is not
reversible — most likely an `if` exit assertion that does not hold on the path
taken
```

This ties the check to the *cleanliness* invariant proved in `rocq/Compile.v`
(`clean_above`): a clean translation is garbage-free, so leftover garbage means
the source program was not reversible. Set `machine.check_clean = False` to
inspect a broken program's final state instead of raising.

Previously `e2` was evaluated only on the else path and then discarded — the
then path skipped it with a `BNE` — so the assertion was never checked. The
program below ran silently to `x = 1`, and its inverse mapped `1` to `-1`
instead of back to `0`:

```janus
int x
procedure main
  if x = 0 then x += 1 else x += 2 fi x = 5
```

PyJanus reports "Assertion failed" for the same program; the two now agree.

## Machine-checked correctness (Rocq)

`rocq/` contains a Rocq 9.1 development proving **semantic preservation** for the
straight-line fragment of the translation — the PISA counterpart of the
whole-translator correctness that the PyJanus development leaves open:

```coq
Theorem compile_spec : forall st sigma sigma' m,
  exec st sigma sigma' -> wf_stmt st ->
  models m sigma -> clean_above scratch m ->
  models (run (compile st) m) sigma' /\ regs (run (compile st) m) = regs m.
```

The second conjunct is *cleanliness*: the register file afterwards is equal to
the one before, so no garbage is left — the "clean" of Axelsen's clean
translation, and what makes the compiled code reversible on the machine
(`compile_reversible`). Also proved: every PISA instruction has a local inverse
(`step_invert`), and the emitted code is well-formed, i.e. never `ADD rd rd`
(`wf_compile`).

Covered: `skip`, `x op= e`, `x <=> y`, sequencing, expressions over `+ - ^`,
and the optimizer passes: `peephole` / `remove_nops` (`Opt.v`: they preserve
`run` on all straight-line code — writing that proof exposed an unsound
cancellation of aliased pairs like `XOR r r ; XOR r r` in `_cancels`, now
fixed) and `remove_unused_labels` (`LOpt.v`: `strip_exec`, axiom-free, on a
PC-based labeled-code machine with direct branches). Not yet: control flow
compilation, procedures, arrays, peephole label forwarding, inlining. No `Admitted`; the only axiom is
functional extensionality. See `rocq/MANIFEST.md` for the exact scope, side
conditions, and next milestones.

```bash
cd rocq && make      # Rocq Prover 9.1.1
```

`Extract.v` extracts the verified compiler to OCaml, and
`tools/rocq_diff.py` diffs it against the Python one — catching the case where
the proof is about a compiler that has drifted from `codegen.py`:

```bash
make -C rocq -f Makefile.driver   # needs OCaml
python3 tools/rocq_diff.py        # 8/8 programs agree
```

It checks both directions at once: the verified compiler's instructions run on
`pisa_interp.py` must give what the formal machine model gives (validating the
Python *interpreter*), and `codegen.py`'s output on the same source must give
the same store (validating the Python *compiler*).

## Cross-checking against PyJanus

`tools/pyjanus_crosscheck.py` runs the same program through this compiler (→ PISA → PISA interpreter) and through the [PyJanus](https://github.com/yokoyama-lab/PyJanus) interpreter, then compares the final store.

```bash
python3 tools/pyjanus_crosscheck.py --pyjanus ~/dev/github.com/yokoyama-lab/PyJanus
```

The two projects share statement syntax but differ in declarations, so the tool translates the source (`emit_jana2014`): PyJanus's `jana2014` dialect has no global variables, so each global is threaded through every procedure as a call-by-reference parameter and declared locally in `main`.

The **common subset** that cross-checks cleanly is:

| Construct | Notes |
|---|---|
| `x += e`, `x -= e`, `x ^= e` | identical syntax |
| `x <=> y`, `a[i] <=> b[j]` | identical syntax |
| `if e1 then S1 else S2 fi e2` | identical; `e2` must genuinely discriminate the branches, which PyJanus checks at runtime and this compiler does not |
| `from e1 do S1 loop S2 until e2` | identical |
| arrays, constant multiplication | identical |
| `call f` / `uncall f` | needs the parameter-threading shim; agrees since the `uncall` fix |

Outside the subset: procedure parameters, local declarations, `print`, and division/modulo.

## Requirements

Python 3.10 or later. No external dependencies.

## Installation

```bash
git clone https://github.com/yokoyama-lab/PyJanus2PISA.git
cd PyJanus2PISA
```

## Usage

```bash
# Compile a Janus program to PISA assembly (stdout)
python3 janus2pisa.py program.janus

# Write output to a file
python3 janus2pisa.py program.janus -o output.pisa

# Compile the semantic inverse P⁻¹
python3 janus2pisa.py --inverse program.janus

# Print the AST
python3 janus2pisa.py --ast program.janus

# Print the token stream
python3 janus2pisa.py --tokens program.janus
```

### Example

```janus
int x
int y
procedure main
  x += 3
  y += x
  x <=> y
```

```bash
$ python3 janus2pisa.py example.janus
       DATA 0
       DATA 0
main_top: BRA main_bot
main: SUBI r1 1
      EXCH r2 r1
      ...
```

## Module Structure

| Module | Description |
|---|---|
| `janus2pisa.py` | CLI entry point |
| `lexer.py` | Tokenizer |
| `parser.py` | Recursive-descent parser; produces AST |
| `syntax.py` | AST node definitions |
| `codegen.py` | AST → PISA instruction list (Axelsen CC 2011, Figs. 5, 6, 11, 12) |
| `regalloc.py` | Three-category register allocator (free / committed / garbage) |
| `pisa.py` | PISA instruction set dataclasses and text printer |
| `pisa_interp.py` | PISA interpreter with Pendulum branch semantics |
| `inverse.py` | Janus program inverter (Yokoyama & Glück PEPM 2007) |

## Janus Language

Janus supports the following statements:

| Statement | Meaning |
|---|---|
| `x += e` | Reversible addition |
| `x -= e` | Reversible subtraction |
| `x ^= e` | Reversible XOR |
| `x <=> y` | Variable swap |
| `if e1 then S1 else S2 fi e2` | Reversible conditional |
| `from e1 do S1 loop S2 until e2` | Reversible loop |
| `call f` / `uncall f` | Forward / backward procedure call |

The `if` statement requires that `e1` and `e2` discriminate the branches: `e1` holds on entry iff `e2` holds on exit. This is the key reversibility constraint.

## Program Inversion

The inverter implements the syntactic inversion rules:

| Statement | Inverse |
|---|---|
| `x += e` | `x -= e` |
| `x ^= e` | `x ^= e` |
| `x <=> y` | `x <=> y` |
| `call f` | `uncall f` |
| `if e1 then S1 else S2 fi e2` | `if e2 then S1⁻¹ else S2⁻¹ fi e1` |
| `from e1 do S1 loop S2 until e2` | `from e2 do S2⁻¹ loop S1⁻¹ until e1` |
| `S1; S2; …; Sn` | `Sn⁻¹; …; S2⁻¹; S1⁻¹` |

The round-trip property P⁻¹(P(σ)) = σ is verified by the test suite for a variety of programs.

## Running Tests

```bash
python3 -m pytest test_janus2pisa.py test_inverse.py test_pisa_interp.py -v
```

## References

- Axelsen, H. B. "Clean Translation of an Imperative Reversible Programming Language." *CC 2011*.
- Axelsen, H. B., Glück, R., and Yokoyama, T. "Reversible Machine Code and Its Abstract Processor Architecture." *CSR 2007*.
- Yokoyama, T. and Glück, R. "A Reversible Programming Language and Its Invertible Self-Interpreter." *PEPM 2007*.

## License

MIT License. See [LICENSE](LICENSE).
