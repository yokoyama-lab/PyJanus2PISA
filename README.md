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
- 312 tests (all passing)

On a representative program exercising conditionals, loops, arrays, and procedure calls, the optimization passes reduce code size from 268 to 217 instructions (≈19%); see `program_stats` in `codegen.py`.

### Limitations

- Multiplication requires one operand to be a compile-time constant, at most 16 bits wide. PISA has no `MUL` instruction, and a data-dependent multiply loop could not be inverted by the straight-line reversal used for expression unevaluation. `variable * variable` raises `CodeGenError`.
- Division (`/`) and modulo (`%`) are not supported by the code generator.
- Procedures take no parameters; all variables are global.

### Known issue: `uncall`

`uncall f` currently compiles to `RBRA f`, which the bundled PISA interpreter executes **forward** — so `uncall f` behaves like `call f` rather than running `f` backwards. The round-trip tests do not catch this because `invert_program` inverts *every* procedure body, so the two inversions cancel; whole-program inversion is therefore correct, but a source program that uses `uncall` directly is miscompiled.

Confirmed by differential testing against PyJanus (see below): for

```janus
int x
procedure bump
  x += 1
procedure main
  uncall bump
```

this compiler yields `x = 1` where the Janus semantics (and PyJanus) give `x = -1`.

Fixing it means either giving the interpreter a Pendulum direction bit so `RBRA` really runs code backwards, or emitting an inverted companion procedure `f⁻¹` for each uncalled `f` and compiling `uncall f` as a call to it. The second option is local to `codegen.py` and can reuse `invert_stmt` from `inverse.py`, but it also requires `invert_program` to stop inverting callee bodies.

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
| `call f` / `uncall f` | needs the parameter-threading shim; `uncall` currently diverges (see above) |

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
