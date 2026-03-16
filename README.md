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
- 283 tests (all passing)

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
