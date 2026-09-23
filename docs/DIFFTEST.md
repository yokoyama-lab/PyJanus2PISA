# Differential testing of PISA interpreters (pyjanus2pisa vs phpisa)

Status: day 1 of a one-week plan (2026-09-23).  Everything here was produced
by the tools under `tools/` on the corpus under `tests/difftest_corpus/`; the
numbers can be regenerated with the commands in "How to run".

The two interpreters under test:

* `pisa_interp.PISAMachine` (this repository) – runs the instruction
  dataclasses of `pisa.py` that `codegen.py` emits; unbounded Python ints;
  procedure calls through a *software call stack*; branches are mostly
  *direct jumps* (see the module docstring of `pisa_interp.py`).
* [phpisa](https://github.com/yokoyama-lab/phpisa) – PHP 8 port of the 2001
  phPISA Pendulum simulator: `PC += (BR == 0 ? DIR : BR)`, a direction bit
  (`RBRA` runs code backwards through reverse templates), 32 registers,
  PHP 64-bit ints, memory aliased with the program array.

Two more producers of PISA text are fed through the same harness:
rfcl's `rl-to-pisa` (RL → "PISA-flavoured" 3-operand text) and the
Rlang-compiler (M. Frank's R → PAL; the checked-in `test/sch/sch.pal`).

## How to run

```bash
# prerequisites: python3 >= 3.10, php >= 8.1, a phpisa checkout (default ../phpisa,
# override with --phpisa DIR or $PHPISA_DIR); rfcl checkout for the rfcl mode.

# 1. adapter: janus2pisa output -> phpisa PAL
python3 janus2pisa.py prog.janus -o prog.pisa
python3 tools/pisa2pal.py prog.pisa -o prog.pal                   # faithful, instruction by instruction
python3 tools/pisa2pal.py --pendulum-cf prog.pisa -o prog.pal     # + Pendulum call protocol and if/from lowering
python3 tools/pisa2pal.py --dialect rfcl rl.pisa -o rl.pal        # rfcl's comma dialect
php ../phpisa/bin/phpisa prog.pal
php tools/phpisa_dump.php --phpisa ../phpisa prog.pal             # final registers + memory as JSON

# 2. PAL -> pisa.py dataclasses -> (extended) PISAMachine
python3 tools/pal2pisa.py ../phpisa/samples/mult.pal --max-steps 100000

# 3. the harness
python3 tools/difftest_pisa.py janus tests/difftest_corpus/*.janus --mode both --keep /tmp/keep
python3 tools/difftest_pisa.py rfcl --rfcl ../rfcl ../rfcl/examples/*.rl
python3 tools/difftest_pisa.py pal ../Rlang-compiler/test/sch/sch.pal --max-steps 3000000 \
        --expect ../Rlang-compiler/test/sch/output.txt

# 4. tests (php tests skip when php or ../phpisa is missing)
python3 -m pytest test_difftest.py -q
```

Per program the `janus` mode reports

| column | meaning |
|---|---|
| fwd | forward program on both interpreters; final variable memory (`DATA` region) and all 32 registers compared |
| bwd | the `--inverse` program, run from the same initial state, compared the same way |
| rt-pisa | forward then inverse on pisa_interp: store back to the initial one? |
| rt-php | the same on phpisa (the inverse program's `DATA` words are patched with the forward result) |

`OK`, `MISMATCH(first differing location)`, `ERROR(side: message)`.
`--mode faithful` converts instruction by instruction; `--mode pendulum-cf`
applies the two adapter transformations described next.  A discrepancy that
only appears because of an adapter transformation is class (c), not a finding.

## Adapter: `tools/pisa2pal.py`

Mapping decisions (janus2pisa dialect → phpisa PAL):

| janus2pisa | phpisa PAL | note |
|---|---|---|
| `rN` | `$N` | 32 registers in both |
| `ADD/SUB/XOR/NEG/ADDI/XORI/EXCH/BRA/RBRA/BEQ/BNE/BGEZ/SWAPBR/DATA/START/FINISH` | same mnemonic | operand order identical (`EXCH rd ra`: `swap(R[d], MEM[R[a]])`) |
| `SUBI rd c` | `ADDI $d -c` | phpisa has no SUBI |
| `SLTX rd rs rt` (rd ^= rs<rt) | `SUB $s $t; A: BGEZ $s B; XORI $d 1; B: BGEZ $s A; ADD $s $t` | phpisa (and Frank's PISA) have no SLTX; the BGEZ pair is a valid Pendulum skip; `rs=r0`/`rt=r0` use `BLEZ`/`BGEZ` without the SUB; error if rd aliases a source |
| `ORX rd rs` (rd \|= rs; rs := 0) | `ANDX $d $d $s; XOR $d $s; XOR $s $s` | exact: `(d & ~s) ^ s = d \| s`; janus2pisa's ORX is *not* PISA's 3-operand `ORX $d $s $t` (d ^= s\|t) |
| `ANDX rd1 rd2 rs` (rd1 ^= rd2&rs; rd2 := 0) | `ANDX $d1 $d2 $s; XOR $d2 $d2` | exact; PISA's ANDX does not clear rd2 |
| `ADDI/XORI` with 1023 < \|c\| ≤ 8184 | several `ADDI` | phpisa immediates are 11-bit two's complement |
| `ADDI/XORI` with larger \|c\| < 2^32 | built in an unused scratch register with `ADDI`/`RL 10` chunks, applied with `ADD`/`SUB`/`XOR`, then un-computed with `ADDI -c`/`RR 10` | negative XOR constants use `x ^ c = ~(x ^ ~c)`, `~y = NEG; ADDI -1`; > 32-bit patterns are an error |
| `ADDI r0 0` (labelled NOP) | kept | any other write to r0 is an error (r0 is hard-wired in pisa_interp, ordinary in phpisa) |
| `start: START` | `.start start` + `START` | phpisa starts at the `.start` label; PISAMachine at the label literally named `start` |
| labels | kept; error if two labels collide case-insensitively | phpisa upper-cases every line |
| branch to a label further than ±1024 lines | error | phpisa encodes offsets in 11 bits and truncates silently |

`--pendulum-calls` (implied by `--pendulum-cf`): janus2pisa's procedure
prologue `f: SUBI r1 1; EXCH r2 r1; SWAPBR r2; NEG r2; EXCH r2 r1; ADDI r1 1`
is replaced by Axelsen's `f: SWAPBR r2; NEG r2; SUBI r1 1; EXCH r2 r1` … `EXCH
r2 r1; ADDI r1 1; f_bot: BRA f_top`, and the stack base (`ADDI r1 K` after
`START`) is moved above the code.  Reason: the branch target `f` must be the
`SWAPBR` for `BRA f` to work on a Pendulum machine, and phpisa's memory *is*
the program array, so janus2pisa's stack at address `nvars+3` would overwrite
instructions.  On pisa_interp the original prologue is a no-op (br is 0 at a
call) — the return is done by the software call stack.

`--pendulum-cf`: at each landing site of a janus2pisa direct jump a branch
that cancels the incoming offset exactly when control arrived by the jump is
inserted, using the compiler's own path flag `rt`:
`if_assert: BEQ rt r0 <the BRA if_assert>` (else path has rt=0, then path
falls through with rt=1); `from_loop: BEQ rt r0 <the loop-test BEQ>`;
`from_exit: BRA <the exit BRA>`; `from_do: BNE rt r0 <the back-edge BRA>; XOR
rt rt` (the `XOR rt rt` codegen emits before the back-edge is dropped so rt=1
discriminates the back edge from the first entry).  Labels are never moved
(codegen's `remove_nops` forwards labels of removed NOPs, so `from_exit` can
alias `main_bot`); the jumping instruction is retargeted to a fresh label.

## Dialect differences

| # | topic | pisa_interp (pyjanus2pisa) | phpisa | evidence |
|---|---|---|---|---|
| 1 | branch semantics | `BRA`/conditional to a non-branch target = direct jump (`pc = target`); Pendulum `br` arithmetic only for detected *pairs* | always `BR += offset`, then `PC += (BR==0 ? DIR : BR)`; the target must cancel BR | `pisa_interp.py:23-30, 345-346`; `phpisa/src/runner.php:60`, `docs/INSTRUCTIONS.md:74` |
| 2 | `RBRA` | identical to `BRA` (no direction bit) | `DIR *= -1; BR += off`; instructions then execute through reverse templates (`ADDI` → `NEG;ADDI;NEG`) | `pisa_interp.py:297`; `phpisa/src/asm_special.php:23-28`, `php_functions.php:decode` |
| 3 | procedure call | `BRA f` to a known procedure pushes `pc+1` on a software stack; return recognised at `f_bot`; prologue executed twice | pure Pendulum protocol: `BRA f` … `f: SWAPBR rro` … `f_bot: BRA f_top` / `f_top: BRA f_bot` falls into `SWAPBR` again | `pisa_interp.py:308-328`; `phpisa/samples/mult.pal` |
| 4 | registers | 32; `r0` reads 0, writes discarded | 32 (`R0..R31`); `R0` ordinary | `pisa_interp.py:186-193`; `php_functions.php:8` |
| 5 | register width | unbounded Python `int` | PHP `int` (64-bit); overflow turns the value into a `float`; only rotates and `SRLX` are 32-bit | `asm_arith_log.php` (`r_ADD` has no mask), `INSTRUCTIONS.md` shift table |
| 6 | immediates | any Python int | 11-bit two's complement; `e_ADDI`/`e_XORI` keep `imm & 0x7FF` silently; labels allowed as `ADDI` immediates (`-LABEL` negates) | `enc_arith_log.php:e_ADDI`, `asm_branch.php:sign_extend11`, `php_functions.php:60` |
| 7 | branch offsets | label lookup, any distance | 11-bit, silently truncated | `enc_branch.php:5`, `INSTRUCTIONS.md:88,140` |
| 8 | memory | separate sparse dict; only the *leading* run of `DATA` words is loaded (addresses 0..k-1) | memory **is** the program array: `EXCH` reads instruction words / DATA at that line and `set_mem` overwrites program lines; `DATA` anywhere is addressable | `pisa_interp.py:130-137`; `php_functions.php:13-25`, `INSTRUCTIONS.md:104` |
| 9 | operand forms | 2-operand `ADD/SUB/XOR`; 2-operand `ORX` and 3-operand `ANDX` with *clearing* semantics; `SLTX`; `SUBI` | 2-operand `ADD/SUB/XOR`; 3-operand XOR-into-dest `ANDX/ORX/NORX`; `ANDIX/ORIX`; shifts/rotates; `BGTZ/BLEZ/BLTZ`; `SHOW/OUTPUT`; no `SLTX`, no `SUBI` | `pisa.py`; `INSTRUCTIONS.md` |
| 10 | entry / exit | label `start` (required), `FINISH` returns the memory dict; garbage check on r3..r31 optional | `.start L` directive; `FINISH` throws `FinishException`; falling off either end stops silently with `finished=false`; `max_steps` 1,000,000 by default | `pisa_interp.py:140-143, 266-269`; `runner.php:45-66` |
| 11 | labels | case-sensitive | case-insensitive (`strtoupper`), `;` comments, commas ignored | `php_functions.php:176-179` |
| 12 | `XOR r r` (clear) | executed (irreversible) | executed (irreversible) | both accept it; codegen emits it for flag clearing |
| 13 | final state | memory dict (+ `regs`, `br`, `dump_state()`) | registers, `pc`, `direction`, `branch_reg`, `finished`, captured output; memory only via the `$program` global (used by `tools/phpisa_dump.php`) | `runner.php:70-77` |

## Corpus

`tests/difftest_corpus/` – 31 files.  janus-examples (15 legal `.j2` + 4
`.j1`): 14 originals translated into 15 files (the Fibonacci program gives
`j1-fib` and `j1-fib-bwd`), 5 skipped with the reason in the file
(`call-stack`, `show-stack`, `swap-stack`: stacks; `j1-factorization`:
division/modulo/`read`; `j1-stack`: fragment).  Each translated file names the
original and what changed in its header.  Own stress programs `s1`–`s10`
(negative numbers, XOR of negatives, overflow past 2^31/2^32/2^63, large
immediates, nested loops, call/uncall/recursion, comparisons with negative
operands, and two deliberately invalid programs `s9-self-assign`,
`s10-loop-assert`).

## Results

Janus corpus, `--mode both` (faithful and pendulum-cf), 2026-09-23.

Summary over the 26 assembled programs (5 SKIPPED):

| mode | fwd | bwd | rt-pisa | rt-php |
|---|---|---|---|---|
| faithful | 0 OK / 0 MISMATCH / 26 ERROR | 0 OK / 0 MISMATCH / 26 ERROR | 23 OK / 2 MISMATCH / 1 ERROR | 0 OK / 0 MISMATCH / 26 ERROR |
| pendulum-cf | 23 OK / 1 MISMATCH / 2 ERROR | 25 OK / 0 MISMATCH / 1 ERROR | 23 OK / 2 MISMATCH / 1 ERROR | 22 OK / 2 MISMATCH / 2 ERROR |

Per program:

| program | mode | fwd | bwd | rt-pisa | rt-php |
|---|---|---|---|---|---|
| call-array | faithful | ERROR(phpisa: did not reach FINISH (pc=-6, br=-9, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-6, br=-9, dir=1)) | OK | ERROR(fwd failed) |
| call-array | pendulum-cf | OK | OK | OK | OK |
| call-stack | faithful | SKIPPED: stacks (`stack s`, push/pop) are not part of pyjanus2pisa's Janus dia… | | | |
| call-stack | pendulum-cf | SKIPPED: stacks (`stack s`, push/pop) are not part of pyjanus2pisa's Janus dia… | | | |
| comment-multipleline | faithful | ERROR(phpisa: did not reach FINISH (pc=-7, br=-9, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-7, br=-9, dir=1)) | OK | ERROR(fwd failed) |
| comment-multipleline | pendulum-cf | OK | OK | OK | OK |
| comment | faithful | ERROR(phpisa: did not reach FINISH (pc=-7, br=-9, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-7, br=-9, dir=1)) | OK | ERROR(fwd failed) |
| comment | pendulum-cf | OK | OK | OK | OK |
| if-then | faithful | ERROR(phpisa: did not reach FINISH (pc=-104, br=-106, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-104, br=-106, dir=1)) | OK | ERROR(fwd failed) |
| if-then | pendulum-cf | OK | OK | OK | OK |
| if-thenelse | faithful | ERROR(phpisa: did not reach FINISH (pc=-107, br=-109, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-107, br=-109, dir=1)) | OK | ERROR(fwd failed) |
| if-thenelse | pendulum-cf | OK | OK | OK | OK |
| j1-factorization | faithful | SKIPPED: uses division `/`, modulo `¥`, variable*variable multiplication and `… | | | |
| j1-factorization | pendulum-cf | SKIPPED: uses division `/`, modulo `¥`, variable*variable multiplication and `… | | | |
| j1-fib-bwd | faithful | ERROR(phpisa: did not reach FINISH (pc=-173, br=-177, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=275, br=136, dir=1)) | OK | ERROR(fwd failed) |
| j1-fib-bwd | pendulum-cf | OK | OK | OK | OK |
| j1-fib | faithful | ERROR(phpisa: did not reach FINISH (pc=-7, br=-15, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-166, br=-170, dir=1)) | OK | ERROR(fwd failed) |
| j1-fib | pendulum-cf | OK | OK | OK | OK |
| j1-sort | faithful | ERROR(phpisa-side adapter: line 27: branch offset 1074 to main_bot ex… | ERROR(phpisa-side adapter: line 1153: branch offset -1125 to main exc… | ERROR(fwd failed) | ERROR(fwd failed) |
| j1-sort | pendulum-cf | ERROR(phpisa-side adapter: line 27: branch offset 1084 to main_bot ex… | ERROR(phpisa-side adapter: line 1164: branch offset -1136 to main exc… | ERROR(fwd failed) | ERROR(fwd failed) |
| j1-stack | faithful | SKIPPED: the original file is a fragment (only `procedure alloc_tmp`) with und… | | | |
| j1-stack | pendulum-cf | SKIPPED: the original file is a fragment (only `procedure alloc_tmp`) with und… | | | |
| loop-do | faithful | ERROR(phpisa: did not reach FINISH (pc=-140, br=-142, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-140, br=-142, dir=1)) | OK | ERROR(fwd failed) |
| loop-do | pendulum-cf | OK | OK | OK | OK |
| loop | faithful | ERROR(phpisa: did not reach FINISH (pc=-140, br=-142, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-140, br=-142, dir=1)) | OK | ERROR(fwd failed) |
| loop | pendulum-cf | OK | OK | OK | OK |
| modminus | faithful | ERROR(phpisa: did not reach FINISH (pc=-10, br=-12, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-10, br=-12, dir=1)) | OK | ERROR(fwd failed) |
| modminus | pendulum-cf | OK | OK | OK | OK |
| reminder | faithful | ERROR(phpisa: did not reach FINISH (pc=-10, br=-12, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-10, br=-12, dir=1)) | OK | ERROR(fwd failed) |
| reminder | pendulum-cf | OK | OK | OK | OK |
| s1-negative | faithful | ERROR(phpisa: did not reach FINISH (pc=-47, br=-51, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-47, br=-51, dir=1)) | OK | ERROR(fwd failed) |
| s1-negative | pendulum-cf | OK | OK | OK | OK |
| s10-loop-assert | faithful | ERROR(phpisa: did not reach FINISH (pc=-153, br=-156, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-160, br=-163, dir=1)) | MISMATCH(mem[1]=3 expected 0) | ERROR(fwd failed) |
| s10-loop-assert | pendulum-cf | ERROR(phpisa: did not reach FINISH (pc=-55, br=-105, dir=1)) | OK | MISMATCH(mem[1]=3 expected 0) | ERROR(fwd failed) |
| s2-xor-negative | faithful | ERROR(phpisa: did not reach FINISH (pc=-71, br=-75, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-71, br=-75, dir=1)) | OK | ERROR(fwd failed) |
| s2-xor-negative | pendulum-cf | OK | OK | OK | OK |
| s3-overflow32 | faithful | ERROR(phpisa: did not reach FINISH (pc=-33, br=-36, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-33, br=-36, dir=1)) | OK | ERROR(fwd failed) |
| s3-overflow32 | pendulum-cf | OK | OK | OK | OK |
| s3-overflow63 | faithful | ERROR(phpisa: did not reach FINISH (pc=-20, br=-23, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-20, br=-23, dir=1)) | OK | ERROR(fwd failed) |
| s3-overflow63 | pendulum-cf | MISMATCH(mem[0]: pisa_interp=9223372036854775808 phpisa=9223372036854… | OK | OK | MISMATCH(mem[0]=9223372036854775806 expected 9223372036854775807) |
| s4-large-imm | faithful | ERROR(phpisa: did not reach FINISH (pc=-84, br=-88, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-84, br=-88, dir=1)) | OK | ERROR(fwd failed) |
| s4-large-imm | pendulum-cf | OK | OK | OK | OK |
| s5-nested-loops | faithful | ERROR(phpisa: did not reach FINISH (pc=-316, br=-320, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-322, br=-326, dir=1)) | OK | ERROR(fwd failed) |
| s5-nested-loops | pendulum-cf | OK | OK | OK | OK |
| s6-uncall | faithful | ERROR(phpisa: did not reach FINISH (pc=-9, br=-34, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-9, br=-34, dir=1)) | OK | ERROR(fwd failed) |
| s6-uncall | pendulum-cf | OK | OK | OK | OK |
| s7-call-chain | faithful | ERROR(phpisa: did not reach FINISH (pc=-15, br=-17, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-33, br=-35, dir=1)) | OK | ERROR(fwd failed) |
| s7-call-chain | pendulum-cf | OK | OK | OK | OK |
| s8-compare-negative | faithful | ERROR(phpisa: did not reach FINISH (pc=-242, br=-249, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-242, br=-249, dir=1)) | OK | ERROR(fwd failed) |
| s8-compare-negative | pendulum-cf | OK | OK | OK | OK |
| s9-self-assign | faithful | ERROR(phpisa: did not reach FINISH (pc=-10, br=-12, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-10, br=-12, dir=1)) | MISMATCH(mem[0]=0 expected 21) | ERROR(fwd failed) |
| s9-self-assign | pendulum-cf | OK | OK | MISMATCH(mem[0]=0 expected 21) | MISMATCH(mem[0]=0 expected 21) |
| show-array | faithful | ERROR(phpisa: did not reach FINISH (pc=-14, br=-18, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-14, br=-18, dir=1)) | OK | ERROR(fwd failed) |
| show-array | pendulum-cf | OK | OK | OK | OK |
| show-stack | faithful | SKIPPED: stacks (`stack s`, `show(s)`) are not part of pyjanus2pisa's Janus di… | | | |
| show-stack | pendulum-cf | SKIPPED: stacks (`stack s`, `show(s)`) are not part of pyjanus2pisa's Janus di… | | | |
| show | faithful | ERROR(phpisa: did not reach FINISH (pc=-10, br=-12, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-10, br=-12, dir=1)) | OK | ERROR(fwd failed) |
| show | pendulum-cf | OK | OK | OK | OK |
| swap-array | faithful | ERROR(phpisa: did not reach FINISH (pc=-47, br=-54, dir=1)) | ERROR(phpisa: did not reach FINISH (pc=-47, br=-54, dir=1)) | OK | ERROR(fwd failed) |
| swap-array | pendulum-cf | OK | OK | OK | OK |
| swap-stack | faithful | SKIPPED: stacks (`stack a`, `push(x, a)`, `a <=> b` on stacks) are not part of… | | | |
| swap-stack | pendulum-cf | SKIPPED: stacks (`stack a`, `push(x, a)`, `a <=> b` on stacks) are not part of… | | | |

RFCL (`rl-to-pisa` → PAL; oracle `rl-run`; inputs 3 [,1]):
| program | phpisa vs rl-run | pisa_interp vs rl-run | notes |
|---|---|---|---|
| copy | OK | OK | inputs x=3 |
| fib_bennett | ERROR(phpisa: did not reach FINISH (pc=-1, br=-2, dir=1)) | ERROR(pisa_interp: Exceeded max_steps=5000000. Possible infinite loop… | inputs n=3; EXCH r3 r4 (rfcl register swap): expanded to an XOR triple |
| hand_copy_multi | OK | OK | inputs x=3 |
| hand_countdown | ERROR(phpisa: did not reach FINISH (pc=-1, br=-2, dir=1)) | OK | inputs n=3 |
| hand_if_multi | ERROR(phpisa: did not reach FINISH (pc=10, br=5, dir=1)) | OK | inputs x=3, flag=1 |
| rgoto_copy | ERROR(phpisa: did not reach FINISH (pc=16, br=6, dir=1)) | ERROR(pisa_interp: Exceeded max_steps=5000000. Possible infinite loop… | inputs x=3 |

Rlang-compiler `test/sch/sch.pal` (Schroedinger simulator, 830 lines,
1000 outer iterations):

| interpreter | result |
|---|---|
| phpisa | assembles and runs; 2056 `OUTPUT` lines after 3,000,000 steps; **48,316** `OUTPUT` lines after 60,000,000 steps (3 min 26 s, still not finished — about 94 of the 1000 outer iterations, so the full run needs ≈ 650M steps / 35 min); values stay plausible (`R27` = PSIR samples become negative as the wave evolves, `R28`/`R29` = loop indices) |
| pisa_interp via `pal2pisa` | loads (label immediates `ADDI $3 -PSIR`, `.START SCHROED`, `OUTPUT`, `RL/RR/SRLVX` handled by the loader/PALMachine); **0** `OUTPUT` lines, spins at pc=711 until `max_steps`: the Pendulum subroutine protocol (`BRA PRINTWAVE` → `PRINTWAVE: SWAPBR $2`, `RBRA HALFSTEP`) is not understood (discrepancies 2 and 3) |
| `output.txt` | contains **no** run output (it is the compiler's listing: source, unoptimised PAL, environment); there is no oracle for the numbers phpisa prints |

Constructs of sch.pal that each side lacks: `pisa.py`/`PISAMachine` – `.START`,
`START`-as-entry, label immediates (incl. `-LABEL`), `OUTPUT`, `RL`, `RR`,
`SRLVX`, `BGTZ/BLEZ/BLTZ` (the loader supplies all but the last three);
phpisa – nothing (it was written for this dialect).

## DISCREPANCY LIST

Classes: (a) Pendulum semantics ambiguity, (b) implementation bug, (c) adapter
limitation.  Reproducers live in the text below and were executed on both
sides (`tools/pisa2pal.py` + `tools/phpisa_dump.php` / `PISAMachine`).

**D1 (b, pyjanus2pisa codegen) – emitted code is not executable on a Pendulum
machine.**  `codegen.py` places the branch target of every call on a data
instruction (`f: SUBI r1 1`, `codegen.py:1141`) and joins `if`/`from` control
flow with jumps to data instructions (`BRA if_assert_K`, `BRA from_do_Z`,
`BEQ rt r0 from_loop_L`, `BRA from_exit_E`; `codegen.py:1000-1120`).  Only
the interpreter's own "simple direct jump" rule (`pisa_interp.py:23-30,
345-346`) makes this work; under `PC += BR` (`runner.php:60`,
`INSTRUCTIONS.md:74`) the branch register stays non-zero and the machine
jumps again.  Effect: in faithful mode **all 25** assembled corpus programs
run off the program on phpisa (`pc<0`, `finished=false`), e.g. `modminus`
stops at `pc=-10, br=-12`.  Minimal program:
```
start: START
       BRA L
       ADDI r3 1
L:     ADDI r4 1        ; data instruction as branch target
finish: FINISH
```
pisa_interp: FINISH, r4=1.  phpisa: r4=1 then `PC += BR` → pc=5, never
reaches FINISH.  The `--pendulum-cf` lowering shows what the compiler should
emit: with it 23/25 programs agree on both interpreters.

**D2 (b, pisa_interp; also an (a) item for the compiler) – `RBRA` is `BRA`.**
`pisa_interp.py:297` handles `(BRA, RBRA)` identically; phpisa flips `DIR`
and executes reverse templates (`asm_special.php:23-28`).  The README admits
the interpreter "has no Pendulum direction bit" and the compiler avoids RBRA
by emitting inverted companion procedures.  Minimal program:
```
start: START
       ADDI r3 1
A:     RBRA B
       ADDI r3 10        ; executed BACKWARDS by phpisa: r3 -= 10
B:     BRA A
finish: FINISH
```
pisa_interp: r3 = 1 (A/B are treated as a paired BRA).  phpisa: r3 = -9.
Consequence: rfcl's `rgoto` programs (`fib_bennett.rl`, `rgoto_copy.rl`)
loop forever on pisa_interp; sch.pal's `RBRA HALFSTEP` cannot run.

**D3 (b, pisa_interp) – the Pendulum subroutine protocol does not run.**
Calls are recognised only for labels with `_top`/`_bot` companions and are
done through a software stack (`pisa_interp.py:308-328`); a textbook
`BRA f` … `f: SWAPBR $2` … `_SUBBOT: BRA _SUBTOP` returns into the body again.
Reproducer: `python3 tools/pal2pisa.py ../phpisa/samples/mult.pal` loops until
`max_steps` (r1 counts up), while `php bin/phpisa samples/mult.pal` prints
`R5 = 3, 9, 9, 3`.  Same cause for sch.pal's 0 outputs.

**D4 (a) – register width.**  pisa_interp: unbounded; phpisa: PHP 64-bit int
that becomes a float on overflow (`r_ADD` does not mask); neither is the
32-bit machine word of Pendulum (only phpisa's rotates/`SRLX` are 32-bit).
`s3-overflow63` (`int x = 9223372036854775807; x += 1; y += x`):
pisa_interp mem[0] = 9223372036854775808, phpisa mem[0] =
9.2233720368547758E+18 → `MISMATCH(mem[0])`; round trip on phpisa fails too.
`s3-overflow32` (past 2^31/2^32) agrees on both — and would wrap on real
hardware.  Ties in with the fixed-width Rocq work (`RevSMod` window).

**D5 (a) encoding / (b) missing check, phpisa – 11-bit immediates and
offsets are truncated silently.**  `e_ADDI` keeps `imm & 0x7FF`
(`enc_arith_log.php`), `e_BRA` likewise (`enc_branch.php:5`), decoded by
`sign_extend11`.  Reproducer (PAL): `ADDI $3 1024` → R3 = **-1024**;
`XORI $4 2047` → R4 = **-1**; pisa_interp gives 1024 / 2047.  The adapter
expands constants (§ Adapter) and refuses far branches: `j1-sort`
(1084 instructions between `main_top` and `main_bot`) cannot be assembled for
phpisa → the only corpus ERROR in pendulum-cf mode (class (c) for the
harness).

**D6 (a) – `r0`.**  Hard-wired zero in pisa_interp (`pisa_interp.py:186-193`),
ordinary register in phpisa (`set_reg`, `php_functions.php:8`).  PAL
`ADDI $0 5; ADD $3 $0` gives R0 = 5, R3 = 5 on phpisa; pisa_interp r3 = 0.
janus2pisa relies on r0 = 0 (`BEQ rt r0 …`) and only writes it in the
`ADDI r0 0` NOP, so the adapter accepts exactly that write.

**D7 (a) – memory model.**  phpisa: memory = program array
(`php_functions.php:13-25`, `INSTRUCTIONS.md:104`); pisa_interp: separate
memory, only the *leading* `DATA` run is loaded (`pisa_interp.py:130-137`).
Reproducers (PAL): `ADDI $3 1; EXCH $4 $3` → phpisa R4 = 2299905 (the
encoded `ADDI` word at line 1), pisa_interp 0; `ADDI $3 D; EXCH $4 $3; FINISH;
D: DATA 7` → phpisa R4 = 7, bare PISAMachine 0 (`pal2pisa` preloads every
DATA line to compensate).  Consequence for janus2pisa: the stack at
`nvars+3` would overwrite instructions on phpisa (the adapter relocates it).

**D8 (b, pyjanus2pisa codegen) – `from` assertions are not checked.**  The
entry and re-entry assertions are evaluated and then cleared with
`XOR rt rt` "safe for non-boolean" (`codegen.py:1066, 1091, 1109`) instead of
being verified like `if` assertions are.  `s10-loop-assert`
(`from y = 0 do x += 1 loop skip until x = 3`, y never changes): pisa_interp
runs to x = 3 with no error and the round trip fails (`rt-pisa MISMATCH
mem[1]=3 expected 0`); PyJanus reports the violation.  Under Pendulum
semantics the violated assertion leaves BR ≠ 0 and phpisa runs off the
program.

**D9 (b, pyjanus2pisa) – `x += x` is accepted.**  Janus forbids the assigned
variable on the right-hand side; the "self-referencing assignment
optimization" compiles it, and `x -= x` is not its inverse.
`s9-self-assign` (`int x = 21; x += x`): fwd/bwd agree on both interpreters
(x = 42), round trip gives 0 on both.

**D10 (b, rfcl `rl-to-pisa`) – the emitted text is not PISA.**  Block entries
are data instructions reached by `BRA` (same defect as D1), `EXCH Ri, Rj` is
a *register* swap (PISA's EXCH is register↔memory), `BLT/BGT` and inline
expressions `(R1 + #2)` have no PISA equivalent, `fi`/`from` assertions are
comments only (`pyrev_fl/pisa.py`).  Results: `copy`, `hand_copy_multi`
agree with `rl-run` on both interpreters; `hand_countdown`, `hand_if_multi`
agree on pisa_interp (direct jumps) but run off on phpisa; `fib_bennett`,
`rgoto_copy` fail on both (phpisa: D1-type jump; pisa_interp: D2).

**D11 (c) – adapter limitations.**  `SLTX`, `ORX`, `ANDX`, `SUBI` are
pyjanus2pisa pseudo-instructions; their expansions are exact but the SLTX
idiom uses a conditional-branch pair that pisa_interp would execute as a
direct jump (so PAL produced from SLTX programs cannot be fed back to
PISAMachine); constants ≥ 2^32 and branches beyond ±1024 lines are refused;
`--pendulum-cf` recognises exactly the shapes `_gen_if`/`_gen_from` emit
today (an unrecognised shape is reported as `ERROR(adapter: pendulum-cf …)`).

Totals: 11 items — (a) 4 (D4, D6, D7, D5-encoding), (b) 6 (D1, D2, D3, D8,
D9, D10; D5's silent truncation), (c) 1 (D11; plus D5's effect on j1-sort).

## RESUME

Done (day 1): adapter (`tools/pisa2pal.py`, both dialects, two lowering
modes), PHP state dumper (`tools/phpisa_dump.php`), PAL loader + extended
machine (`tools/pal2pisa.py`), harness (`tools/difftest_pisa.py`: janus /
rfcl / pal), 31-file corpus, 21 tests (`test_difftest.py`, php tests skip
without php/phpisa; CI clones phpisa), this document.

sch.pal on phpisa: 60M steps (3.5 min) give 48,316 output lines and no
FINISH; the complete run needs ≈ 650M steps
(`php tools/phpisa_dump.php --phpisa ../phpisa --max-steps 700000000 …/sch.pal`,
≈ 40 min) — not done yet.

Next:

* Day 2 – make `codegen.py` emit Pendulum-valid code natively (port the
  `--pendulum-calls`/`--pendulum-cf` rewrites: `f: SWAPBR` first, cancelling
  branches at joins, stack above code or a `.data` area) and add a strict
  Pendulum mode to `pisa_interp.py` (`PC += BR`, direction bit, reverse
  templates).  Then `--mode faithful` should be all-OK and
  `pal2pisa ../phpisa/samples/mult.pal` should print `R5 = 3 9 9 3`.
  Check D8 (verify `from` assertions instead of clearing rt) and D9 (reject
  `x op= e` with `x ∈ vars(e)`).
* Day 3 – fixed-width registers: `--width 32` in both interpreters and the
  adapter (worker C's `RevSMod` window); rerun `s3-*`; add wrap-around
  programs (`x += 2147483647; x += 1` should give -2147483648).
* Day 4 – rfcl: propose a PAL backend for `rl-to-pisa` (block entry via
  `BRA`/`SWAPBR`-free paired branches, `EXCH` → XOR triple, `BLT` → SLTX-style
  BGEZ pair); run all 6 examples on several input vectors
  (`--inputs 5,0`, `--inputs 0,1`).
* Day 5 – sch.pal: full run on phpisa (docker `php:8.4-cli-alpine`, or raise
  `max_steps`), compare the 3 × 128 `OUTPUT` values with the C version
  `test/sch/Cversion-schii`; run on the strict pisa_interp mode from day 2.
* Day 6 – random Janus program generator (straight-line + if/from, small
  constants and 2^31 boundaries) feeding the harness; minimise any new
  mismatch with the `--keep` files.
* Day 7 – write-up; file the (b) items upstream (pyjanus2pisa: D1, D8, D9;
  phpisa: D5 range check; rfcl: D10).

Exact commands are in "How to run"; kept intermediate files of the last run:
`--keep DIR` writes `<name>.fwd.pisa`, `<name>.inv.pisa`,
`<name>.<mode>.{fwd,inv,rt}.pal`.
