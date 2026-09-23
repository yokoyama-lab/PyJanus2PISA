"""Tests for the PISA differential-testing tools (tools/pisa2pal.py,
tools/pal2pisa.py, tools/difftest_pisa.py).

The pure-Python tests always run.  The tests that need `php` and a phpisa
checkout (default ../phpisa, or $PHPISA_DIR) skip cleanly when either is
missing; likewise for an rfcl checkout (../rfcl or $RFCL_DIR).
"""

import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))

import pisa2pal                                        # noqa: E402
import pal2pisa                                        # noqa: E402
import difftest_pisa as dt                             # noqa: E402
from lexer import tokenize                             # noqa: E402
from parser import parse                               # noqa: E402
from codegen import compile_program                    # noqa: E402
from pisa import print_program                         # noqa: E402
from pisa_interp import PISAMachine                    # noqa: E402

CORPUS = os.path.join(HERE, "tests", "difftest_corpus")
PHPISA = dt.DEFAULT_PHPISA
RFCL = dt.DEFAULT_RFCL
have_php = shutil.which("php") is not None and os.path.isfile(os.path.join(PHPISA, "src", "runner.php"))
have_rfcl = os.path.isdir(os.path.join(RFCL, "pyrev_fl"))
needs_php = pytest.mark.skipif(not have_php, reason="php or phpisa checkout not available")


def pal_lines(text: str):
    return [l.split(None, 1)[1] if ":" in l.split()[0] else l.strip()
            for l in text.splitlines() if l.strip() and not l.startswith(";") and ".start" not in l]


# --- pisa2pal: instruction mapping (pure Python) -------------------------------

def test_subi_becomes_addi_negative():
    ir = pisa2pal.parse_pyjanus("start: START\n SUBI r3 5\n FINISH\n")
    assert "ADDI $3 -5" in pisa2pal.convert(ir).pal


def test_orx_andx_expansions_are_exact():
    ir = pisa2pal.parse_pyjanus("start: START\n ORX r3 r4\n ANDX r5 r6 r7\n FINISH\n")
    lines = pal_lines(pisa2pal.convert(ir).pal)
    assert lines[1:4] == ["ANDX $3 $3 $4", "XOR $3 $4", "XOR $4 $4"]
    assert lines[4:6] == ["ANDX $5 $6 $7", "XOR $6 $6"]


def test_sltx_branch_idiom():
    ir = pisa2pal.parse_pyjanus("start: START\n SLTX r3 r4 r5\n FINISH\n")
    pal = pisa2pal.convert(ir).pal
    assert "SUB $4 $5" in pal and "XORI $3 1" in pal and "ADD $4 $5" in pal
    assert pal.count("BGEZ $4") == 2
    with pytest.raises(pisa2pal.ConvertError):
        pisa2pal.convert(ir, emulate_sltx=False)


def test_r0_write_is_rejected_but_nop_allowed():
    ok = pisa2pal.parse_pyjanus("start: START\nL: ADDI r0 0\n FINISH\n")
    assert "ADDI $0 0" in pisa2pal.convert(ok).pal
    with pytest.raises(pisa2pal.ConvertError):
        pisa2pal.convert(pisa2pal.parse_pyjanus("start: START\n ADDI r0 5\n FINISH\n"))


def test_large_immediates_and_branch_range():
    ir = pisa2pal.parse_pyjanus("start: START\n ADDI r3 5000\n ADDI r4 123456789\n XORI r5 -70000\n FINISH\n")
    res = pisa2pal.convert(ir)
    assert res.scratch == 31
    assert "ADDI $3 1023" in res.pal and "RL $31 10" in res.pal and "RR $31 10" in res.pal
    far = "start: START\n BRA L\n" + " ADDI r3 1\n" * 1100 + "L: FINISH\n"
    with pytest.raises(pisa2pal.ConvertError, match="11-bit"):
        pisa2pal.convert(pisa2pal.parse_pyjanus(far))


def test_rfcl_dialect():
    src = "start:\n    XOR R2, R1, R0\n    ADD R2, #1, R0\n    SUB R1, #3, R0\n    EXCH R1, R2\n    BEQ R1, R0, done\n    BRA start\ndone:\n    HALT\n"
    res = pisa2pal.convert(pisa2pal.parse_rfcl(src))
    lines = pal_lines(res.pal)
    assert lines[:4] == ["START", "XOR $2 $1", "ADDI $2 1", "ADDI $1 -3"]
    assert lines[4:7] == ["XOR $1 $2", "XOR $2 $1", "XOR $1 $2"]
    assert "BEQ $1 $0 done" in res.pal and ".start start" in res.pal
    with pytest.raises(pisa2pal.ConvertError):
        pisa2pal.parse_rfcl("s:\n    BLT R1, R2, s\n    HALT\n")


def _compile(path):
    with open(path) as f:
        return compile_program(parse(tokenize(f.read())))


def test_pendulum_lowering_inserts_cancelling_branches():
    code = _compile(os.path.join(CORPUS, "if-thenelse.janus"))
    ir = pisa2pal.parse_pyjanus(print_program(code))
    res = pisa2pal.convert(ir, pendulum_cf=True)
    assert res.cf.ifs == 1 and res.lowering.procedures == ["main"]
    lines = pal_lines(res.pal)
    assert any(l.startswith("BEQ $3 $0 _ifj") for l in lines)
    i = lines.index("SWAPBR $2")
    assert lines[i:i + 4] == ["SWAPBR $2", "NEG $2", "ADDI $1 -1", "EXCH $2 $1"]


@pytest.mark.parametrize("name", ["modminus", "s1-negative", "s2-xor-negative", "s4-large-imm", "swap-array"])
def test_pal_roundtrip_on_pisa_interp(name):
    """PISA -> PAL (faithful) -> pal2pisa -> PISAMachine gives the same store as
    the original program on PISAMachine (checks the instruction mapping,
    including the scratch-register constant building, without PHP)."""
    code = _compile(os.path.join(CORPUS, name + ".janus"))
    m1 = PISAMachine(code); m1.check_clean = False; m1.run()
    pal = pisa2pal.convert(pisa2pal.parse_pyjanus(print_program(code))).pal
    m2, ld = pal2pisa.load(pal)
    m2.run()
    nvars = sum(1 for li in code if type(li.instr).__name__ == "DATA")
    assert [m1.mem.get(a, 0) for a in range(nvars)] == [m2.mem.get(a, 0) for a in range(nvars)]
    assert m1.regs[3:] == m2.regs[3:]


def test_pal2pisa_label_immediates_and_trailing_data():
    pal = "  .start s\ns: START\n ADDI $3 D\n EXCH $4 $3\n ADDI $5 -D\n FINISH\nD: DATA 7\n"
    m, ld = pal2pisa.load(pal)
    m.run()
    assert m.regs[4] == 7 and m.regs[3] == 5 and m.regs[5] == -5
    assert "appended 'start: BRA s'" in ld.notes[0]


# --- the harness against phpisa ---------------------------------------------------

@needs_php
@pytest.mark.parametrize("name", ["call-array", "if-thenelse", "s5-nested-loops", "s6-uncall", "s8-compare-negative"])
def test_harness_agrees_in_pendulum_cf_mode(name):
    r = dt.difftest_janus(os.path.join(CORPUS, name + ".janus"), "pendulum-cf", PHPISA, None, 2_000_000)
    assert r.status == "OK", r.notes
    assert (r.fwd, r.bwd, r.rt_pisa, r.rt_php) == ("OK", "OK", "OK", "OK"), r


@needs_php
def test_harness_faithful_mode_documents_direct_jump_gap():
    r = dt.difftest_janus(os.path.join(CORPUS, "modminus.janus"), "faithful", PHPISA, None, 100_000)
    assert r.fwd.startswith("ERROR(phpisa: did not reach FINISH")
    assert r.rt_pisa == "OK"


@needs_php
def test_harness_reports_register_width_discrepancy():
    r = dt.difftest_janus(os.path.join(CORPUS, "s3-overflow63.janus"), "pendulum-cf", PHPISA, None, 100_000)
    assert r.fwd.startswith("MISMATCH(mem[0]")


@needs_php
@pytest.mark.skipif(not have_rfcl, reason="rfcl checkout not available")
def test_rfcl_copy_agrees_with_rl_run():
    r = dt.difftest_rfcl(os.path.join(RFCL, "examples", "copy.rl"), RFCL, PHPISA, None, None, 100_000)
    assert (r.status, r.fwd, r.bwd) == ("OK", "OK", "OK"), r
