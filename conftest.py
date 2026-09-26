"""Session-wide check: every program compiled by the test suite is well formed.

Wraps `CodeGen.gen_program` (unoptimised code) and the last pass of
`compile_program` (optimised code) so that *any* test that compiles a Janus
program also asserts `pisa.is_wf` on every emitted instruction.  This is
independent of `codegen.check_wf` (which compile_program also runs), so
removing that check does not remove this one.
"""

import codegen
from pisa import is_wf, format_instr

WF_STATS = {"unoptimised": 0, "optimised": 0, "instructions": 0}

_orig_gen_program = codegen.CodeGen.gen_program
_orig_last_pass = codegen.remove_unused_labels


def _assert_wf(code, stage):
    bad = [f"#{i} {format_instr(li.instr)}" for i, li in enumerate(code)
           if not is_wf(li.instr)]
    assert not bad, f"{stage} code not locally invertible: {', '.join(bad[:5])}"
    WF_STATS[stage] += 1
    WF_STATS["instructions"] += len(code)


def _gen_program(self, prog):
    code = _orig_gen_program(self, prog)
    _assert_wf(code, "unoptimised")
    return code


def _last_pass(code):
    out = _orig_last_pass(code)
    _assert_wf(out, "optimised")
    return out


codegen.CodeGen.gen_program = _gen_program
codegen.remove_unused_labels = _last_pass


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        f"is_wf checked on {WF_STATS['unoptimised']} unoptimised and "
        f"{WF_STATS['optimised']} optimised programs "
        f"({WF_STATS['instructions']} instructions)")
