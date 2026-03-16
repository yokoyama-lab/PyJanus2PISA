"""Register allocator for PISA code generation.

Implements the 3-category register management from Section 4.5 of the paper:
- Free registers: zero-cleared, available for use
- Commit registers: hold values needed later
- Garbage registers: hold unneeded values, not yet cleared

Registers:
  r0  = constant 0
  r1  = rsp (stack pointer)
  r2  = rro (return offset)
  r3..r31 = general purpose
"""


class RegAllocError(Exception):
    pass


class RegAlloc:
    NUM_REGS = 32  # r0..r31
    RESERVED = {"r0", "r1", "r2"}  # r0=zero, r1=sp, r2=rro

    def __init__(self):
        self.free = set(f"r{i}" for i in range(3, self.NUM_REGS))
        self.commit = set()
        self.garbage = set()
        self._spill_code = []  # collects spill instructions when needed

    def alloc(self) -> str:
        """Allocate a free register. Raises if none available."""
        if self.free:
            reg = min(self.free, key=lambda r: int(r[1:]))
            self.free.remove(reg)
            return reg
        raise RegAllocError("No free registers available")

    def commit_reg(self, reg: str):
        """Mark register as committed (holds a needed value)."""
        self.commit.add(reg)

    def to_garbage(self, reg: str):
        """Mark a committed register as garbage (value no longer needed)."""
        self.commit.discard(reg)
        self.garbage.add(reg)

    def free_reg(self, reg: str):
        """Return a zeroed register to the free pool."""
        self.commit.discard(reg)
        self.garbage.discard(reg)
        self.free.add(reg)

    def is_free(self, reg: str) -> bool:
        return reg in self.free

    def is_commit(self, reg: str) -> bool:
        return reg in self.commit

    def is_garbage(self, reg: str) -> bool:
        return reg in self.garbage
