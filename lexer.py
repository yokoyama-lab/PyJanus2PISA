"""Janus lexer (hand-written tokenizer)."""

from dataclasses import dataclass
from typing import List


@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r})"


KEYWORDS = {
    "procedure", "call", "uncall",
    "if", "then", "else", "fi",
    "from", "do", "loop", "until",
    "skip", "local", "delocal", "print",
    "int",
}

# Multi-character operators (order matters: longest first)
MULTI_OPS = [
    "+=", "-=", "^=", "<=>",
    "<=", ">=", "!=", "&&", "||",
    "//",
]

SINGLE_OPS = set("+-*/^&|<>=()[]{}.,;#%")


class LexError(Exception):
    def __init__(self, msg, line, col):
        super().__init__(f"Lex error at line {line}, col {col}: {msg}")
        self.line = line
        self.col = col


def tokenize(source: str) -> List[Token]:
    """Tokenize Janus source code."""
    tokens = []
    i = 0
    line = 1
    col = 1
    n = len(source)

    while i < n:
        ch = source[i]

        # Newline
        if ch == '\n':
            i += 1
            line += 1
            col = 1
            continue

        # Whitespace
        if ch in ' \t\r':
            i += 1
            col += 1
            continue

        # Line comment: // to end of line
        if i + 1 < n and source[i:i+2] == '//':
            while i < n and source[i] != '\n':
                i += 1
            continue

        # Integer literal
        if ch.isdigit():
            start = i
            start_col = col
            while i < n and source[i].isdigit():
                i += 1
                col += 1
            tokens.append(Token("INT", source[start:i], line, start_col))
            continue

        # Negative integer literal (only if preceded by nothing useful)
        # Handled as unary minus in parser instead

        # Identifier or keyword
        if ch.isalpha() or ch == '_':
            start = i
            start_col = col
            while i < n and (source[i].isalnum() or source[i] == '_'):
                i += 1
                col += 1
            word = source[start:i]
            if word in KEYWORDS:
                tokens.append(Token(word.upper(), word, line, start_col))
            else:
                tokens.append(Token("IDENT", word, line, start_col))
            continue

        # Multi-character operators
        matched = False
        for op in MULTI_OPS:
            if source[i:i+len(op)] == op:
                tokens.append(Token("OP", op, line, col))
                i += len(op)
                col += len(op)
                matched = True
                break
        if matched:
            continue

        # Single-character operators and punctuation
        if ch in SINGLE_OPS:
            tokens.append(Token("OP", ch, line, col))
            i += 1
            col += 1
            continue

        raise LexError(f"Unexpected character {ch!r}", line, col)

    tokens.append(Token("EOF", "", line, col))
    return tokens
