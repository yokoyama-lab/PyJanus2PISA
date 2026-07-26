(** * Extract.v — extract the verified compiler to OCaml

    [Compile.compile] is proved correct in Compile.v but is a *re-implementation*
    of the scheme `codegen.py` uses; the two are otherwise only kept in step by
    hand.  Extracting the verified compiler lets us run it and diff its output
    against the Python one (see ../tools/rocq_diff.py).

    Caveat, deliberately recorded: [ExtrOcamlNatInt] / [ExtrOcamlZInt] realise
    [nat] and [Z] by OCaml's native [int].  That is the usual efficiency
    trade-off, but it is *not* covered by the proofs — the theorems are about
    unbounded [nat]/[Z], so the extracted code inherits them only as long as no
    value overflows a 63-bit int.  Register indices and variable offsets are
    tiny; program constants are the ones to watch. *)

From Stdlib Require Import ZArith List Extraction.
From Stdlib Require Import ExtrOcamlBasic ExtrOcamlNatInt ExtrOcamlZInt.
Require Import PISA Src Compile.

Extraction Language OCaml.

(** Keep the constructors' own names so the driver can pattern-match them. *)
Extraction Inline scratch.

Separate Extraction compile invert_code run zero_state eval.
