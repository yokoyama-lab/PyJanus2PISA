(** * Test.v — executable checks and the axiom footprint

    The theorems in Compile.v are about a symbolic program; these examples
    additionally *run* the compiler and the machine on concrete input, which
    catches definitional mistakes a correctness proof about the wrong compiler
    would not.  Everything here is closed by [reflexivity], so it is checked
    at build time. *)

From Stdlib Require Import ZArith List Lia.
Require Import PISA Src Compile.
Import ListNotations.
Open Scope Z_scope.

(** The same program used by [tools/pyjanus_crosscheck.py]:

      x += 3
      y += x + 2
      x <=> y

    with [x] at variable 0 and [y] at variable 1.  Expected final store:
    [x = 5], [y = 3]. *)
Definition prog : stmt :=
  Seq (Assign 0%nat AAdd (Cst 3))
      (Seq (Assign 1%nat AAdd (Bin OAdd (Var 0%nat) (Cst 2)))
           (Swap 0%nat 1%nat)).

Definition final : state := run (compile prog) zero_state.

(** ** The compiled code computes the right thing *)

Example prog_x : mem final (Z.of_nat 0) = 5.
Proof. reflexivity. Qed.

Example prog_y : mem final (Z.of_nat 1) = 3.
Proof. reflexivity. Qed.

(** ** …and leaves no garbage: every scratch register is back to 0 *)

Example prog_clean_3 : regs final 3%nat = 0.
Proof. reflexivity. Qed.

Example prog_clean_4 : regs final 4%nat = 0.
Proof. reflexivity. Qed.

Example prog_clean_5 : regs final 5%nat = 0.
Proof. reflexivity. Qed.

Example prog_clean_6 : regs final 6%nat = 0.
Proof. reflexivity. Qed.

(** ** The program is well-formed and its source semantics agrees *)

Example prog_wf : wf_stmt prog.
Proof. cbn; repeat split; lia. Qed.

Example prog_exec :
  exec prog empty (sw (update (update empty 0%nat 3) 1%nat 5) 0%nat 1%nat).
Proof.
  unfold prog.
  eapply E_Seq; [apply (E_Assign 0%nat AAdd (Cst 3) empty); reflexivity |].
  eapply E_Seq.
  - apply (E_Assign 1%nat AAdd (Bin OAdd (Var 0%nat) (Cst 2))); reflexivity.
  - apply E_Swap.
Qed.

(** The source semantics gives the same values the machine produced above. *)
Example prog_src_x : sw (update (update empty 0%nat 3) 1%nat 5) 0%nat 1%nat 0%nat = 5.
Proof. reflexivity. Qed.

Example prog_src_y : sw (update (update empty 0%nat 3) 1%nat 5) 0%nat 1%nat 1%nat = 3.
Proof. reflexivity. Qed.

(** ** Machine-level reversibility, on this concrete program *)

Example prog_reversible : run (invert_code (compile prog)) final = zero_state.
Proof. apply compile_reversible. Qed.

(** ** The store block of an assignment really is a paired exchange *)

Example gen_assign_shape :
  gen_assign 0%nat AAdd (Cst 7)
  = [ IAddi 3%nat 7
    ; IAddi 4%nat 0; IExch 5%nat 4%nat; IAdd 5%nat 3%nat
    ; IExch 5%nat 4%nat; ISubi 4%nat 0
    ; ISubi 3%nat 7 ].
Proof. reflexivity. Qed.

(** ** Axiom footprint

    [functional_extensionality_dep] is the only assumption, and it is used
    exclusively to promote pointwise equality of the register file / memory
    (both are higher-order maps [reg -> Z] and [addr -> Z]) to Leibniz
    equality — the same trade-off documented in the R-CORE development.
    Eliminating it would require a first-order representation of the machine
    state.  Everything else is closed. *)

Print Assumptions compile_spec.
Print Assumptions compile_reversible.
Print Assumptions gen_expr_spec.
