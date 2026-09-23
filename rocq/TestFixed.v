(** * TestFixed.v — Test.v on the fixed-width machine, at 8 and 3 bits

    EXPERIMENT (see FIXED_WIDTH_REPORT.md).  The concrete program of Test.v
    is run on PISAFixed at b = 8 (PyJanus's [-m 8] width): all values fit,
    so every example of Test.v holds by computation.  At b = 3 (window
    [-4,4)) the intermediate value 5 overflows and the results differ from
    Janus-over-Z — the machine is still reversible, but it no longer
    computes the Janus function. *)

From Stdlib Require Import ZArith List Lia.
Require Import RevSMod PISAFixed Src CompileFixed.
Import ListNotations.
Open Scope Z_scope.

(** Same program as Test.v: x += 3; y += x + 2; x <=> y. *)
Definition prog : stmt :=
  Seq (Assign 0%nat AAdd (Cst 3))
      (Seq (Assign 1%nat AAdd (Bin OAdd (Var 0%nat) (Cst 2)))
           (Swap 0%nat 1%nat)).

(** ** 8 bits: everything of Test.v holds unchanged *)

Definition final8 : state := run 8 (compile prog) zero_state.

Example prog_x_8 : mem final8 (Z.of_nat 0) = 5.
Proof. reflexivity. Qed.

Example prog_y_8 : mem final8 (Z.of_nat 1) = 3.
Proof. reflexivity. Qed.

Example prog_clean_3_8 : regs final8 3%nat = 0.
Proof. reflexivity. Qed.

Example prog_clean_4_8 : regs final8 4%nat = 0.
Proof. reflexivity. Qed.

Example prog_clean_5_8 : regs final8 5%nat = 0.
Proof. reflexivity. Qed.

Example prog_clean_6_8 : regs final8 6%nat = 0.
Proof. reflexivity. Qed.

Example prog_wf : wf_stmt prog.
Proof. cbn; repeat split; lia. Qed.

(** NEW obligation: the variables live at representable addresses. *)
Example prog_vars_ok_8 : stmt_vars_ok 8 prog.
Proof. cbn; unfold RevSMod.in_window, half; cbn; repeat split; lia. Qed.

(* prog_exec / prog_src_x / prog_src_y are source-side and UNCHANGED (Test.v). *)

(** prog_reversible: MODIFIED — needs [wf_state 8 zero_state], which holds. *)
Example prog_reversible_8 : run 8 (invert_code (compile prog)) final8 = zero_state.
Proof. apply compile_reversible; [lia | apply wf_zero_state; lia]. Qed.

(* gen_assign_shape: UNCHANGED (about code) *)
Example gen_assign_shape :
  gen_assign 0%nat AAdd (Cst 7)
  = [ IAddi 3%nat 7
    ; IAddi 4%nat 0; IExch 5%nat 4%nat; IAdd 5%nat 3%nat
    ; IExch 5%nat 4%nat; ISubi 4%nat 0
    ; ISubi 3%nat 7 ].
Proof. reflexivity. Qed.

(** ** 3 bits: the same program, different answer

    Janus over Z: x = 5, y = 3.  On the 3-bit machine [y += x + 2] computes
    3 + 2 = 5, which wraps to -3, and the swap then puts -3 into x. *)

Definition final3 : state := run 3 (compile prog) zero_state.

Example prog_x_3 : mem final3 (Z.of_nat 0) = -3.
Proof. reflexivity. Qed.

Example prog_y_3 : mem final3 (Z.of_nat 1) = 3.
Proof. reflexivity. Qed.

(** Test.v's [prog_x] statement is FALSE at 3 bits ... *)
Example prog_x_3_differs : mem final3 (Z.of_nat 0) <> 5.
Proof. discriminate. Qed.

(** ... but it is exactly the Janus value modulo the window, as
    [compile_spec_w] predicts (5 wraps to -3 at 3 bits). *)
Example prog_x_3_is_wrapped : mem final3 (Z.of_nat 0) = wrap 3 5.
Proof. reflexivity. Qed.

(** ... and the machine is still clean and reversible. *)
Example prog_clean_3_3 : regs final3 3%nat = 0 /\ regs final3 4%nat = 0 /\
                         regs final3 5%nat = 0 /\ regs final3 6%nat = 0.
Proof. repeat split; reflexivity. Qed.

Example prog_reversible_3 : run 3 (invert_code (compile prog)) final3 = zero_state.
Proof. apply compile_reversible; [lia | apply wf_zero_state; lia]. Qed.

(** ** Axiom footprint: still only functional extensionality *)

Print Assumptions compile_spec_w.
Print Assumptions compile_spec_in_window.
Print Assumptions compile_spec_unwrapped_fails.
Print Assumptions compile_reversible.
Print Assumptions step_invert_needs_wf.
