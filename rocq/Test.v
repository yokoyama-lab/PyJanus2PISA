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
Print Assumptions Compile.gen_ungen_spec.
Print Assumptions Compile.ungen_expr_spec.

(** ** Milestone 3: comparisons and [&&] / [||] on the machine

    [x += 3 ; y += 5 ; z += x < y ; w += (x = y) || ((y - 5) && 1)]:
    [z = 1], [w = 0] (since [y - 5 = 0]), registers r3..r12 clean. *)
Definition prog_cmp : stmt :=
  Seq (Assign 0%nat AAdd (Cst 3))
  (Seq (Assign 1%nat AAdd (Cst 5))
  (Seq (Assign 2%nat AAdd (Bin OLt (Var 0%nat) (Var 1%nat)))
       (Assign 3%nat AAdd (Bin OOr (Bin OEq (Var 0%nat) (Var 1%nat))
                                   (Bin OAnd (Bin OSub (Var 1%nat) (Cst 5)) (Cst 1)))))).

Example prog_cmp_run :
  let s := run (compile prog_cmp) zero_state in
  map (mem s) [0; 1; 2; 3] = [3; 5; 1; 0]
  /\ map (regs s) (seq 3 10) = repeat 0 10.
Proof. vm_compute. split; reflexivity. Qed.

(** ** The reversible lowering of docs/EXPR_LOWERING.md, instruction by instruction *)

(** [x < y] into r3: result register first, operands above it, both
    uncomputed right after the [SLTX], right before left. *)
Example gen_lt_shape :
  gen_expr (Bin OLt (Var 0%nat) (Var 1%nat)) 3%nat
  = gen_var 0%nat 4%nat ++ gen_var 1%nat 5%nat ++ [ISltx 3 4 5]%nat
    ++ invert_code (gen_var 1%nat 5%nat) ++ invert_code (gen_var 0%nat 4%nat).
Proof. reflexivity. Qed.

(** [x = y]: two exclusive [SLTX] and [XORI 1] — no extra register, no ORX. *)
Example gen_eq_shape :
  gen_expr (Bin OEq (Var 0%nat) (Var 1%nat)) 3%nat
  = gen_var 0%nat 4%nat ++ gen_var 1%nat 5%nat
    ++ [ISltx 3 4 5; ISltx 3 5 4; IXori 3 1]%nat
    ++ invert_code (gen_var 1%nat 5%nat) ++ invert_code (gen_var 0%nat 4%nat).
Proof. reflexivity. Qed.

(** A constant right operand of [+ - ^] is an immediate ([x - 3]: [SUBI]). *)
Example gen_imm_shape :
  gen_expr (Bin OSub (Var 0%nat) (Cst 3)) 3%nat = gen_var 0%nat 3%nat ++ [ISubi 3%nat 3].
Proof. reflexivity. Qed.

(** [x + y]: in place, only the right operand is uncomputed. *)
Example gen_add_shape :
  gen_expr (Bin OAdd (Var 0%nat) (Var 1%nat)) 3%nat
  = gen_var 0%nat 3%nat ++ gen_var 1%nat 4%nat ++ [IAdd 3 4]%nat
    ++ invert_code (gen_var 1%nat 4%nat).
Proof. reflexivity. Qed.

(** [x != 0]: `_gen_nonzero`; its uncomputation runs the SLTX pair in the
    opposite order. *)
Example gen_nz_shape :
  gen_expr (Bin ONe (Var 0%nat) (Cst 0)) 3%nat
  = gen_var 0%nat 4%nat ++ [ISltx 3 4 0; ISltx 3 0 4]%nat ++ invert_code (gen_var 0%nat 4%nat)
  /\ ungen_expr (Bin ONe (Var 0%nat) (Cst 0)) 3%nat
  = gen_var 0%nat 4%nat ++ [ISltx 3 0 4; ISltx 3 4 0]%nat ++ invert_code (gen_var 0%nat 4%nat).
Proof. split; reflexivity. Qed.

(** [x && 5]: `_logical_operands` makes [x] into [x != 0] and folds [5 != 0]
    to the literal 1; Pendulum [ANDX] combines. *)
Example gen_and_shape :
  gen_expr (Bin OAnd (Var 0%nat) (Cst 5)) 3%nat
  = nz_code (gen_var 0%nat) 4%nat ++ [IAddi 5%nat 1] ++ [IAndx 3 4 5]%nat
    ++ [ISubi 5%nat 1] ++ invert_code (nz_code (gen_var 0%nat) 4%nat).
Proof. reflexivity. Qed.

(** Literal operands are folded ([3 <= 5] is the constant 1). *)
Example gen_fold_shape : gen_expr (Bin OLe (Cst 3) (Cst 5)) 3%nat = [IAddi 3%nat 1].
Proof. reflexivity. Qed.

(** Comparisons and [&&]/[||] are reversible at the instruction level now
    (the former [compile_not_reversible] was this program). *)
Example prog_cmp_reversible :
  run (invert_code (compile prog_cmp)) (run (compile prog_cmp) zero_state) = zero_state.
Proof. apply compile_reversible. Qed.

Example eq_reversible_vm :
  let st := Assign 0%nat AAdd (Bin OEq (Var 1%nat) (Var 2%nat)) in
  let s := run (invert_code (compile st)) (run (compile st) zero_state) in
  map (mem s) [0; 1; 2] = [0; 0; 0] /\ map (regs s) (seq 0 12) = repeat 0 12.
Proof. vm_compute. split; reflexivity. Qed.

Print Assumptions wf_compile.
Print Assumptions wf_gen_expr.
