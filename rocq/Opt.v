(** * Opt.v — the optimizer passes preserve the machine semantics

    `codegen.py` post-processes emitted code with `peephole` (cancel adjacent
    mutual-inverse pairs, iterated to a fixed point) and `remove_nops` (drop
    zero-constant immediates).  This file proves both passes preserve `run` on
    the straight-line fragment, for ALL code — not only compiler output.

    The label-handling parts of the Python passes (label preservation in
    `_peephole_pass`, `remove_unused_labels`) have no counterpart here because
    the model has no labels; they are future work along with control flow.

    A soundness subtlety, found while writing this proof: an *aliased* register
    pair such as [XOR r r ; XOR r r] is NOT a mutual-inverse pair (the first
    clears r, the second is a no-op on 0 — net effect r := 0, not identity),
    yet the Python `_cancels` deleted it.  [cancels] below requires distinct
    operand registers, and `_cancels` was fixed to match. *)

From Stdlib Require Import ZArith List Lia Bool.
Require Import PISA.
Import ListNotations.
Open Scope Z_scope.

(** ** Which adjacent pairs cancel (mirror of the fixed `_cancels`) *)

Definition cancels (a b : instr) : bool :=
  match a, b with
  | IAddi r c,  ISubi r' c'  => Nat.eqb r r' && Z.eqb c c'
  | ISubi r c,  IAddi r' c'  => Nat.eqb r r' && Z.eqb c c'
  | IXori r c,  IXori r' c'  => Nat.eqb r r' && Z.eqb c c'
  | INeg r,     INeg r'      => Nat.eqb r r'
  | IAdd rd rs, ISub rd' rs' =>
      Nat.eqb rd rd' && Nat.eqb rs rs' && negb (Nat.eqb rd rs)
  | ISub rd rs, IAdd rd' rs' =>
      Nat.eqb rd rd' && Nat.eqb rs rs' && negb (Nat.eqb rd rs)
  | IXor rd rs, IXor rd' rs' =>
      Nat.eqb rd rd' && Nat.eqb rs rs' && negb (Nat.eqb rd rs)
  | IExch rd ra, IExch rd' ra' =>
      Nat.eqb rd rd' && Nat.eqb ra ra' && negb (Nat.eqb rd ra)
  | _, _ => false
  end.

(** A cancelling pair is exactly "a well-formed instruction followed by its
    inverse" — so [step_invert] does all the semantic work. *)
Lemma cancels_spec : forall a b,
  cancels a b = true -> b = invert_instr a /\ wf_instr a.
Proof.
  intros a b H; destruct a; destruct b; simpl in H; try discriminate;
  repeat (apply andb_prop in H as [H ?]);
  repeat match goal with
  | H : Nat.eqb _ _ = true |- _ => apply Nat.eqb_eq in H; subst
  | H : Z.eqb _ _ = true |- _ => apply Z.eqb_eq in H; subst
  | H : negb (Nat.eqb ?x ?y) = true |- _ =>
      apply negb_true_iff, Nat.eqb_neq in H
  end; simpl; auto.
Qed.

Theorem cancels_undo : forall a b s,
  cancels a b = true -> step b (step a s) = s.
Proof.
  intros a b s H; destruct (cancels_spec a b H) as [-> Hwf].
  now apply step_invert.
Qed.

(** ** One peephole pass *)

Fixpoint peephole_pass (c : code) : code :=
  match c with
  | a :: ((b :: t) as rest) =>
      if cancels a b then peephole_pass t else a :: peephole_pass rest
  | _ => c
  end.

(** The pass never grows the code (needed for fixed-point termination). *)
Lemma peephole_pass_len : forall c, (length (peephole_pass c) <= length c)%nat.
Proof.
  intros c; remember (length c) as n eqn:Hn; revert c Hn.
  induction n as [n IHn] using lt_wf_ind; intros c Hn; subst.
  destruct c as [| a [| b t]]; simpl; try lia.
  destruct (cancels a b).
  - specialize (IHn (length t) ltac:(simpl; lia) t eq_refl); lia.
  - specialize (IHn (length (b :: t)) ltac:(simpl; lia) (b :: t) eq_refl);
    simpl in *; lia.
Qed.

Theorem peephole_pass_run : forall c s, run (peephole_pass c) s = run c s.
Proof.
  (* strong induction on length: the recursive call skips two elements *)
  intros c; remember (length c) as n eqn:Hn; revert c Hn.
  induction n as [n IHn] using lt_wf_ind; intros c Hn s; subst.
  destruct c as [| a [| b t]]; cbn [peephole_pass]; try reflexivity.
  destruct (cancels a b) eqn:Hc.
  - rewrite !run_cons.
    rewrite (IHn (length t) ltac:(simpl; lia) t eq_refl).
    now rewrite cancels_undo.
  - rewrite !run_cons.
    apply (IHn (length (b :: t)) ltac:(simpl; lia) (b :: t) eq_refl).
Qed.

(** ** Fixed-point iteration (mirror of `peephole`) *)

Fixpoint peephole_iter (fuel : nat) (c : code) : code :=
  match fuel with
  | O => c
  | S f =>
      let c' := peephole_pass c in
      if Nat.eqb (length c') (length c) then c' else peephole_iter f c'
  end.

Definition peephole (c : code) : code := peephole_iter (length c) c.

Theorem peephole_iter_run : forall fuel c s, run (peephole_iter fuel c) s = run c s.
Proof.
  induction fuel as [| f IH]; intros c s; simpl.
  - reflexivity.
  - destruct (Nat.eqb (length (peephole_pass c)) (length c)).
    + apply peephole_pass_run.
    + rewrite IH. apply peephole_pass_run.
Qed.

Corollary peephole_run : forall c s, run (peephole c) s = run c s.
Proof. intros; apply peephole_iter_run. Qed.

(** ** NOP removal *)

Definition is_nop (i : instr) : bool :=
  match i with
  | IAddi _ c | ISubi _ c | IXori _ c => Z.eqb c 0
  | _ => false
  end.

Lemma nop_step : forall i s, is_nop i = true -> step i s = s.
Proof.
  intros i s H; destruct i; simpl in H; try discriminate;
  apply Z.eqb_eq in H; subst; destruct s as [R M]; simpl.
  - replace (R rd + 0) with (R rd) by ring. now rewrite rupd_id.
  - replace (R rd - 0) with (R rd) by ring. now rewrite rupd_id.
  - rewrite Z.lxor_0_r. now rewrite rupd_id.
Qed.

Definition remove_nops (c : code) : code :=
  filter (fun i => negb (is_nop i)) c.

Theorem remove_nops_run : forall c s, run (remove_nops c) s = run c s.
Proof.
  induction c as [| i c IH]; intros s; unfold remove_nops; cbn [filter].
  - reflexivity.
  - destruct (is_nop i) eqn:Hn; cbn [negb].
    + rewrite (run_cons i c), nop_step by assumption. apply IH.
    + rewrite !run_cons. apply IH.
Qed.

(** ** The full pipeline, as `compile_program` runs it *)

Definition optimize (c : code) : code := remove_nops (peephole c).

Theorem optimize_run : forall c s, run (optimize c) s = run c s.
Proof. intros; unfold optimize; now rewrite remove_nops_run, peephole_run. Qed.

(** ** Sanity checks *)

Example ex_cancel_pair :
  peephole [IAddi 3%nat 5; ISubi 3%nat 5] = [].
Proof. reflexivity. Qed.

Example ex_cascade :
  (* inner pair cancels first, exposing the outer pair on the next iteration *)
  peephole [IAdd 3%nat 4%nat; IAddi 5%nat 1; ISubi 5%nat 1; ISub 3%nat 4%nat] = [].
Proof. reflexivity. Qed.

Example ex_aliased_xor_kept :
  (* the pair that _cancels wrongly deleted: must be preserved *)
  peephole [IXor 3%nat 3%nat; IXor 3%nat 3%nat]
  = [IXor 3%nat 3%nat; IXor 3%nat 3%nat].
Proof. reflexivity. Qed.

Example ex_nops_removed :
  remove_nops [IAddi 3%nat 0; IAdd 3%nat 4%nat; IXori 5%nat 0]
  = [IAdd 3%nat 4%nat].
Proof. reflexivity. Qed.

Print Assumptions optimize_run.
