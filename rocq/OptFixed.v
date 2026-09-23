(** * OptFixed.v — Opt.v on the fixed-width machine (PISAFixed)

    EXPERIMENT (see FIXED_WIDTH_REPORT.md).  The optimizer passes are
    width-independent (they rewrite code, not values) and are copied
    verbatim.  What changes is their soundness statements: Opt.v proves
    [run (optimize c) s = run c s] for EVERY state; on the fixed-width
    machine a cancelling pair such as [ADDI r 0 ; SUBI r 0] is a no-op only
    on representable states (PISAFixed.step_invert_needs_wf), so every
    [*_run] theorem gains the hypothesis [wf_state s].  The same holds for
    [nop_step]: [ADDI r 0] normalises r, which is the identity only when r is
    already in the window. *)

From Stdlib Require Import ZArith List Lia Bool.
Require Import RevSMod PISAFixed.
Import ListNotations.
Open Scope Z_scope.

(** ** Width-independent part — verbatim Opt.v *)

Definition cancels (i j : instr) : bool :=
  match i, j with
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

(* cancels_spec: UNCHANGED *)
Lemma cancels_spec : forall i j,
  cancels i j = true -> j = invert_instr i /\ wf_instr i.
Proof.
  intros i j H; destruct i; destruct j; simpl in H; try discriminate;
  repeat (apply andb_prop in H as [H ?]);
  repeat match goal with
  | H : Nat.eqb _ _ = true |- _ => apply Nat.eqb_eq in H; subst
  | H : Z.eqb _ _ = true |- _ => apply Z.eqb_eq in H; subst
  | H : negb (Nat.eqb ?x ?y) = true |- _ =>
      apply negb_true_iff, Nat.eqb_neq in H
  end; simpl; auto.
Qed.

Fixpoint peephole_pass (c : code) : code :=
  match c with
  | i :: ((j :: t) as rest) =>
      if cancels i j then peephole_pass t else i :: peephole_pass rest
  | _ => c
  end.

(* peephole_pass_len: UNCHANGED *)
Lemma peephole_pass_len : forall c, (length (peephole_pass c) <= length c)%nat.
Proof.
  intros c; remember (length c) as n eqn:Hn; revert c Hn.
  induction n as [n IHn] using lt_wf_ind; intros c Hn; subst.
  destruct c as [| i [| j t]]; simpl; try lia.
  destruct (cancels i j).
  - specialize (IHn (length t) ltac:(simpl; lia) t eq_refl); lia.
  - specialize (IHn (length (j :: t)) ltac:(simpl; lia) (j :: t) eq_refl);
    simpl in *; lia.
Qed.

Fixpoint peephole_iter (fuel : nat) (c : code) : code :=
  match fuel with
  | O => c
  | S f =>
      let c' := peephole_pass c in
      if Nat.eqb (length c') (length c) then c' else peephole_iter f c'
  end.

Definition peephole (c : code) : code := peephole_iter (length c) c.

Definition is_nop (i : instr) : bool :=
  match i with
  | IAddi _ c | ISubi _ c | IXori _ c => Z.eqb c 0
  | _ => false
  end.

Definition remove_nops (c : code) : code :=
  filter (fun i => negb (is_nop i)) c.

Definition optimize (c : code) : code := remove_nops (peephole c).

(** ** The width-parameterised part *)

Section Fixed.

Variable b : nat.
Hypothesis Hb : (0 < b)%nat.

Local Notation step := (step b).
Local Notation run := (run b).
Local Notation wf_state := (wf_state b).

(** cancels_undo: MODIFIED — needs [wf_state s] (via step_invert). *)
Theorem cancels_undo : forall i j s,
  wf_state s -> cancels i j = true -> step j (step i s) = s.
Proof.
  intros i j s Hs H; destruct (cancels_spec i j H) as [-> Hwf].
  now apply step_invert.
Qed.

(** peephole_pass_run: MODIFIED — needs [wf_state s]; the non-cancelling
    case additionally threads the invariant through [step_wf]. *)
Theorem peephole_pass_run : forall c s,
  wf_state s -> run (peephole_pass c) s = run c s.
Proof.
  intros c; remember (length c) as n eqn:Hn; revert c Hn.
  induction n as [n IHn] using lt_wf_ind; intros c Hn s Hs; subst.
  destruct c as [| i [| j t]]; cbn [peephole_pass]; try reflexivity.
  destruct (cancels i j) eqn:Hc.
  - rewrite !run_cons.
    rewrite (IHn (length t) ltac:(simpl; lia) t eq_refl s Hs).
    now rewrite (cancels_undo i j s Hs Hc).
  - rewrite !run_cons.
    apply (IHn (length (j :: t)) ltac:(simpl; lia) (j :: t) eq_refl).
    apply step_wf; [exact Hb | exact Hs].
Qed.

(** peephole_iter_run / peephole_run: MODIFIED — needs [wf_state s]. *)
Theorem peephole_iter_run : forall fuel c s,
  wf_state s -> run (peephole_iter fuel c) s = run c s.
Proof.
  induction fuel as [| f IH]; intros c s Hs; simpl.
  - reflexivity.
  - destruct (Nat.eqb (length (peephole_pass c)) (length c)).
    + now apply peephole_pass_run.
    + rewrite IH by exact Hs. now apply peephole_pass_run.
Qed.

Corollary peephole_run : forall c s, wf_state s -> run (peephole c) s = run c s.
Proof. intros; now apply peephole_iter_run. Qed.

(** nop_step: MODIFIED — needs [wf_state s]: [ADDI r 0] normalises r. *)
Lemma nop_step : forall i s, wf_state s -> is_nop i = true -> step i s = s.
Proof.
  intros i s [HR HM] H; destruct i; simpl in H; try discriminate;
  apply Z.eqb_eq in H; subst; destruct s as [R M]; simpl in *.
  - rewrite Z.add_0_r, wrap_id by (exact Hb || apply HR). now rewrite rupd_id.
  - rewrite Z.sub_0_r, wrap_id by (exact Hb || apply HR). now rewrite rupd_id.
  - rewrite Z.lxor_0_r, wrap_id by (exact Hb || apply HR). now rewrite rupd_id.
Qed.

(** remove_nops_run: MODIFIED — needs [wf_state s]. *)
Theorem remove_nops_run : forall c s, wf_state s -> run (remove_nops c) s = run c s.
Proof.
  induction c as [| i c IH]; intros s Hs; unfold remove_nops; cbn [filter].
  - reflexivity.
  - destruct (is_nop i) eqn:Hn; cbn [negb].
    + rewrite (run_cons b i c), nop_step by assumption. now apply IH.
    + rewrite !run_cons. apply IH, step_wf; [exact Hb | exact Hs].
Qed.

(** optimize_run: MODIFIED — needs [wf_state s]. *)
Theorem optimize_run : forall c s, wf_state s -> run (optimize c) s = run c s.
Proof.
  intros; unfold optimize.
  rewrite remove_nops_run by assumption.
  now apply peephole_run.
Qed.

(** Why: the pair [ADDI r0 0 ; SUBI r0 0] is deleted by [peephole_pass]
    but is not a no-op on an unrepresentable state. *)
Theorem peephole_pass_run_needs_wf :
  exists c s, run (peephole_pass c) s <> run c s.
Proof.
  exists [IAddi 0%nat 0; ISubi 0%nat 0], (bad_state b).
  cbn [peephole_pass cancels Nat.eqb Z.eqb andb].
  intro H.
  assert (Hr := f_equal (fun st => regs st 0%nat) H).
  rewrite !run_cons, !run_nil in Hr.
  rewrite addi_subi_bad in Hr by exact Hb.
  unfold bad_state in Hr; simpl in Hr.
  pose proof (half_pos b Hb); lia.
Qed.

End Fixed.

(** ** Sanity checks — UNCHANGED (they are about code, not values) *)

Example ex_cancel_pair :
  peephole [IAddi 3%nat 5; ISubi 3%nat 5] = [].
Proof. reflexivity. Qed.

Example ex_cascade :
  peephole [IAdd 3%nat 4%nat; IAddi 5%nat 1; ISubi 5%nat 1; ISub 3%nat 4%nat] = [].
Proof. reflexivity. Qed.

Example ex_aliased_xor_kept :
  peephole [IXor 3%nat 3%nat; IXor 3%nat 3%nat]
  = [IXor 3%nat 3%nat; IXor 3%nat 3%nat].
Proof. reflexivity. Qed.

Example ex_nops_removed :
  remove_nops [IAddi 3%nat 0; IAdd 3%nat 4%nat; IXori 5%nat 0]
  = [IAdd 3%nat 4%nat].
Proof. reflexivity. Qed.

Print Assumptions optimize_run.
