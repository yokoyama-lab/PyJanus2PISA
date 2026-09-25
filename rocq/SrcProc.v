(** * SrcProc.v — Janus with parameterless procedures: [call f] / [uncall f]

    The source language of CompileLoop.v ([lstmt]) extended with calls of
    global-variable procedures, as in Janus: procedures take no parameters
    and act on the global store; [uncall f] runs the inverse of [f]'s body.
    The rules [EP_Call] / [EP_Uncall] are those of [RevProc.v] in the PyJanus
    development specialised to no parameters ([E_Uncall] there is
    [exec (invert (pbody p))]).  Recursion is allowed: nothing restricts the
    call graph, and every result below is by induction on the derivation.

    Also here: reversibility ([exec_p_rev]), determinism ([exec_p_det]), a
    frame lemma ([exec_p_frame]: a variable no body assigns is left alone),
    and a fuel interpreter [run_p] proved sound for [exec_p] ([run_p_sound]),
    used to state the expected results of the examples in TestProc.v. *)

From Stdlib Require Import ZArith List Lia Bool.
From Stdlib Require Import FunctionalExtensionality.
Require Import Src.
Import ListNotations.
Open Scope Z_scope.

Definition pname := nat.

Inductive pstmt :=
| PBase   (s : stmt)
| PSeq    (a c : pstmt)
| PIf     (e1 : expr) (a c : pstmt) (e2 : expr)   (** [if e1 then a else c fi e2] *)
| PLoop   (e1 : expr) (a c : pstmt) (e2 : expr)   (** [from e1 do a loop c until e2] *)
| PCall   (f : pname)
| PUncall (f : pname).

(** Procedure [f] is the [f]-th body; the name of the entry point is a
    parameter of the whole-program theorem. *)
Definition penv := list pstmt.

Fixpoint invert_p (st : pstmt) : pstmt :=
  match st with
  | PBase s         => PBase (invert s)
  | PSeq a c        => PSeq (invert_p c) (invert_p a)
  | PIf e1 a c e2   => PIf e2 (invert_p a) (invert_p c) e1
  | PLoop e1 a c e2 => PLoop e2 (invert_p a) (invert_p c) e1
  | PCall f         => PUncall f
  | PUncall f       => PCall f
  end.

Inductive exec_p (Γ : penv) : pstmt -> store -> store -> Prop :=
| EP_Base : forall s σ σ', exec s σ σ' -> exec_p Γ (PBase s) σ σ'
| EP_Seq : forall a c σ m σ',
    exec_p Γ a σ m -> exec_p Γ c m σ' -> exec_p Γ (PSeq a c) σ σ'
| EP_IfTrue : forall e1 a c e2 σ σ',
    eval σ e1 <> 0 -> exec_p Γ a σ σ' -> eval σ' e2 <> 0 -> exec_p Γ (PIf e1 a c e2) σ σ'
| EP_IfFalse : forall e1 a c e2 σ σ',
    eval σ e1 = 0 -> exec_p Γ c σ σ' -> eval σ' e2 = 0 -> exec_p Γ (PIf e1 a c e2) σ σ'
| EP_Loop : forall e1 a c e2 σ σ',
    eval σ e1 <> 0 -> lp_p Γ e1 a c e2 σ σ' -> exec_p Γ (PLoop e1 a c e2) σ σ'
| EP_Call : forall f body σ σ',
    nth_error Γ f = Some body -> exec_p Γ body σ σ' -> exec_p Γ (PCall f) σ σ'
| EP_Uncall : forall f body σ σ',
    nth_error Γ f = Some body -> exec_p Γ (invert_p body) σ σ' -> exec_p Γ (PUncall f) σ σ'
with lp_p (Γ : penv) : expr -> pstmt -> pstmt -> expr -> store -> store -> Prop :=
| LPP_One : forall e1 a c e2 σ σ',
    exec_p Γ a σ σ' -> eval σ' e2 <> 0 -> lp_p Γ e1 a c e2 σ σ'
| LPP_More : forall e1 a c e2 σ σ1 σ2 σ',
    exec_p Γ a σ σ1 -> eval σ1 e2 = 0 ->
    exec_p Γ c σ1 σ2 -> eval σ2 e1 = 0 ->
    lp_p Γ e1 a c e2 σ2 σ' -> lp_p Γ e1 a c e2 σ σ'.

Scheme exec_p_mut := Induction for exec_p Sort Prop
  with lp_p_mut   := Induction for lp_p Sort Prop.

(** ** Inversion is an involution *)

Lemma ainv_invol : forall o, ainv (ainv o) = o.
Proof. destruct o; reflexivity. Qed.

Lemma invert_invol : forall s, invert (invert s) = s.
Proof.
  induction s; simpl; try reflexivity.
  - now rewrite ainv_invol.
  - now rewrite IHs1, IHs2.
Qed.

Lemma invert_p_invol : forall st, invert_p (invert_p st) = st.
Proof.
  induction st; simpl; try reflexivity; try congruence.
  now rewrite invert_invol.
Qed.

(** ** Reversibility *)

Lemma lp_p_exit : forall Γ e1 a c e2 σ σ', lp_p Γ e1 a c e2 σ σ' -> eval σ' e2 <> 0.
Proof. intros until σ'; intro H; induction H; assumption. Qed.

Inductive opn_p (Γ : penv) (e1 : expr) (a c : pstmt) (e2 : expr) : store -> store -> Prop :=
| OP_nil  : forall σ, opn_p Γ e1 a c e2 σ σ
| OP_cons : forall σ σ1 σ2 σ',
    exec_p Γ a σ σ1 -> eval σ1 e2 = 0 ->
    exec_p Γ c σ1 σ2 -> eval σ2 e1 = 0 ->
    opn_p Γ e1 a c e2 σ2 σ' -> opn_p Γ e1 a c e2 σ σ'.

Lemma opn_p_snoc : forall Γ e1 a c e2 σ m m1 m2,
  opn_p Γ e1 a c e2 σ m ->
  exec_p Γ a m m1 -> eval m1 e2 = 0 -> exec_p Γ c m1 m2 -> eval m2 e1 = 0 ->
  opn_p Γ e1 a c e2 σ m2.
Proof.
  intros Γ e1 a c e2 σ m m1 m2 H. revert m1 m2.
  induction H; intros m1 m2 Ha He2 Hc He1.
  - eapply OP_cons; eauto. apply OP_nil.
  - eapply OP_cons; eauto.
Qed.

Lemma opn_p_to_lp : forall Γ e1 a c e2 σ m σ',
  opn_p Γ e1 a c e2 σ m -> exec_p Γ a m σ' -> eval σ' e2 <> 0 -> lp_p Γ e1 a c e2 σ σ'.
Proof.
  intros Γ e1 a c e2 σ m σ' H. induction H; intros Ha He.
  - apply LPP_One; assumption.
  - eapply LPP_More; eauto.
Qed.

Theorem exec_p_rev : forall Γ st σ σ', exec_p Γ st σ σ' -> exec_p Γ (invert_p st) σ' σ.
Proof.
  intros Γ st σ σ' H.
  induction H using exec_p_mut
    with (P0 := fun e1 a c e2 σ σ' (_ : lp_p Γ e1 a c e2 σ σ') =>
      exists q, opn_p Γ e2 (invert_p a) (invert_p c) e1 σ' q /\ exec_p Γ (invert_p a) q σ);
    cbn [invert_p].
  - constructor; now apply exec_rev.
  - econstructor; eassumption.
  - apply EP_IfTrue; assumption.
  - apply EP_IfFalse; assumption.
  - destruct IHexec_p as [q [Hopn Hq]].
    apply EP_Loop; [eapply lp_p_exit; eassumption |].
    eapply opn_p_to_lp; eassumption.
  - eapply EP_Uncall; eassumption.
  - eapply EP_Call; [eassumption |]. now rewrite invert_p_invol in IHexec_p.
  - exists σ'. split; [apply OP_nil | assumption].
  - destruct IHexec_p3 as [q [Hopn Hq]].
    exists σ1. split; [| assumption].
    eapply opn_p_snoc; eassumption.
Qed.

(** ** Determinism *)

Lemma exec_det : forall s σ σ1 σ2, exec s σ σ1 -> exec s σ σ2 -> σ1 = σ2.
Proof.
  intros s σ σ1 σ2 H1; revert σ2; induction H1; intros σ2' H2; inversion H2; subst;
    try reflexivity.
  match goal with Ha : exec s1 _ _ |- _ => apply IHexec1 in Ha; subst end.
  now apply IHexec2.
Qed.

Theorem exec_p_det : forall Γ st σ σ1, exec_p Γ st σ σ1 ->
  forall σ2, exec_p Γ st σ σ2 -> σ1 = σ2.
Proof.
  intros Γ st σ σ1 H.
  induction H using exec_p_mut
    with (P0 := fun e1 a c e2 σ σ1 (_ : lp_p Γ e1 a c e2 σ σ1) =>
      forall σ2, lp_p Γ e1 a c e2 σ σ2 -> σ1 = σ2);
    intros σ2' H2; inversion H2; subst.
  - eapply exec_det; eassumption.
  - match goal with Ha : exec_p Γ a σ _ |- _ => apply IHexec_p1 in Ha; subst end.
    now apply IHexec_p2.
  - now apply IHexec_p.
  - contradiction.
  - contradiction.
  - now apply IHexec_p.
  - now apply IHexec_p.
  - match goal with H1 : nth_error Γ f = Some _, H2 : nth_error Γ f = Some _ |- _ =>
      rewrite H1 in H2; injection H2 as <- end.
    now apply IHexec_p.
  - match goal with H1 : nth_error Γ f = Some _, H2 : nth_error Γ f = Some _ |- _ =>
      rewrite H1 in H2; injection H2 as <- end.
    now apply IHexec_p.
  - now apply IHexec_p.
  - match goal with Ha : exec_p Γ a σ _ |- _ => apply IHexec_p in Ha; subst end.
    contradiction.
  - match goal with Ha : exec_p Γ a σ _ |- _ => apply IHexec_p1 in Ha; subst end.
    contradiction.
  - match goal with Ha : exec_p Γ a σ _ |- _ => apply IHexec_p1 in Ha; subst end.
    match goal with
    | IH : forall x, exec_p Γ c ?m x -> ?y = x, Hc : exec_p Γ c ?m ?z |- _ =>
        tryif constr_eq z y then fail else (apply IH in Hc; subst z)
    end.
    now apply IHexec_p3.
Qed.

(** ** Which variables a statement may assign, and the frame lemma *)

Fixpoint smods (s : stmt) (x : var) : bool :=
  match s with
  | Skip         => false
  | Assign y _ _ => Nat.eqb y x
  | Swap y z     => Nat.eqb y x || Nat.eqb z x
  | Seq a b      => smods a x || smods b x
  end.

(** Calls are accounted for by [env_nomod]: a body never assigns [x]. *)
Fixpoint pmods (st : pstmt) (x : var) : bool :=
  match st with
  | PBase s         => smods s x
  | PSeq a c        => pmods a x || pmods c x
  | PIf _ a c _     => pmods a x || pmods c x
  | PLoop _ a c _   => pmods a x || pmods c x
  | PCall _ | PUncall _ => false
  end.

Definition env_nomod (Γ : penv) (x : var) : Prop :=
  forall f body, nth_error Γ f = Some body -> pmods body x = false.

Lemma smods_invert : forall s x, smods (invert s) x = smods s x.
Proof.
  induction s; intros; simpl; try reflexivity; [].
  rewrite IHs1, IHs2; apply orb_comm.
Qed.

Lemma pmods_invert : forall st x, pmods (invert_p st) x = pmods st x.
Proof.
  induction st; intros; simpl; try reflexivity.
  - apply smods_invert.
  - rewrite IHst1, IHst2; apply orb_comm.
  - now rewrite IHst1, IHst2.
  - now rewrite IHst1, IHst2.
Qed.

Lemma exec_frame : forall s σ σ' x, exec s σ σ' -> smods s x = false -> σ' x = σ x.
Proof.
  intros s σ σ' x H; induction H; simpl; intro Hm.
  - reflexivity.
  - apply update_neq. intro E; subst. now rewrite Nat.eqb_refl in Hm.
  - apply orb_false_elim in Hm as [H1 H2]. unfold sw.
    rewrite update_neq by (intro E; subst; now rewrite Nat.eqb_refl in H2).
    apply update_neq. intro E; subst; now rewrite Nat.eqb_refl in H1.
  - apply orb_false_elim in Hm as [H1 H2]. rewrite IHexec2, IHexec1; auto.
Qed.

Theorem exec_p_frame : forall Γ st σ σ' x,
  exec_p Γ st σ σ' -> env_nomod Γ x -> pmods st x = false -> σ' x = σ x.
Proof.
  intros Γ st σ σ' x H Henv.
  induction H using exec_p_mut
    with (P0 := fun e1 a c e2 σ σ' (_ : lp_p Γ e1 a c e2 σ σ') =>
      pmods a x = false -> pmods c x = false -> σ' x = σ x);
    cbn [pmods]; intros;
    repeat match goal with H : (_ || _)%bool = false |- _ => apply orb_false_elim in H as [? ?] end.
  - eapply exec_frame; eassumption.
  - rewrite IHexec_p2, IHexec_p1; auto.
  - auto.
  - auto.
  - auto.
  - apply IHexec_p. eapply Henv; eassumption.
  - apply IHexec_p. rewrite pmods_invert. eapply Henv; eassumption.
  - auto.
  - rewrite IHexec_p3, IHexec_p2, IHexec_p1; auto.
Qed.

(** ** Calls, and where they may occur *)

Fixpoint has_call (st : pstmt) : bool :=
  match st with
  | PBase _ => false
  | PSeq a c | PIf _ a c _ | PLoop _ a c _ => has_call a || has_call c
  | PCall _ | PUncall _ => true
  end.

(** Well-formedness: [wf_stmt] on the straight-line leaves (no aliased
    swap), and nothing else: calls may occur anywhere.  (While `_gen_from`
    ran the [loop] part S2 with the loop's flag register at 1, a call in S2
    was miscompiled, and [wf_p] required [has_call c = false] there; now S2
    runs with the flag at 0 — see [s2_call_works] in TestProc.v.) *)
Fixpoint wf_p (st : pstmt) : Prop :=
  match st with
  | PBase s         => wf_stmt s
  | PSeq a c        => wf_p a /\ wf_p c
  | PIf _ a c _     => wf_p a /\ wf_p c
  | PLoop _ a c _   => wf_p a /\ wf_p c
  | PCall _ | PUncall _ => True
  end.

Definition env_wf (Γ : penv) : Prop :=
  forall f body, nth_error Γ f = Some body -> wf_p body.

Lemma wf_invert : forall s, wf_stmt s -> wf_stmt (invert s).
Proof. induction s; simpl; tauto. Qed.

Lemma has_call_invert : forall st, has_call (invert_p st) = has_call st.
Proof.
  induction st; simpl; try reflexivity.
  - rewrite IHst1, IHst2; apply orb_comm.
  - now rewrite IHst1, IHst2.
  - now rewrite IHst1, IHst2.
Qed.

Lemma wf_p_invert : forall st, wf_p st -> wf_p (invert_p st).
Proof.
  induction st; simpl; try tauto.
  apply wf_invert.
Qed.

(** ** A fuel interpreter, sound for [exec_p] *)

Fixpoint run_s (s : stmt) (σ : store) : option store :=
  match s with
  | Skip => Some σ
  | Assign x o e =>
      if occurs x e then None else Some (update σ x (adenote o (σ x) (eval σ e)))
  | Swap x y => Some (sw σ x y)
  | Seq a b => match run_s a σ with Some m => run_s b m | None => None end
  end.

Lemma run_s_sound : forall s σ σ', run_s s σ = Some σ' -> exec s σ σ'.
Proof.
  induction s; intros σ σ' H; simpl in H.
  - injection H as <-; constructor.
  - destruct (occurs x e) eqn:E; [discriminate |]. injection H as <-. now constructor.
  - injection H as <-; constructor.
  - destruct (run_s s1 σ) as [m |] eqn:E; [| discriminate].
    econstructor; eauto.
Qed.

Fixpoint run_p (fuel : nat) (Γ : penv) (st : pstmt) (σ : store) {struct fuel} : option store :=
  match fuel with
  | O => None
  | S f =>
      match st with
      | PBase s => run_s s σ
      | PSeq a c => match run_p f Γ a σ with Some m => run_p f Γ c m | None => None end
      | PIf e1 a c e2 =>
          if eval σ e1 =? 0 then
            match run_p f Γ c σ with
            | Some σ' => if eval σ' e2 =? 0 then Some σ' else None
            | None => None
            end
          else
            match run_p f Γ a σ with
            | Some σ' => if eval σ' e2 =? 0 then None else Some σ'
            | None => None
            end
      | PLoop e1 a c e2 => if eval σ e1 =? 0 then None else run_lp f Γ e1 a c e2 σ
      | PCall g => match nth_error Γ g with Some body => run_p f Γ body σ | None => None end
      | PUncall g =>
          match nth_error Γ g with Some body => run_p f Γ (invert_p body) σ | None => None end
      end
  end
with run_lp (fuel : nat) (Γ : penv) (e1 : expr) (a c : pstmt) (e2 : expr) (σ : store)
  {struct fuel} : option store :=
  match fuel with
  | O => None
  | S f =>
      match run_p f Γ a σ with
      | None => None
      | Some σ1 =>
          if eval σ1 e2 =? 0 then
            match run_p f Γ c σ1 with
            | Some σ2 => if eval σ2 e1 =? 0 then run_lp f Γ e1 a c e2 σ2 else None
            | None => None
            end
          else Some σ1
      end
  end.

Lemma run_p_sound : forall fuel Γ,
  (forall st σ σ', run_p fuel Γ st σ = Some σ' -> exec_p Γ st σ σ') /\
  (forall e1 a c e2 σ σ', run_lp fuel Γ e1 a c e2 σ = Some σ' -> lp_p Γ e1 a c e2 σ σ').
Proof.
  induction fuel as [| f IH]; intro Γ; split; intros *; simpl; try discriminate.
  - destruct (IH Γ) as [IHs IHl].
    destruct st as [s | a c | e1 a c e2 | e1 a c e2 | g | g]; intro H.
    + constructor; now apply run_s_sound.
    + destruct (run_p f Γ a σ) as [m |] eqn:E; [| discriminate].
      econstructor; eauto.
    + destruct (Z.eqb_spec (eval σ e1) 0) as [He1 | He1].
      * destruct (run_p f Γ c σ) as [m |] eqn:E; [| discriminate].
        destruct (Z.eqb_spec (eval m e2) 0) as [He2 | He2]; [| discriminate].
        injection H as <-. apply EP_IfFalse; auto.
      * destruct (run_p f Γ a σ) as [m |] eqn:E; [| discriminate].
        destruct (Z.eqb_spec (eval m e2) 0) as [He2 | He2]; [discriminate |].
        injection H as <-. apply EP_IfTrue; auto.
    + destruct (Z.eqb_spec (eval σ e1) 0) as [He1 | He1]; [discriminate |].
      apply EP_Loop; auto.
    + destruct (nth_error Γ g) as [body |] eqn:E; [| discriminate].
      eapply EP_Call; eauto.
    + destruct (nth_error Γ g) as [body |] eqn:E; [| discriminate].
      eapply EP_Uncall; eauto.
  - destruct (IH Γ) as [IHs IHl]. intro H.
    destruct (run_p f Γ a σ) as [σ1 |] eqn:E1; [| discriminate].
    destruct (Z.eqb_spec (eval σ1 e2) 0) as [He2 | He2].
    + destruct (run_p f Γ c σ1) as [σ2 |] eqn:E2; [| discriminate].
      destruct (Z.eqb_spec (eval σ2 e1) 0) as [He1 | He1]; [| discriminate].
      eapply LPP_More; eauto.
    + injection H as <-. apply LPP_One; auto.
Qed.

Corollary run_p_exec : forall fuel Γ st σ σ',
  run_p fuel Γ st σ = Some σ' -> exec_p Γ st σ σ'.
Proof. intros fuel Γ; apply (proj1 (run_p_sound fuel Γ)). Qed.
