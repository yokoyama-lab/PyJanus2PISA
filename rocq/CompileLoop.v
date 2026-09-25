(** * CompileLoop.v — compiling [from e1 do S1 loop S2 until e2], and its correctness

    This file adds Janus's loop to the verified translation of CompileIf.v
    and proves it correct on the control-flow machine of PISACtl.v.  The
    emitted code is, line for line, the layout `_gen_from` in `codegen.py`
    produces since PR #6 (labels allocated in its order: `from_test` = [n],
    `from_loop` = [n+1], `from_exit` = [n+2], `from_do` = [n+3]):

<<
            <rt ^= _as_flag(e1)>                           -- rt := (e1 != 0)
            XORI rt 1                                      -- rt := 0 iff e1 holds
            BNE rt r0 finish                               -- entry assertion violated
   do:      <S1>                   (label on S1's first line, or on a NOP
                                    `ADDI r0 0` when S1 emits no code)
   test:    <rt ^= _as_flag(e2)>                           -- rt := (e2 != 0)
            BEQ rt r0 loop
            XORI rt 1                                      -- rt := 0
            BRA exit
   loop:    XORI rt 1                                      -- rt := 1
            <S2>
            <rt ^= _as_flag(e1)>                           -- rt := 1 xor (e1 != 0)
            XORI rt 1                                      -- rt := 0 iff e1 fails
            BNE rt r0 finish                               -- re-entry assertion violated
            BRA do
   exit:    ADDI r0 0                                      -- NOP
>>

    None of the branches is paired (their targets are data lines or
    [finish]), so only the direct-jump behaviour of PISACtl.v is exercised.
    [finish] is a label the enclosing program provides; the compiler takes it
    as a parameter [fin] and never defines it.

    Tests are normalised to 0/1 exactly as in CompileIf.v ([flag_block]), so
    the source relation [exec_l] is Janus's own (true = nonzero), with no
    Boolean-valuedness side condition.

    Results:
    - [compile_l_spec]: semantic preservation with a clean register file for
      every execution the source admits (all `BNE rt r0 finish` fall through);
      [compile_l_program]: the closed program with a `finish:` line.
    - [compile_l_if_violation], [compile_l_loop_entry_violation],
      [compile_l_loop_reentry_violation]: a violated `fi` assertion, a false
      entry assertion, and a true re-entry assertion after the first round
      make the code jump to [finish] with the flag register at [1] — the
      state `pisa_interp.py` reports as garbage at FINISH.  (The previous
      layout cleared the flag with `XOR rt rt` and accepted such programs
      silently; see MANIFEST.md.)

    The source language [lstmt] subsumes [cstmt] of CompileIf.v
    ([lift], [exec_c_lift], [compile_l_lift]); its [If] case reuses the
    layout lemmas of CompileIf.v unchanged. *)

From Stdlib Require Import ZArith List Lia Bool.
From Stdlib Require Import FunctionalExtensionality.
Require Import PISA Src Compile PISACtl CompileIf.
Import ListNotations.
Open Scope Z_scope.

(** ** Relabeling one data line

    `_gen_from` puts the label `from_do` on the first line of S1's code.  The
    induction hypothesis for S1 is about S1's code as compiled, i.e. with
    that line unlabeled; this section transfers a run across the change.
    The only requirement is that the new label is not defined elsewhere. *)

Lemma find_label_app : forall l p q,
  find_label l (p ++ q)
  = match find_label l p with
    | Some k => Some k
    | None => option_map (fun k => (length p + k)%nat) (find_label l q)
    end.
Proof.
  intros l p q; induction p as [| [[l'|] i] t IH]; simpl.
  - destruct (find_label l q); reflexivity.
  - destruct (Nat.eqb l l'); [reflexivity |]. rewrite IH.
    destruct (find_label l t); [reflexivity |]. simpl.
    destruct (find_label l q); reflexivity.
  - rewrite IH. destruct (find_label l t); [reflexivity |]. simpl.
    destruct (find_label l q); reflexivity.
Qed.

Section Relabel.

Variables (pre post : lprog) (i : instr) (l0 : label).

Let P  := pre ++ (None, COp i) :: post.
Let P' := pre ++ (Some l0, COp i) :: post.

Hypothesis Hfresh : ~ In l0 (labels P).

Lemma relabel_find_other : forall l, l <> l0 -> find_label l P' = find_label l P.
Proof.
  intros l Hl; unfold P, P'. rewrite !find_label_app. cbn [find_label].
  destruct (Nat.eqb_spec l l0); [contradiction | reflexivity].
Qed.

Lemma relabel_find_some : forall l t, find_label l P = Some t -> find_label l P' = Some t.
Proof.
  intros l t H. rewrite relabel_find_other; [exact H |].
  intros ->. rewrite (find_label_notin l0 P Hfresh) in H. discriminate.
Qed.

Lemma relabel_find_back : forall l t,
  find_label l P' = Some t -> find_label l P = Some t \/ t = length pre.
Proof.
  intros l t H. destruct (Nat.eq_dec l l0) as [-> | Hne].
  - right. unfold P' in H. rewrite find_label_app in H.
    assert (Hp : find_label l0 pre = None).
    { apply find_label_notin. intro Hin; apply Hfresh.
      unfold P; rewrite labels_app; apply in_or_app; now left. }
    rewrite Hp in H; cbn [find_label] in H. rewrite Nat.eqb_refl in H.
    cbn [option_map] in H. injection H as <-. lia.
  - left. now rewrite <- relabel_find_other.
Qed.

Lemma relabel_nth : forall k,
  nth_error P' k
  = if Nat.eqb k (length pre) then Some (Some l0, COp i) else nth_error P k.
Proof.
  intro k; unfold P, P'.
  destruct (Nat.lt_trichotomy k (length pre)) as [Hlt | [-> | Hgt]].
  - rewrite !nth_error_app1 by exact Hlt.
    destruct (Nat.eqb_spec k (length pre)); [lia | reflexivity].
  - rewrite Nat.eqb_refl, nth_error_app2, Nat.sub_diag by lia. reflexivity.
  - rewrite !nth_error_app2 by lia.
    destruct (Nat.eqb_spec k (length pre)); [lia |].
    destruct (k - length pre)%nat eqn:E; [lia | reflexivity].
Qed.

Lemma relabel_nth_at : nth_error P (length pre) = Some (None, COp i).
Proof. unfold P; eapply nth_error_at; reflexivity. Qed.

Lemma relabel_points_back : forall b a,
  a <> length pre -> points_back P' b a = points_back P b a.
Proof.
  intros b a Ha.
  assert (G : forall l'', match find_label l'' P' with Some a' => Nat.eqb a' a | None => false end
                        = match find_label l'' P with Some a' => Nat.eqb a' a | None => false end).
  { intro l''. destruct (find_label l'' P) as [a'|] eqn:E1.
    - now rewrite (relabel_find_some _ _ E1).
    - destruct (find_label l'' P') as [a'|] eqn:E2; [| reflexivity].
      destruct (relabel_find_back _ _ E2) as [E3 | ->]; [congruence |].
      apply Nat.eqb_neq. auto. }
  unfold points_back.
  destruct (bra_target b); [apply G |].
  destruct (cond_target b); [apply G | reflexivity].
Qed.

Lemma relabel_paired : forall a lo x l t,
  nth_error P a = Some (lo, x) -> bra_target x = Some l -> find_label l P = Some t ->
  paired P' a = paired P a.
Proof.
  intros a lo x l t Ha Hx Hl.
  assert (Hane : a <> length pre).
  { intro E; subst a. rewrite relabel_nth_at in Ha. injection Ha as _ <-. discriminate. }
  unfold paired. rewrite (relabel_nth a).
  destruct (Nat.eqb_spec a (length pre)); [contradiction |].
  rewrite Ha, Hx, (relabel_find_some _ _ Hl), Hl.
  rewrite (relabel_nth t).
  destruct (Nat.eqb_spec t (length pre)) as [-> | Htne].
  - rewrite relabel_nth_at. reflexivity.
  - destruct (nth_error P t) as [[lo2 b'] |]; [| reflexivity].
    now apply relabel_points_back.
Qed.

Lemma relabel_cond : forall pc br s tk l c',
  cond_step P pc br s tk l = Some c' -> cond_step P' pc br s tk l = Some c'.
Proof.
  intros pc br s tk l c' H; unfold cond_step in *.
  destruct (br =? 0), tk; try exact H;
    (destruct (find_label l P) as [t|] eqn:Hl; [| discriminate]);
    rewrite (relabel_find_some _ _ Hl); exact H.
Qed.

Lemma relabel_cstep : forall c c', cstep P c = Some c' -> cstep P' c = Some c'.
Proof.
  intros [pc br s] c' H; unfold cstep in *.
  rewrite relabel_nth.
  destruct (Nat.eqb_spec pc (length pre)) as [-> | Hne].
  - rewrite relabel_nth_at in H. exact H.
  - destruct (nth_error P pc) as [[lo x] |] eqn:Hn; [| discriminate].
    destruct x; try exact H; try (now apply relabel_cond).
    + unfold bra_step in *.
      destruct (find_label l P) as [t|] eqn:Hl; [| discriminate].
      rewrite (relabel_find_some _ _ Hl), (relabel_paired pc lo (CBra l) l t Hn eq_refl Hl).
      exact H.
    + unfold bra_step in *.
      destruct (find_label l P) as [t|] eqn:Hl; [| discriminate].
      rewrite (relabel_find_some _ _ Hl), (relabel_paired pc lo (CRbra l) l t Hn eq_refl Hl).
      exact H.
Qed.

Lemma steps_relabel : forall c c', steps P c c' -> steps P' c c'.
Proof.
  intros c c' H; induction H as [c | c c' c'' Hs Hst IH]; [constructor |].
  eapply steps_step; [apply relabel_cstep; exact Hs | exact IH].
Qed.

End Relabel.

(** ** Source language with [If] and loops *)

Inductive lstmt :=
| LBase (s : stmt)
| LSeq  (a c : lstmt)
| LIf   (e1 : expr) (a c : lstmt) (e2 : expr)    (** [if e1 then a else c fi e2] *)
| LLoop (e1 : expr) (a c : lstmt) (e2 : expr).   (** [from e1 do a loop c until e2] *)

Fixpoint wf_lstmt (st : lstmt) : Prop :=
  match st with
  | LBase s       => wf_stmt s
  | LSeq a c      => wf_lstmt a /\ wf_lstmt c
  | LIf _ a c _   => wf_lstmt a /\ wf_lstmt c
  | LLoop _ a c _ => wf_lstmt a /\ wf_lstmt c
  end.

(** The rules of [Janus.v] ([E_Loop], [L_one], [L_more]), truth being
    nonzero: the entry assertion [e1] holds on entry and fails ([= 0]) on
    every re-entry; the exit test [e2] is nonzero when the loop exits and [0]
    when it continues.  [lp_l e1 a c e2 σ σ'] is  a (c a)^k  from [σ] to [σ']. *)
Inductive exec_l : lstmt -> store -> store -> Prop :=
| EL_Base : forall s σ σ', exec s σ σ' -> exec_l (LBase s) σ σ'
| EL_Seq : forall a c σ m σ',
    exec_l a σ m -> exec_l c m σ' -> exec_l (LSeq a c) σ σ'
| EL_IfTrue : forall e1 a c e2 σ σ',
    eval σ e1 <> 0 -> exec_l a σ σ' -> eval σ' e2 <> 0 ->
    exec_l (LIf e1 a c e2) σ σ'
| EL_IfFalse : forall e1 a c e2 σ σ',
    eval σ e1 = 0 -> exec_l c σ σ' -> eval σ' e2 = 0 ->
    exec_l (LIf e1 a c e2) σ σ'
| EL_Loop : forall e1 a c e2 σ σ',
    eval σ e1 <> 0 -> lp_l e1 a c e2 σ σ' -> exec_l (LLoop e1 a c e2) σ σ'
with lp_l : expr -> lstmt -> lstmt -> expr -> store -> store -> Prop :=
| LP_One : forall e1 a c e2 σ σ',
    exec_l a σ σ' -> eval σ' e2 <> 0 -> lp_l e1 a c e2 σ σ'
| LP_More : forall e1 a c e2 σ σ1 σ2 σ',
    exec_l a σ σ1 -> eval σ1 e2 = 0 ->
    exec_l c σ1 σ2 -> eval σ2 e1 = 0 ->
    lp_l e1 a c e2 σ2 σ' -> lp_l e1 a c e2 σ σ'.

Scheme exec_l_mut := Induction for exec_l Sort Prop
  with lp_l_mut   := Induction for lp_l Sort Prop.

(** *** Reversibility of the source (the proof of [Janus.exec_rev]) *)

Fixpoint invert_l (st : lstmt) : lstmt :=
  match st with
  | LBase s        => LBase (invert s)
  | LSeq a c       => LSeq (invert_l c) (invert_l a)
  | LIf e1 a c e2   => LIf e2 (invert_l a) (invert_l c) e1
  | LLoop e1 a c e2 => LLoop e2 (invert_l a) (invert_l c) e1
  end.

Lemma lp_l_exit : forall e1 a c e2 σ σ', lp_l e1 a c e2 σ σ' -> eval σ' e2 <> 0.
Proof. intros until σ'; intro H; induction H; assumption. Qed.

(** Zero or more continuing rounds [a ; c], no exit baked in. *)
Inductive opn_l (e1 : expr) (a c : lstmt) (e2 : expr) : store -> store -> Prop :=
| OL_nil  : forall σ, opn_l e1 a c e2 σ σ
| OL_cons : forall σ σ1 σ2 σ',
    exec_l a σ σ1 -> eval σ1 e2 = 0 ->
    exec_l c σ1 σ2 -> eval σ2 e1 = 0 ->
    opn_l e1 a c e2 σ2 σ' -> opn_l e1 a c e2 σ σ'.

Lemma opn_l_snoc : forall e1 a c e2 σ m m1 m2,
  opn_l e1 a c e2 σ m ->
  exec_l a m m1 -> eval m1 e2 = 0 -> exec_l c m1 m2 -> eval m2 e1 = 0 ->
  opn_l e1 a c e2 σ m2.
Proof.
  intros e1 a c e2 σ m m1 m2 H. revert m1 m2.
  induction H; intros m1 m2 Ha He2 Hc He1.
  - eapply OL_cons; eauto. apply OL_nil.
  - eapply OL_cons; eauto.
Qed.

Lemma opn_l_to_lp : forall e1 a c e2 σ m σ',
  opn_l e1 a c e2 σ m -> exec_l a m σ' -> eval σ' e2 <> 0 -> lp_l e1 a c e2 σ σ'.
Proof.
  intros e1 a c e2 σ m σ' H. induction H; intros Ha He.
  - apply LP_One; assumption.
  - eapply LP_More; eauto.
Qed.

Theorem exec_l_rev : forall st σ σ', exec_l st σ σ' -> exec_l (invert_l st) σ' σ.
Proof.
  intros st σ σ' H.
  induction H using exec_l_mut
    with (P0 := fun e1 a c e2 σ σ' (_ : lp_l e1 a c e2 σ σ') =>
      exists q, opn_l e2 (invert_l a) (invert_l c) e1 σ' q /\ exec_l (invert_l a) q σ);
    cbn [invert_l].
  - constructor; now apply exec_rev.
  - econstructor; eassumption.
  - apply EL_IfTrue; assumption.
  - apply EL_IfFalse; assumption.
  - destruct IHexec_l as [q [Hopn Hq]].
    apply EL_Loop; [eapply lp_l_exit; eassumption |].
    eapply opn_l_to_lp; eassumption.
  - exists σ'. split; [apply OL_nil | assumption].
  - destruct IHexec_l3 as [q [Hopn Hq]].
    exists σ1. split; [| assumption].
    eapply opn_l_snoc; eassumption.
Qed.

(** *** [lstmt] subsumes [cstmt] *)

Fixpoint lift (st : cstmt) : lstmt :=
  match st with
  | CBase s       => LBase s
  | CSeq a c      => LSeq (lift a) (lift c)
  | CIf e1 a c e2 => LIf e1 (lift a) (lift c) e2
  end.

Lemma exec_c_lift : forall st σ σ', exec_c st σ σ' -> exec_l (lift st) σ σ'.
Proof.
  intros st σ σ' H; induction H; cbn [lift].
  - now constructor.
  - econstructor; eassumption.
  - now apply EL_IfTrue.
  - now apply EL_IfFalse.
Qed.

Lemma wf_lift : forall st, wf_cstmt st -> wf_lstmt (lift st).
Proof. induction st; simpl; tauto. Qed.

(** ** The compiler *)

(** `ADDI r0 0`, the NOP `_gen_from` emits for an empty [S1], at [loop] and
    at [exit]. *)
Definition nop : instr := IAddi 0%nat 0.

Lemma step_nop : forall s, step nop s = s.
Proof.
  intros [R M]; cbn [step nop regs mem]. now rewrite Z.add_0_r, rupd_id.
Qed.

(** `code_do[0] = LabeledInstr(entry_do, code_do[0].instr)`, or a labeled NOP. *)
Definition label_first (l : label) (p : lprog) : lprog :=
  match p with
  | [] => [(Some l, COp nop)]
  | (_, x) :: t => (Some l, x) :: t
  end.

(** [loop_code fin b n e1 e2 pa pb]: flag register [b], labels [n .. n+3],
    compiled bodies [pa] (S1) and [pb] (S2), `finish` label [fin].  The flag
    is 0 while S1 and S2 run: `loop:` is a NOP (the `BEQ` that jumps there is
    taken only when the flag is 0), and the re-entry assertion leaves
    [rt = e1 != 0], which the `BNE` checks directly. *)
Definition loop_code (fin : label) (b : reg) (n : label) (e1 e2 : expr)
                     (pa pb : lprog) : lprog :=
  ops (flag_block e1 b)
  ++ (None, COp (IXori b 1))
  :: (None, CBne b 0%nat fin)
  :: label_first (n + 3)%nat pa
  ++ ops_l n (flag_block e2 b)
  ++ (None, CBeq b 0%nat (S n))
  :: (None, COp (IXori b 1))
  :: (None, CBra (n + 2)%nat)
  :: (Some (S n), COp nop)
  :: pb
  ++ ops (flag_block e1 b)
  ++ (None, CBne b 0%nat fin)
  :: (None, CBra (n + 3)%nat)
  :: (Some (n + 2)%nat, COp nop)
  :: [].

Fixpoint compile_l (fin : label) (st : lstmt) (b : reg) (n : label) : lprog * label :=
  match st with
  | LBase s => (ops (compile_at b s), n)
  | LSeq a c =>
      let '(p1, n1) := compile_l fin a b n in
      let '(p2, n2) := compile_l fin c b n1 in
      (p1 ++ p2, n2)
  | LIf e1 a c e2 =>
      let '(pa, na) := compile_l fin a (S b) (n + 5)%nat in
      let '(pc, nc) := compile_l fin c (S b) na in
      (if_code fin b n e1 e2 pa pc, nc)
  | LLoop e1 a c e2 =>
      let '(pa, na) := compile_l fin a (S b) (n + 4)%nat in
      let '(pc, nc) := compile_l fin c (S b) na in
      (loop_code fin b n e1 e2 pa pc, nc)
  end.

(** On the [If] fragment the loop compiler *is* the [If] compiler, so
    [compile_l_spec] contains [compile_c_spec]. *)
Lemma compile_l_lift : forall fin st b n, compile_l fin (lift st) b n = compile_c fin st b n.
Proof.
  intros fin st; induction st as [s | a IHa c IHc | e1 a IHa c IHc e2]; intros b n;
    simpl; try reflexivity.
  - rewrite IHa. destruct (compile_c fin a b n) as [p1 n1]. now rewrite IHc.
  - rewrite IHa. destruct (compile_c fin a (S b) (n + 5)%nat) as [pa na]. now rewrite IHc.
Qed.

(** ** Shape facts: labels, and the first line *)

Lemma labels_label_first : forall l p l',
  In l' (labels (label_first l p)) -> l' = l \/ In l' (labels p).
Proof.
  intros l [| [[lo|] x] t] l' H; simpl in H.
  - intuition.
  - destruct H as [H | H]; [now left | right; now right].
  - destruct H as [H | H]; [now left | now right].
Qed.

Lemma labels_loop_code : forall fin b n e1 e2 pa pb l,
  In l (labels (loop_code fin b n e1 e2 pa pb)) ->
  l = (n + 3)%nat \/ In l (labels pa) \/ l = n \/ l = S n \/ In l (labels pb)
  \/ l = (n + 2)%nat.
Proof.
  intros fin b n e1 e2 pa pb l H; unfold loop_code in H.
  in_labels H;
  repeat match goal with
  | H : In _ (labels (label_first _ _)) |- _ => apply labels_label_first in H
  | H : _ \/ _ |- _ => destruct H
  end; subst; intuition auto.
Qed.

Lemma compile_l_labels : forall fin st b n p n',
  compile_l fin st b n = (p, n') ->
  (n <= n')%nat /\ (forall l, In l (labels p) -> (n <= l < n')%nat).
Proof.
  intros fin; induction st as [s | a IHa c IHc | e1 a IHa c IHc e2 | e1 a IHa c IHc e2];
    intros b n p n' Hc; simpl in Hc.
  - injection Hc as <- <-. split; [lia |].
    intros l H; rewrite labels_ops in H; destruct H.
  - destruct (compile_l fin a b n) as [p1 n1] eqn:E1.
    destruct (compile_l fin c b n1) as [p2 n2] eqn:E2.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ E1) as [Hle1 Hin1].
    destruct (IHc _ _ _ _ E2) as [Hle2 Hin2].
    split; [lia |].
    intros l H; rewrite labels_app, in_app_iff in H.
    destruct H as [H | H]; [apply Hin1 in H | apply Hin2 in H]; lia.
  - destruct (compile_l fin a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_l fin c (S b) na) as [pb nb] eqn:Eb.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ Ea) as [Hlea Hina].
    destruct (IHc _ _ _ _ Eb) as [Hleb Hinb].
    split; [lia |].
    intros l H; apply labels_if_code in H.
    repeat match goal with
    | H : _ \/ _ |- _ => destruct H
    | H : In _ (labels pa) |- _ => apply Hina in H
    | H : In _ (labels pb) |- _ => apply Hinb in H
    end; subst; lia.
  - destruct (compile_l fin a (S b) (n + 4)%nat) as [pa na] eqn:Ea.
    destruct (compile_l fin c (S b) na) as [pb nb] eqn:Eb.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ Ea) as [Hlea Hina].
    destruct (IHc _ _ _ _ Eb) as [Hleb Hinb].
    split; [lia |].
    intros l H; apply labels_loop_code in H.
    repeat match goal with
    | H : _ \/ _ |- _ => destruct H
    | H : In _ (labels pa) |- _ => apply Hina in H
    | H : In _ (labels pb) |- _ => apply Hinb in H
    end; subst; lia.
Qed.

(** Compiled code is empty or starts with an unlabeled data line, so
    [label_first] never overwrites a label (`_gen_from` would silently drop
    it) and the line `BRA do` jumps to is never a branch. *)
Definition head_ok (p : lprog) : Prop :=
  p = [] \/ exists x t, p = (None, COp x) :: t.

Lemma compile_l_head : forall fin st b n p n',
  compile_l fin st b n = (p, n') -> head_ok p.
Proof.
  intros fin; induction st as [s | a IHa c IHc | e1 a IHa c IHc e2 | e1 a IHa c IHc e2];
    intros b n p n' Hc; simpl in Hc.
  - injection Hc as <- <-. destruct (compile_at b s) as [| x t]; [now left |].
    right; exists x, (ops t); reflexivity.
  - destruct (compile_l fin a b n) as [p1 n1] eqn:E1.
    destruct (compile_l fin c b n1) as [p2 n2] eqn:E2.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ E1) as [-> | [x [t ->]]]; [exact (IHc _ _ _ _ E2) |].
    right; exists x, (t ++ p2); reflexivity.
  - destruct (compile_l fin a (S b) (n + 5)%nat) as [pa na].
    destruct (compile_l fin c (S b) na) as [pb nb].
    injection Hc as <- <-. unfold if_code.
    destruct (flag_block e1 b) as [| x t] eqn:E;
      [exfalso; eapply flag_block_nonempty; eassumption |].
    right; eexists; eexists; reflexivity.
  - destruct (compile_l fin a (S b) (n + 4)%nat) as [pa na].
    destruct (compile_l fin c (S b) na) as [pb nb].
    injection Hc as <- <-. unfold loop_code.
    destruct (flag_block e1 b) as [| x t] eqn:E;
      [exfalso; eapply flag_block_nonempty; eassumption |].
    right; eexists; eexists; reflexivity.
Qed.

Lemma label_first_head : forall l p, head_ok p ->
  exists x t, label_first l p = (Some l, COp x) :: t.
Proof.
  intros l p [-> | [x [t ->]]]; simpl; eauto.
Qed.

(** ** Running the loop layout

    As for [If] (Section [IfLayout] of CompileIf.v) everything is relative
    to a prefix [pre] and a suffix [post].  The bodies enter through
    [frag_spec], which is exactly the shape of the main theorem for a
    fragment, quantified over its context. *)

Definition frag_spec (p : lprog) (lo hi : label) (σ σ' : store) (ms : state) : Prop :=
  forall pre' post',
    (forall l, In l (labels pre')  -> ~ (lo <= l < hi)%nat) ->
    (forall l, In l (labels post') -> ~ (lo <= l < hi)%nat) ->
    exists ms',
      steps (pre' ++ p ++ post') (mkC (length pre') 0 ms)
            (mkC (length pre' + length p) 0 ms')
      /\ models ms' σ' /\ regs ms' = regs ms.

Section LoopLayout.

Variable pre post : lprog.
Variable b : reg.
Variable n na nc : label.
Variable e1 e2 : expr.
Variable pa pb : lprog.
Variable fin : label.

Hypothesis Hb    : b <> 0%nat.
Hypothesis Hn    : (n + 4 <= na <= nc)%nat.
Hypothesis Hpre  : forall l, In l (labels pre)  -> ~ (n <= l < nc)%nat.
Hypothesis Hpost : forall l, In l (labels post) -> ~ (n <= l < nc)%nat.
Hypothesis Hpa   : forall l, In l (labels pa)   -> (n + 4 <= l < na)%nat.
Hypothesis Hpb   : forall l, In l (labels pb)   -> (na <= l < nc)%nat.
Hypothesis Hhead : head_ok pa.

Let ltest := n.
Let lloop := S n.
Let lexit := (n + 2)%nat.
Let ldo   := (n + 3)%nat.
Let T1 := flag_block e1 b.
Let T2 := flag_block e2 b.
Let LA := label_first ldo pa.

Let P := pre ++ loop_code fin b n e1 e2 pa pb ++ post.

Let n0 := length pre.
Let t1 := length T1.
Let t2 := length T2.
Let la := length LA.
Let lb := length pb.
Let pK  := (n0 + t1)%nat.     (* XORI after the entry assertion *)
Let pE1 := S pK.              (* BNE rt r0 finish (entry) *)
Let pDo := S pE1.             (* do:   S1 *)
Let pT  := (pDo + la)%nat.    (* test: the exit test *)
Let pQ  := (pT + t2)%nat.     (* BEQ rt r0 loop *)
Let pC2 := S pQ.              (* XORI *)
Let pJ  := S pC2.             (* BRA exit *)
Let pL  := S pJ.              (* loop: NOP *)
Let pS  := S pL.              (* S2 *)
Let pR  := (pS + lb)%nat.     (* the re-entry assertion *)
Let pN3 := (pR + t1)%nat.     (* BNE rt r0 finish (re-entry) *)
Let pBk := S pN3.             (* BRA do *)
Let pX  := S pBk.             (* exit: NOP *)

(** Prefixes ending just before a line of interest, and the two body contexts. *)
Let A0 := pre ++ ops T1 ++ [(None, COp (IXori b 1)); (None, CBne b 0%nat fin)].
Let A1 := A0 ++ LA ++ ops_l ltest T2.
Let A4 := A1 ++ [(None, CBeq b 0%nat lloop); (None, COp (IXori b 1)); (None, CBra lexit)].
Let A5 := A4 ++ [(Some lloop, COp nop)].
Let A6 := A5 ++ pb ++ ops T1.
Let A8 := A6 ++ [(None, CBne b 0%nat fin); (None, CBra ldo)].
Let Z9 := (Some lexit, COp nop) :: post.
Let postS := ops T1 ++ (None, CBne b 0%nat fin) :: (None, CBra ldo) :: Z9.
Let postD := ops_l ltest T2 ++ (None, CBeq b 0%nat lloop) :: (None, COp (IXori b 1))
             :: (None, CBra lexit) :: (Some lloop, COp nop) :: pb ++ postS.

Ltac len_solve :=
  unfold pX, pBk, pN3, pR, pS, pL, pJ, pC2, pQ, pT, pDo, pE1, pK, lb, la, t2, t1, n0,
         A8, A6, A5, A4, A1, A0, T1, T2 in *;
  repeat first [ rewrite length_app | rewrite length_ops | rewrite length_ops_l
               | progress cbn [length] ];
  unfold lprog, line, label in *;
  lia.

Ltac lbl_solve :=
  unfold A8, A6, A5, A4, A1, A0, postD, postS, Z9, LA in *;
  let H := fresh in
  first [ intros ? H | intro H ]; in_labels H;
  repeat match goal with
  | H : In _ (labels (label_first _ _)) |- _ =>
      apply labels_label_first in H; destruct H as [H | H]
  | H : In _ (labels pre) |- _ => apply Hpre in H
  | H : In _ (labels post) |- _ => apply Hpost in H
  | H : In _ (labels pa) |- _ => apply Hpa in H
  | H : In _ (labels pb) |- _ => apply Hpb in H
  end;
  unfold ltest, lloop, lexit, ldo in *; lia.

Lemma P_D : P = A0 ++ LA ++ postD.
Proof. unfold P, A0, LA, postD, postS, Z9, loop_code; norm_app; reflexivity. Qed.

Lemma P_S : P = A5 ++ pb ++ postS.
Proof. unfold P, A5, A4, A1, A0, LA, postS, Z9, loop_code; norm_app; reflexivity. Qed.

Lemma len_loop_code : (length pre + length (loop_code fin b n e1 e2 pa pb))%nat = S pX.
Proof. unfold loop_code; fold T1 T2 ldo LA; len_solve. Qed.

Lemma pDo_eq : (length pre + S (S (length (flag_block e1 b))))%nat = pDo.
Proof. len_solve. Qed.

(** *** The lines *)

Lemma nth_K : nth_error P pK = Some (None, COp (IXori b 1)).
Proof.
  replace pK with (length (pre ++ ops T1)) by len_solve.
  eapply nth_error_at. unfold P, loop_code; norm_app; reflexivity.
Qed.

Lemma nth_E1 : nth_error P pE1 = Some (None, CBne b 0%nat fin).
Proof.
  replace pE1 with (length (pre ++ ops T1 ++ [(None, COp (IXori b 1))])) by len_solve.
  eapply nth_error_at. unfold P, loop_code; norm_app; reflexivity.
Qed.

Lemma nth_Do : exists x, nth_error P pDo = Some (Some ldo, COp x).
Proof.
  destruct (label_first_head ldo pa Hhead) as [x [t Ht]].
  exists x. replace pDo with (length A0) by len_solve.
  eapply nth_error_at. rewrite P_D. unfold LA; rewrite Ht. reflexivity.
Qed.

Lemma nth_Q : nth_error P pQ = Some (None, CBeq b 0%nat lloop).
Proof.
  replace pQ with (length A1) by len_solve.
  eapply nth_error_at. unfold P, A1, A0, loop_code; norm_app; reflexivity.
Qed.

Lemma nth_C2 : nth_error P pC2 = Some (None, COp (IXori b 1)).
Proof.
  replace pC2 with (length (A1 ++ [(None, CBeq b 0%nat lloop)])) by len_solve.
  eapply nth_error_at. unfold P, A1, A0, loop_code; norm_app; reflexivity.
Qed.

Lemma nth_J : nth_error P pJ = Some (None, CBra lexit).
Proof.
  replace pJ with (length (A1 ++ [(None, CBeq b 0%nat lloop); (None, COp (IXori b 1))]))
    by len_solve.
  eapply nth_error_at. unfold P, A1, A0, loop_code; norm_app; reflexivity.
Qed.

Lemma nth_L : nth_error P pL = Some (Some lloop, COp nop).
Proof.
  replace pL with (length A4) by len_solve.
  eapply nth_error_at. unfold P, A4, A1, A0, loop_code; norm_app; reflexivity.
Qed.

Lemma nth_N3 : nth_error P pN3 = Some (None, CBne b 0%nat fin).
Proof.
  replace pN3 with (length A6) by len_solve.
  eapply nth_error_at. unfold P, A6, A5, A4, A1, A0, loop_code; norm_app; reflexivity.
Qed.

Lemma nth_Bk : nth_error P pBk = Some (None, CBra ldo).
Proof.
  replace pBk with (length (A6 ++ [(None, CBne b 0%nat fin)])) by len_solve.
  eapply nth_error_at. unfold P, A6, A5, A4, A1, A0, loop_code; norm_app; reflexivity.
Qed.

Lemma nth_X : nth_error P pX = Some (Some lexit, COp nop).
Proof.
  replace pX with (length A8) by len_solve.
  eapply nth_error_at. unfold P, A8, A6, A5, A4, A1, A0, loop_code; norm_app; reflexivity.
Qed.

(** *** The labels *)

Lemma find_ldo : find_label ldo P = Some pDo.
Proof.
  destruct (label_first_head ldo pa Hhead) as [x [t Ht]].
  replace pDo with (length A0) by len_solve.
  eapply find_label_at; [rewrite P_D; unfold LA; rewrite Ht; reflexivity | lbl_solve].
Qed.

Lemma find_lloop : find_label lloop P = Some pL.
Proof.
  replace pL with (length A4) by len_solve.
  eapply find_label_at;
    [unfold P, A4, A1, A0, loop_code; norm_app; reflexivity | lbl_solve].
Qed.

Lemma find_lexit : find_label lexit P = Some pX.
Proof.
  replace pX with (length A8) by len_solve.
  eapply find_label_at;
    [unfold P, A8, A6, A5, A4, A1, A0, loop_code; norm_app; reflexivity | lbl_solve].
Qed.

(** *** No branch of the layout is paired *)

Lemma paired_J : paired P pJ = false.
Proof. eapply paired_bra_op; [apply nth_J | apply find_lexit | apply nth_X]. Qed.

Lemma paired_Bk : paired P pBk = false.
Proof.
  destruct nth_Do as [x Hx].
  eapply paired_bra_op; [apply nth_Bk | apply find_ldo | exact Hx].
Qed.

(** *** Entry: [rt := (e1 != 0) xor 1], reaching the entry `BNE` *)

Lemma entry_check : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  steps P (mkC n0 0 (mkState R M))
          (mkC pE1 0 (mkState (rupd b (Z.lxor (truth (eval σ e1)) 1) R) M)).
Proof.
  intros R M σ Hmod Hcl H0.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT1 : run T1 (mkState R M) = mkState (rupd b (truth (eval σ e1)) R) M).
  { unfold T1. rewrite (flag_block_spec e1 b _ σ).
    - cbn [regs mem]. now rewrite Hb0, Z.lxor_0_l.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  eapply steps_trans.
  { eapply (steps_ops T1 P pre). unfold P, loop_code; norm_app; reflexivity. }
  rewrite HT1.
  apply steps_one.
  rewrite (cstep_op P _ 0 _ None (IXori b 1)) by (apply nth_K).
  cbn [step regs mem]. now rewrite rupd_same, rupd_shadow.
Qed.

Lemma entry_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 ->
  steps P (mkC n0 0 (mkState R M)) (mkC pDo 0 (mkState R M)).
Proof.
  intros R M σ Hmod Hcl H0 He1.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply steps_trans; [apply (entry_check R M σ Hmod Hcl H0) |].
  rewrite (truth_nz _ He1). change (Z.lxor 1 1) with 0.
  rewrite rupd_zero by exact Hb0.
  apply steps_one.
  eapply cstep_bne_direct_not_taken; [apply nth_E1 | cbn [regs]; congruence].
Qed.

(** A false entry assertion: [rt = 1] at the `BNE`, which jumps to [finish]. *)
Lemma entry_violation : forall R M σ f,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 -> find_label fin P = Some f ->
  steps P (mkC n0 0 (mkState R M)) (mkC f 0 (mkState (rupd b 1 R) M)).
Proof.
  intros R M σ f Hmod Hcl H0 He1 Hf.
  eapply steps_trans; [apply (entry_check R M σ Hmod Hcl H0) |].
  rewrite He1, truth_0. change (Z.lxor 0 1) with 1.
  apply steps_one.
  eapply cstep_bne_direct_taken; [apply nth_E1 | exact Hf |].
  cbn [regs]. rewrite rupd_same, rupd_other by lia. lia.
Qed.

(** *** S1, through the relabeling of its first line *)

Lemma do_steps : forall ms σ σ',
  frag_spec pa (n + 4)%nat na σ σ' ms ->
  exists ms', steps P (mkC pDo 0 ms) (mkC pT 0 ms') /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros ms σ σ' IH.
  destruct Hhead as [Hnil | [x [t Hcons]]].
  - (* S1 emits no code: the label sits on a NOP *)
    destruct (IH A0 ((Some ldo, COp nop) :: postD)) as [ms' [Hst [Hm Hr]]];
      [lbl_solve | lbl_solve |].
    exists ms'. split; [| split; assumption].
    assert (E : A0 ++ pa ++ (Some ldo, COp nop) :: postD = P)
      by (rewrite P_D; unfold LA; rewrite Hnil; reflexivity).
    rewrite E, Hnil in Hst. cbn [length] in Hst. rewrite Nat.add_0_r in Hst.
    assert (Hla : la = 1%nat) by (unfold la, LA; rewrite Hnil; reflexivity).
    replace pDo with (length A0) by len_solve.
    eapply steps_trans; [exact Hst |].
    apply steps_one.
    rewrite (cstep_op P (length A0) 0 ms' (Some ldo) nop), step_nop.
    + unfold pT; rewrite Hla. do 2 f_equal. len_solve.
    + eapply nth_error_at. rewrite P_D. unfold LA; rewrite Hnil. reflexivity.
  - (* S1's first line is an unlabeled data line: relabel it *)
    assert (Ht : forall l, In l (labels t) -> (n + 4 <= l < na)%nat)
      by (intros l Hl; apply Hpa; rewrite Hcons; exact Hl).
    destruct (IH A0 postD) as [ms' [Hst [Hm Hr]]]; [lbl_solve | lbl_solve |].
    exists ms'. split; [| split; assumption].
    rewrite Hcons in Hst.
    assert (HF : ~ In ldo (labels (A0 ++ (None, COp x) :: t ++ postD))).
    { rewrite labels_app; cbn [labels]; rewrite labels_app.
      intro H; rewrite !in_app_iff in H. destruct H as [H | [H | H]].
      - revert H; lbl_solve.
      - apply Ht in H; unfold ldo in *; lia.
      - revert H; lbl_solve. }
    pose proof (steps_relabel A0 (t ++ postD) x ldo HF _ _ Hst) as Hst'.
    assert (E : A0 ++ (Some ldo, COp x) :: t ++ postD = P)
      by (rewrite P_D; unfold LA; rewrite Hcons; reflexivity).
    rewrite <- E.
    replace pDo with (length A0) by len_solve.
    replace pT with (length A0 + length ((None, COp x) :: t))%nat; [exact Hst' |].
    unfold pT, la, LA; rewrite Hcons. cbn [label_first length]. len_solve.
Qed.

(** *** The exit test when it holds: [rt := 1], fall through, restore, jump out *)

Lemma exit_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e2 <> 0 ->
  steps P (mkC pT 0 (mkState R M)) (mkC (S pX) 0 (mkState R M)).
Proof.
  intros R M σ Hmod Hcl H0 He2.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT2 : run T2 (mkState R M) = mkState (rupd b 1 R) M).
  { unfold T2. rewrite (flag_block_spec e2 b _ σ).
    - cbn [regs mem]. now rewrite Hb0, Z.lxor_0_l, (truth_nz _ He2).
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  replace pT with (length (A0 ++ LA)) by len_solve.
  eapply steps_trans.
  { eapply (steps_ops_l ltest T2 P (A0 ++ LA)).
    unfold P, A0, LA, loop_code; norm_app; reflexivity. }
  rewrite HT2.
  replace (length (A0 ++ LA) + length T2)%nat with pQ by len_solve.
  (* BEQ rt r0 loop: rt = 1, not taken *)
  eapply steps_step.
  { eapply cstep_beq_direct_not_taken; [apply nth_Q |].
    cbn [regs]. rewrite rupd_same, rupd_other by auto. rewrite H0. discriminate. }
  (* XORI rt 1: rt := 0 *)
  eapply steps_step; [eapply cstep_op; apply nth_C2 |].
  cbn [step regs mem]. rewrite rupd_same, rupd_shadow.
  change (Z.lxor 1 1) with 0. rewrite rupd_zero by exact Hb0.
  (* BRA exit: a direct jump *)
  eapply steps_step.
  { eapply cstep_bra_direct; [apply nth_J | apply find_lexit | apply paired_J]. }
  (* exit: NOP *)
  apply steps_one.
  rewrite (cstep_op P pX 0 _ (Some lexit) nop) by apply nth_X.
  now rewrite step_nop.
Qed.

(** *** The exit test when it fails: [rt] stays 0, jump to [loop] (a NOP) *)

Lemma iter_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e2 = 0 ->
  steps P (mkC pT 0 (mkState R M)) (mkC pS 0 (mkState R M)).
Proof.
  intros R M σ Hmod Hcl H0 He2.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT2 : run T2 (mkState R M) = mkState R M).
  { unfold T2. rewrite (flag_block_spec e2 b _ σ).
    - cbn [regs mem]. rewrite Hb0, He2, truth_0. change (Z.lxor 0 0) with 0.
      now rewrite rupd_zero.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  replace pT with (length (A0 ++ LA)) by len_solve.
  eapply steps_trans.
  { eapply (steps_ops_l ltest T2 P (A0 ++ LA)).
    unfold P, A0, LA, loop_code; norm_app; reflexivity. }
  rewrite HT2.
  replace (length (A0 ++ LA) + length T2)%nat with pQ by len_solve.
  (* BEQ rt r0 loop: taken, br = 0, a direct jump *)
  eapply steps_step.
  { eapply cstep_beq_direct_taken; [apply nth_Q | apply find_lloop |].
    cbn [regs]. congruence. }
  (* loop: NOP *)
  apply steps_one.
  rewrite (cstep_op P pL 0 _ (Some lloop) nop) by apply nth_L.
  now rewrite step_nop.
Qed.

(** *** S2 *)

Lemma s2_steps : forall ms σ σ',
  frag_spec pb na nc σ σ' ms ->
  exists ms', steps P (mkC pS 0 ms) (mkC pR 0 ms') /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros ms σ σ' IH.
  destruct (IH A5 postS) as [ms' [Hst [Hm Hr]]]; [lbl_solve | lbl_solve |].
  exists ms'. split; [| split; assumption].
  rewrite <- P_S in Hst.
  replace pS with (length A5) by len_solve.
  replace pR with (length A5 + length pb)%nat by len_solve.
  exact Hst.
Qed.

(** *** Re-entry: [rt := e1 != 0], reaching the re-entry `BNE` *)

Lemma reentry_check : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  steps P (mkC pR 0 (mkState R M))
          (mkC pN3 0 (mkState (rupd b (truth (eval σ e1)) R) M)).
Proof.
  intros R M σ Hmod Hcl H0.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT1 : run T1 (mkState R M) = mkState (rupd b (truth (eval σ e1)) R) M).
  { unfold T1. rewrite (flag_block_spec e1 b _ σ).
    - cbn [regs mem]. now rewrite Hb0, Z.lxor_0_l.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  replace pR with (length (A5 ++ pb)) by len_solve.
  replace pN3 with (length (A5 ++ pb) + length T1)%nat by len_solve.
  rewrite <- HT1.
  eapply (steps_ops T1 P (A5 ++ pb)).
  unfold P, A5, A4, A1, A0, LA, loop_code; norm_app; reflexivity.
Qed.

(** A false re-entry assertion (as the source requires): [rt] is 0, jump back to [do]. *)
Lemma back_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 ->
  steps P (mkC pR 0 (mkState R M)) (mkC pDo 0 (mkState R M)).
Proof.
  intros R M σ Hmod Hcl H0 He1.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply steps_trans; [apply (reentry_check R M σ Hmod Hcl H0) |].
  rewrite He1, truth_0.
  rewrite rupd_zero by exact Hb0.
  eapply steps_step.
  { eapply cstep_bne_direct_not_taken; [apply nth_N3 | cbn [regs]; congruence]. }
  apply steps_one.
  eapply cstep_bra_direct; [apply nth_Bk | apply find_ldo | apply paired_Bk].
Qed.

(** A true re-entry assertion: [rt = 1] at the `BNE`, which jumps to [finish]. *)
Lemma back_violation : forall R M σ f,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 -> find_label fin P = Some f ->
  steps P (mkC pR 0 (mkState R M)) (mkC f 0 (mkState (rupd b 1 R) M)).
Proof.
  intros R M σ f Hmod Hcl H0 He1 Hf.
  eapply steps_trans; [apply (reentry_check R M σ Hmod Hcl H0) |].
  rewrite (truth_nz _ He1).
  apply steps_one.
  eapply cstep_bne_direct_taken; [apply nth_N3 | exact Hf |].
  cbn [regs]. rewrite rupd_same, rupd_other by lia. lia.
Qed.

(** *** The three shapes the loop's derivation takes, with explicit positions *)

Lemma loop_enter : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 ->
  steps P (mkC (length pre) 0 (mkState R M))
          (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M)).
Proof. intros; rewrite pDo_eq; eapply entry_steps; eassumption. Qed.

(** The last round: S1, then the exit test holds. *)
Lemma loop_last : forall R M σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  frag_spec pa (n + 4)%nat na σ σ' (mkState R M) -> eval σ' e2 <> 0 ->
  exists ms',
    steps P (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M))
            (mkC (length pre + length (loop_code fin b n e1 e2 pa pb)) 0 ms')
    /\ models ms' σ' /\ regs ms' = R.
Proof.
  intros R M σ σ' Hmod Hcl H0 IH He2.
  destruct (do_steps _ _ _ IH) as [[R1 M1] [Hst [Hm Hr]]]; cbn [regs] in Hr; subst R1.
  exists (mkState R M1). split; [| split; [exact Hm | reflexivity]].
  rewrite pDo_eq, len_loop_code.
  eapply steps_trans; [exact Hst |].
  eapply exit_steps; eassumption.
Qed.

(** A continuing round: S1, the exit test fails, S2, the re-entry assertion
    fails; back at [do] with the same registers. *)
Lemma loop_round : forall R M σ σ1 σ2,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  frag_spec pa (n + 4)%nat na σ σ1 (mkState R M) -> eval σ1 e2 = 0 ->
  (forall M1, models (mkState R M1) σ1 ->
              frag_spec pb na nc σ1 σ2 (mkState R M1)) ->
  eval σ2 e1 = 0 ->
  exists M2,
    steps P (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M))
            (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M2))
    /\ models (mkState R M2) σ2.
Proof.
  intros R M σ σ1 σ2 Hmod Hcl H0 IHa He2 IHc He1.
  destruct (do_steps _ _ _ IHa) as [[R1 M1] [Hst1 [Hm1 Hr1]]]; cbn [regs] in Hr1; subst R1.
  destruct (s2_steps _ _ _ (IHc M1 Hm1)) as [[R2 M2] [Hst2 [Hm2 Hr2]]];
    cbn [regs] in Hr2; subst R2.
  exists M2. split; [| exact Hm2].
  rewrite pDo_eq.
  eapply steps_trans; [exact Hst1 |].
  eapply steps_trans; [eapply iter_steps; eassumption |].
  eapply steps_trans; [exact Hst2 |].
  eapply back_steps; [exact Hm2 | exact Hcl | exact H0 | exact He1].
Qed.

(** The same, closed by the rest of the loop from [do] (the induction
    hypothesis on the remaining rounds). *)
Lemma loop_more : forall R M σ σ1 σ2 σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  frag_spec pa (n + 4)%nat na σ σ1 (mkState R M) -> eval σ1 e2 = 0 ->
  (forall M1, models (mkState R M1) σ1 ->
              frag_spec pb na nc σ1 σ2 (mkState R M1)) ->
  eval σ2 e1 = 0 ->
  (forall M2, models (mkState R M2) σ2 ->
     exists ms',
       steps P (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M2))
               (mkC (length pre + length (loop_code fin b n e1 e2 pa pb)) 0 ms')
       /\ models ms' σ' /\ regs ms' = R) ->
  exists ms',
    steps P (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M))
            (mkC (length pre + length (loop_code fin b n e1 e2 pa pb)) 0 ms')
    /\ models ms' σ' /\ regs ms' = R.
Proof.
  intros R M σ σ1 σ2 σ' Hmod Hcl H0 IHa He2 IHc He1 IHlp.
  destruct (loop_round R M σ σ1 σ2 Hmod Hcl H0 IHa He2 IHc He1) as [M2 [Hst Hm2]].
  destruct (IHlp M2 Hm2) as [ms' [Hst' [Hm' Hr']]].
  exists ms'. split; [eapply steps_trans; eassumption | auto].
Qed.

(** The first round with a true re-entry assertion: entry, S1, the exit
    test fails, S2, and the re-entry `BNE` jumps to [finish] with [rt = 1]. *)
Lemma loop_first_reentry_violation : forall R M σ σ1 σ2 f,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 ->
  frag_spec pa (n + 4)%nat na σ σ1 (mkState R M) -> eval σ1 e2 = 0 ->
  (forall M1, models (mkState R M1) σ1 ->
              frag_spec pb na nc σ1 σ2 (mkState R M1)) ->
  eval σ2 e1 <> 0 -> find_label fin P = Some f ->
  exists M2,
    steps P (mkC (length pre) 0 (mkState R M)) (mkC f 0 (mkState (rupd b 1 R) M2))
    /\ models (mkState R M2) σ2.
Proof.
  intros R M σ σ1 σ2 f Hmod Hcl H0 He1 IHa He2 IHc He1' Hf.
  destruct (do_steps _ _ _ IHa) as [[R1 M1] [Hst1 [Hm1 Hr1]]]; cbn [regs] in Hr1; subst R1.
  destruct (s2_steps _ _ _ (IHc M1 Hm1)) as [[R2 M2] [Hst2 [Hm2 Hr2]]];
    cbn [regs] in Hr2; subst R2.
  exists M2. split; [| exact Hm2].
  eapply steps_trans; [eapply entry_steps; eassumption |].
  eapply steps_trans; [exact Hst1 |].
  eapply steps_trans; [eapply iter_steps; eassumption |].
  eapply steps_trans; [exact Hst2 |].
  eapply back_violation; [exact Hm2 | exact Hcl | exact H0 | exact He1' | exact Hf].
Qed.

End LoopLayout.

(** ** Main theorem

    The analogue of [compile_c_spec] for [lstmt]: a compiled statement,
    embedded between any [pre] and [post] that do not define its labels,
    started with [br = 0] at its first line, reaches the line after its
    last with [br = 0], a memory representing the final store and the
    *same* register file.  Nothing is assumed about the `finish` label
    [fin]: on a valid execution no `BNE rt r0 finish` is taken.

    The proof is a mutual induction on the derivation ([exec_l_mut]); the
    loop rounds are the induction on [lp_l], with the invariant "at [do],
    [br = 0], the same registers (flag [rt = 0]), memory representing the
    current store". *)

Ltac loop_side :=
  match goal with
  | |- head_ok _ => eapply compile_l_head; eassumption
  | H : ?g |- ?g => exact H
  | |- _ => first [eassumption | lia]
  end.

Theorem compile_l_spec : forall fin st σ σ', exec_l st σ σ' ->
  forall b n p n' ms pre post,
  wf_lstmt st ->
  compile_l fin st b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  exists ms',
    steps (pre ++ p ++ post)
          (mkC (length pre) 0 ms) (mkC (length pre + length p) 0 ms')
    /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros fin st σ σ' Hex.
  induction Hex as [s σ σ' Hs | a c σ m σ' Ha IHa Hc IHc
                   | e1 a c e2 σ σ' He1 Ha IHa He2 | e1 a c e2 σ σ' He1 Hc IHc He2
                   | e1 a c e2 σ σ' He1 Hlp IHlp
                   | e1 a c e2 σ σ' Ha IHa He2
                   | e1 a c e2 σ σ1 σ2 σ' Ha IHa He2 Hc IHc He1 Hlp IHlp]
    using exec_l_mut with
    (P0 := fun e1 a c e2 σ σ' (_ : lp_l e1 a c e2 σ σ') =>
      forall b n pa na pc nc R M pre post,
      wf_lstmt a -> wf_lstmt c ->
      compile_l fin a (S b) (n + 4)%nat = (pa, na) ->
      compile_l fin c (S b) na = (pc, nc) ->
      b <> 0%nat -> models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
      (forall l, In l (labels pre)  -> ~ (n <= l < nc)%nat) ->
      (forall l, In l (labels post) -> ~ (n <= l < nc)%nat) ->
      exists ms',
        steps (pre ++ loop_code fin b n e1 e2 pa pc ++ post)
              (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M))
              (mkC (length pre + length (loop_code fin b n e1 e2 pa pc)) 0 ms')
        /\ models ms' σ' /\ regs ms' = R).
  - (* LBase *)
    intros b n p n' ms pre post Hwf Hcomp Hb Hmod Hcl H0 Hpre Hpost; simpl in Hcomp.
    injection Hcomp as <- <-.
    destruct (compile_at_spec b s σ σ' ms Hs Hwf Hmod Hcl) as [Hm Hr].
    exists (run (compile_at b s) ms). split; [| split; assumption].
    rewrite length_ops. eapply steps_ops. reflexivity.
  - (* LSeq *)
    intros b n p n' ms pre post Hwf Hcomp Hb Hmod Hcl H0 Hpre Hpost; simpl in Hcomp.
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_l fin a b n) as [p1 n1] eqn:E1.
    destruct (compile_l fin c b n1) as [p2 n2] eqn:E2.
    injection Hcomp as <- <-.
    destruct (compile_l_labels _ _ _ _ _ _ E1) as [Hle1 Hin1].
    destruct (compile_l_labels _ _ _ _ _ _ E2) as [Hle2 Hin2].
    destruct (IHa b n p1 n1 ms pre (p2 ++ post) Hwfa E1 Hb Hmod Hcl H0)
      as [ms1 [Hst1 [Hm1 Hr1]]].
    { intros l Hl; apply Hpre in Hl; lia. }
    { intros l Hl; rewrite labels_app, in_app_iff in Hl.
      destruct Hl as [Hl | Hl]; [apply Hin2 in Hl | apply Hpost in Hl]; lia. }
    assert (Hcl1 : clean_above b ms1) by (intros r Hr; rewrite Hr1; now apply Hcl).
    assert (H01 : regs ms1 0%nat = 0) by (rewrite Hr1; exact H0).
    destruct (IHc b n1 p2 n2 ms1 (pre ++ p1) post Hwfc E2 Hb Hm1 Hcl1 H01)
      as [ms2 [Hst2 [Hm2 Hr2]]].
    { intros l Hl; rewrite labels_app, in_app_iff in Hl.
      destruct Hl as [Hl | Hl]; [apply Hpre in Hl | apply Hin1 in Hl]; lia. }
    { intros l Hl; apply Hpost in Hl; lia. }
    exists ms2. split; [| split; [exact Hm2 | now rewrite Hr2, Hr1]].
    rewrite <- app_assoc, length_app in Hst2.
    rewrite length_app, <- app_assoc.
    eapply steps_trans; [exact Hst1 |].
    replace (length pre + (length p1 + length p2))%nat
      with (length pre + length p1 + length p2)%nat by lia.
    exact Hst2.
  - (* LIf, then path: the layout lemmas of CompileIf.v *)
    intros b n p n' ms pre post Hwf Hcomp Hb Hmod Hcl H0 Hpre Hpost; simpl in Hcomp.
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_l fin a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_l fin c (S b) na) as [pb nb] eqn:Eb.
    injection Hcomp as <- <-.
    destruct (compile_l_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_l_labels _ _ _ _ _ _ Eb) as [Hleb Hinb].
    destruct ms as [R M].
    assert (HclS : clean_above (S b) (mkState R M)) by (intros r Hr; apply Hcl; lia).
    destruct (IHa (S b) (n + 5)%nat pa na (mkState R M)
                  (pre ++ ops (flag_block e1 b)
                       ++ [(Some (S n), CBeq b 0%nat n); (None, COp (IXori b 1))])
                  ((None, COp (IXori b 1))
                   :: ops_l (n + 2)%nat (flag_block e2 b)
                   ++ (Some (n + 3)%nat, CBra (n + 4)%nat)
                   :: (Some n, CBra (S n))
                   :: pb
                   ++ (None, CBra (n + 2)%nat)
                   :: (Some (n + 4)%nat, CBra (n + 3)%nat)
                   :: (None, CBne b 0%nat fin)
                   :: post)
                  Hwfa Ea ltac:(lia) Hmod HclS H0)
      as [[R' M'] [Hst [Hm' Hr']]].
    { intros l Hl; in_labels Hl;
      repeat match goal with
      | H : In _ (labels pre) |- _ => apply Hpre in H
      end; subst; lia. }
    { intros l Hl; in_labels Hl;
      repeat match goal with
      | H : In _ (labels pb) |- _ => apply Hinb in H
      | H : In _ (labels post) |- _ => apply Hpost in H
      end; subst; lia. }
    cbn [regs] in Hr'.
    exists (mkState R M'). split; [| split; [exact Hm' | reflexivity]].
    apply (if_true_run pre post b n e1 e2 pa pb fin Hb) with (σ := σ) (σ' := σ') (R' := R');
      try assumption.
    all: intros l Hl;
         first [apply Hpre in Hl | apply Hpost in Hl | apply Hina in Hl | apply Hinb in Hl];
         lia.
  - (* LIf, else path *)
    intros b n p n' ms pre post Hwf Hcomp Hb Hmod Hcl H0 Hpre Hpost; simpl in Hcomp.
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_l fin a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_l fin c (S b) na) as [pb nb] eqn:Eb.
    injection Hcomp as <- <-.
    destruct (compile_l_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_l_labels _ _ _ _ _ _ Eb) as [Hleb Hinb].
    destruct ms as [R M].
    assert (HclS : clean_above (S b) (mkState R M)) by (intros r Hr; apply Hcl; lia).
    destruct (IHc (S b) na pb nb (mkState R M)
                  (pre ++ ops (flag_block e1 b)
                       ++ (Some (S n), CBeq b 0%nat n)
                       :: (None, COp (IXori b 1))
                       :: pa
                       ++ (None, COp (IXori b 1))
                       :: ops_l (n + 2)%nat (flag_block e2 b)
                       ++ [(Some (n + 3)%nat, CBra (n + 4)%nat); (Some n, CBra (S n))])
                  ((None, CBra (n + 2)%nat) :: (Some (n + 4)%nat, CBra (n + 3)%nat)
                   :: (None, CBne b 0%nat fin) :: post)
                  Hwfc Eb ltac:(lia) Hmod HclS H0)
      as [[R' M'] [Hst [Hm' Hr']]].
    { intros l Hl; in_labels Hl;
      repeat match goal with
      | H : In _ (labels pa) |- _ => apply Hina in H
      | H : In _ (labels pre) |- _ => apply Hpre in H
      end; subst; lia. }
    { intros l Hl; in_labels Hl;
      repeat match goal with
      | H : In _ (labels post) |- _ => apply Hpost in H
      end; subst; lia. }
    cbn [regs] in Hr'.
    exists (mkState R M'). split; [| split; [exact Hm' | reflexivity]].
    apply (if_false_run pre post b n e1 e2 pa pb fin Hb) with (σ := σ) (σ' := σ') (R' := R');
      try assumption.
    all: intros l Hl;
         first [apply Hpre in Hl | apply Hpost in Hl | apply Hina in Hl | apply Hinb in Hl];
         lia.
  - (* LLoop: the entry assertion, then the rounds from [do] *)
    intros b n p n' ms pre post Hwf Hcomp Hb Hmod Hcl H0 Hpre Hpost; simpl in Hcomp.
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_l fin a (S b) (n + 4)%nat) as [pa na] eqn:Ea.
    destruct (compile_l fin c (S b) na) as [pc nc] eqn:Ec.
    injection Hcomp as <- <-.
    destruct (compile_l_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_l_labels _ _ _ _ _ _ Ec) as [Hlec Hinc].
    destruct ms as [R M].
    destruct (IHlp b n pa na pc nc R M pre post Hwfa Hwfc Ea Ec Hb Hmod Hcl H0 Hpre Hpost)
      as [ms' [Hst [Hm Hr]]].
    exists ms'. split; [| split; [exact Hm | exact Hr]].
    eapply steps_trans; [| exact Hst].
    eapply (loop_enter pre post b n na nc e1 e2 pa pc fin).
    all: loop_side.
  - (* LP_One: S1, the exit test holds *)
    intros b n pa na pc nc R M pre post Hwfa Hwfc Ea Ec Hb Hmod Hcl H0 Hpre Hpost.
    destruct (compile_l_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_l_labels _ _ _ _ _ _ Ec) as [Hlec Hinc].
    assert (IHa' : frag_spec pa (n + 4)%nat na σ σ' (mkState R M)).
    { intros pre' post' Hpre' Hpost'.
      apply (IHa (S b) (n + 4)%nat pa na (mkState R M) pre' post' Hwfa Ea ltac:(lia) Hmod);
        [intros r Hr; apply Hcl; lia | exact H0 | exact Hpre' | exact Hpost']. }
    eapply (loop_last pre post b n na nc e1 e2 pa pc fin).
    all: loop_side.
  - (* LP_More: S1, the test fails, S2, the re-entry assertion fails, and again *)
    intros b n pa na pc nc R M pre post Hwfa Hwfc Ea Ec Hb Hmod Hcl H0 Hpre Hpost.
    destruct (compile_l_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_l_labels _ _ _ _ _ _ Ec) as [Hlec Hinc].
    assert (IHa' : frag_spec pa (n + 4)%nat na σ σ1 (mkState R M)).
    { intros pre' post' Hpre' Hpost'.
      apply (IHa (S b) (n + 4)%nat pa na (mkState R M) pre' post' Hwfa Ea ltac:(lia) Hmod);
        [intros r Hr; apply Hcl; lia | exact H0 | exact Hpre' | exact Hpost']. }
    assert (IHc' : forall M1, models (mkState R M1) σ1 ->
                   frag_spec pc na nc σ1 σ2 (mkState R M1)).
    { intros M1 Hm1 pre' post' Hpre' Hpost'.
      apply (IHc (S b) na pc nc (mkState R M1) pre' post' Hwfc Ec ltac:(lia) Hm1);
        [intros r Hr; apply Hcl; lia | exact H0 | exact Hpre' | exact Hpost']. }
    assert (IHlp' : forall M2, models (mkState R M2) σ2 ->
       exists ms',
         steps (pre ++ loop_code fin b n e1 e2 pa pc ++ post)
               (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M2))
               (mkC (length pre + length (loop_code fin b n e1 e2 pa pc)) 0 ms')
         /\ models ms' σ' /\ regs ms' = R).
    { intros M2 Hm2. exact (IHlp b n pa na pc nc R M2 pre post Hwfa Hwfc Ea Ec Hb Hm2 Hcl H0
                                 Hpre Hpost). }
    eapply (loop_more pre post b n na nc e1 e2 pa pc fin) with (σ := σ) (σ1 := σ1) (σ2 := σ2).
    all: loop_side.
Qed.

(** ** Whole program

    `finish` is label [fin_label] = 0 (CompileIf.v), the statement's labels
    start at 1, and [with_finish] appends the `finish:` line.  Every valid
    execution runs to the end of the program without ever branching to it. *)

Lemma with_finish_nop : forall p, with_finish p = p ++ [(Some fin_label, COp nop)].
Proof. reflexivity. Qed.

Corollary compile_l_program : forall st σ σ' p n' ms,
  exec_l st σ σ' -> wf_lstmt st ->
  compile_l fin_label st scratch 1%nat = (p, n') ->
  models ms σ -> clean_above scratch ms -> regs ms 0%nat = 0 ->
  exists ms' fuel,
    exec_fuel fuel (with_finish p) (mkC 0%nat 0 ms)
    = Some (mkC (length (with_finish p)) 0 ms')
    /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros st σ σ' p n' ms Hex Hwf Hc Hmod Hcl H0.
  destruct (compile_l_spec fin_label st σ σ' Hex scratch 1%nat p n' ms []
              [(Some fin_label, COp nop)] Hwf Hc)
    as [ms' [Hst [Hm Hr]]];
    [unfold scratch; lia | exact Hmod | exact Hcl | exact H0
    | intros l Hl; destruct Hl
    | intros l Hl; destruct Hl as [<- | []]; unfold fin_label; lia |].
  exists ms'.
  cbn [app length Nat.add] in Hst.
  assert (Hst' : steps (with_finish p) (mkC 0%nat 0 ms) (mkC (S (length p)) 0 ms')).
  { rewrite with_finish_nop.
    eapply steps_trans; [exact Hst |].
    apply steps_one.
    rewrite (cstep_op (p ++ [(Some fin_label, COp nop)]) (length p) 0 ms'
                      (Some fin_label) nop) by (eapply nth_error_at; reflexivity).
    now rewrite step_nop. }
  assert (Hlen : length (with_finish p) = S (length p))
    by (rewrite with_finish_nop, length_app; cbn [length]; lia).
  destruct (steps_exec_fuel _ _ _ Hst') as [fuel Hf].
  { cbn [cpc]. apply nth_error_None. lia. }
  exists fuel. rewrite Hlen. auto.
Qed.

(** ** Violated assertions reach [finish] with a dirty flag

    Each theorem below starts the compiled statement in a state from which
    Janus would abort with an assertion failure, and shows that the code
    jumps to [finish] (wherever the enclosing program put it) with the flag
    register [b] equal to 1 and every other register unchanged — the
    register file `pisa_interp.py` rejects at FINISH.  The source has no
    execution in these situations ([exec_l] requires the assertions), so
    [compile_l_spec] says nothing about them; these theorems do. *)

Theorem compile_l_if_violation : forall fin e1 a c e2 σ σ' b n p n' ms pre post f,
  wf_lstmt a -> wf_lstmt c ->
  (eval σ e1 <> 0 /\ exec_l a σ σ' /\ eval σ' e2 = 0) \/
  (eval σ e1 = 0 /\ exec_l c σ σ' /\ eval σ' e2 <> 0) ->
  compile_l fin (LIf e1 a c e2) b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  find_label fin (pre ++ p ++ post) = Some f ->
  exists M',
    steps (pre ++ p ++ post) (mkC (length pre) 0 ms)
          (mkC f 0 (mkState (rupd b 1 (regs ms)) M'))
    /\ models (mkState (regs ms) M') σ'.
Proof.
  intros fin e1 a c e2 σ σ' b n p n' ms pre post f Hwfa Hwfc Hcase Hcomp Hb Hmod Hcl H0
         Hpre Hpost Hf; simpl in Hcomp.
  destruct (compile_l fin a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
  destruct (compile_l fin c (S b) na) as [pb nb] eqn:Eb.
  injection Hcomp as <- <-.
  destruct (compile_l_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
  destruct (compile_l_labels _ _ _ _ _ _ Eb) as [Hleb Hinb].
  destruct ms as [R M]; cbn [regs mem] in *.
  assert (HclS : clean_above (S b) (mkState R M)) by (intros r Hr; apply Hcl; lia).
  destruct Hcase as [[He1 [Ha He2]] | [He1 [Hc He2]]].
  - destruct (compile_l_spec fin a σ σ' Ha (S b) (n + 5)%nat pa na (mkState R M)
                (pre ++ ops (flag_block e1 b)
                     ++ [(Some (S n), CBeq b 0%nat n); (None, COp (IXori b 1))])
                ((None, COp (IXori b 1))
                 :: ops_l (n + 2)%nat (flag_block e2 b)
                 ++ (Some (n + 3)%nat, CBra (n + 4)%nat)
                 :: (Some n, CBra (S n))
                 :: pb
                 ++ (None, CBra (n + 2)%nat)
                 :: (Some (n + 4)%nat, CBra (n + 3)%nat)
                 :: (None, CBne b 0%nat fin)
                 :: post)
                Hwfa Ea ltac:(lia) Hmod HclS H0)
      as [[R' M'] [Hst [Hm' Hr']]].
    { intros l Hl; in_labels Hl;
      repeat match goal with
      | H : In _ (labels pre) |- _ => apply Hpre in H
      end; subst; lia. }
    { intros l Hl; in_labels Hl;
      repeat match goal with
      | H : In _ (labels pb) |- _ => apply Hinb in H
      | H : In _ (labels post) |- _ => apply Hpost in H
      end; subst; lia. }
    cbn [regs] in Hr'.
    exists M'. split; [| rewrite <- Hr'; exact Hm'].
    apply (if_true_violation pre post b n e1 e2 pa pb fin Hb)
      with (σ := σ) (σ' := σ') (R' := R'); try assumption.
    all: intros l Hl;
         first [apply Hpre in Hl | apply Hpost in Hl | apply Hina in Hl | apply Hinb in Hl];
         lia.
  - destruct (compile_l_spec fin c σ σ' Hc (S b) na pb nb (mkState R M)
                (pre ++ ops (flag_block e1 b)
                     ++ (Some (S n), CBeq b 0%nat n)
                     :: (None, COp (IXori b 1))
                     :: pa
                     ++ (None, COp (IXori b 1))
                     :: ops_l (n + 2)%nat (flag_block e2 b)
                     ++ [(Some (n + 3)%nat, CBra (n + 4)%nat); (Some n, CBra (S n))])
                ((None, CBra (n + 2)%nat) :: (Some (n + 4)%nat, CBra (n + 3)%nat)
                 :: (None, CBne b 0%nat fin) :: post)
                Hwfc Eb ltac:(lia) Hmod HclS H0)
      as [[R' M'] [Hst [Hm' Hr']]].
    { intros l Hl; in_labels Hl;
      repeat match goal with
      | H : In _ (labels pa) |- _ => apply Hina in H
      | H : In _ (labels pre) |- _ => apply Hpre in H
      end; subst; lia. }
    { intros l Hl; in_labels Hl;
      repeat match goal with
      | H : In _ (labels post) |- _ => apply Hpost in H
      end; subst; lia. }
    cbn [regs] in Hr'.
    exists M'. split; [| rewrite <- Hr'; exact Hm'].
    apply (if_false_violation pre post b n e1 e2 pa pb fin Hb)
      with (σ := σ) (σ' := σ') (R' := R'); try assumption.
    all: intros l Hl;
         first [apply Hpre in Hl | apply Hpost in Hl | apply Hina in Hl | apply Hinb in Hl];
         lia.
Qed.

Theorem compile_l_loop_entry_violation : forall fin e1 a c e2 σ b n p n' ms pre post f,
  compile_l fin (LLoop e1 a c e2) b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  eval σ e1 = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  find_label fin (pre ++ p ++ post) = Some f ->
  steps (pre ++ p ++ post) (mkC (length pre) 0 ms)
        (mkC f 0 (mkState (rupd b 1 (regs ms)) (mem ms))).
Proof.
  intros fin e1 a c e2 σ b n p n' ms pre post f Hcomp Hb Hmod Hcl H0 He1 Hpre Hpost Hf;
    simpl in Hcomp.
  destruct (compile_l fin a (S b) (n + 4)%nat) as [pa na] eqn:Ea.
  destruct (compile_l fin c (S b) na) as [pc nc] eqn:Ec.
  injection Hcomp as <- <-.
  destruct (compile_l_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
  destruct (compile_l_labels _ _ _ _ _ _ Ec) as [Hlec Hinc].
  destruct ms as [R M]; cbn [regs mem] in *.
  eapply (entry_violation pre post b n na nc e1 e2 pa pc fin).
  all: loop_side.
Qed.

Theorem compile_l_loop_reentry_violation :
  forall fin e1 a c e2 σ σ1 σ2 b n p n' ms pre post f,
  wf_lstmt a -> wf_lstmt c ->
  eval σ e1 <> 0 -> exec_l a σ σ1 -> eval σ1 e2 = 0 ->
  exec_l c σ1 σ2 -> eval σ2 e1 <> 0 ->
  compile_l fin (LLoop e1 a c e2) b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  find_label fin (pre ++ p ++ post) = Some f ->
  exists M2,
    steps (pre ++ p ++ post) (mkC (length pre) 0 ms)
          (mkC f 0 (mkState (rupd b 1 (regs ms)) M2))
    /\ models (mkState (regs ms) M2) σ2.
Proof.
  intros fin e1 a c e2 σ σ1 σ2 b n p n' ms pre post f Hwfa Hwfc He1 Ha He2 Hc He1'
         Hcomp Hb Hmod Hcl H0 Hpre Hpost Hf; simpl in Hcomp.
  destruct (compile_l fin a (S b) (n + 4)%nat) as [pa na] eqn:Ea.
  destruct (compile_l fin c (S b) na) as [pc nc] eqn:Ec.
  injection Hcomp as <- <-.
  destruct (compile_l_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
  destruct (compile_l_labels _ _ _ _ _ _ Ec) as [Hlec Hinc].
  destruct ms as [R M]; cbn [regs mem] in *.
  assert (IHa' : frag_spec pa (n + 4)%nat na σ σ1 (mkState R M)).
  { intros pre' post' Hpre' Hpost'.
    apply (compile_l_spec fin a σ σ1 Ha (S b) (n + 4)%nat pa na (mkState R M) pre' post'
             Hwfa Ea ltac:(lia) Hmod);
      [intros r Hr; apply Hcl; lia | exact H0 | exact Hpre' | exact Hpost']. }
  assert (IHc' : forall M1, models (mkState R M1) σ1 ->
                 frag_spec pc na nc σ1 σ2 (mkState R M1)).
  { intros M1 Hm1 pre' post' Hpre' Hpost'.
    apply (compile_l_spec fin c σ1 σ2 Hc (S b) na pc nc (mkState R M1) pre' post'
             Hwfc Ec ltac:(lia) Hm1);
      [intros r Hr; apply Hcl; lia | exact H0 | exact Hpre' | exact Hpost']. }
  eapply (loop_first_reentry_violation pre post b n na nc e1 e2 pa pc fin)
    with (σ := σ) (σ1 := σ1).
  all: loop_side.
Qed.

(** ** Executable checks

    Variables: [x0 x1 x2 c y0 y1 d] at addresses [0 .. 6].  The same
    programs, written in Janus, were compiled by `codegen.py` and run on
    `pisa_interp.py`, and the code emitted here was run on `pisa_interp.py`
    too (tools/rocq_loop_crosscheck.py); see MANIFEST.md. *)

Definition observe_l (p : lprog) (r : option cstate) :=
  match r with
  | Some c => (Nat.eqb (cpc c) (length p), cbr c,
               map (fun a => mem (cst c) (Z.of_nat a)) (seq 0 7),
               map (regs (cst c)) (seq 3 6))
  | None => (false, 1, [], [])
  end.

(** The compiled statement followed by the `finish:` line. *)
Definition prog_l (st : lstmt) : lprog := with_finish (fst (compile_l fin_label st scratch 1%nat)).

Definition run_l (st : lstmt) :=
  let p := prog_l st in observe_l p (exec_fuel 5000 p (mkC 0%nat 0 zero_state)).

(** [x1 <=> x2 ; x0 <=> x1]: moves the token one place. *)
Definition rot : lstmt := LBase (Seq (Swap 1%nat 2%nat) (Swap 0%nat 1%nat)).
Definition inc (x : var) (k : Z) : lstmt := LBase (Assign x AAdd (Cst k)).

(** [x0 += 1 ; from x0 do c += 1 loop rot until x2]: three rounds of S1,
    two of S2. *)
Definition prog_loop : lstmt :=
  LSeq (inc 0%nat 1) (LLoop (Var 0%nat) (inc 3%nat 1) rot (Var 2%nat)).

Example ex_loop :
  run_l prog_loop = (true, 0, [0; 0; 1; 3; 0; 0; 0], [0; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** An [If] inside S1: [if x2 then c += 10 else c += 1 fi x2], so c = 1 + 1 + 10. *)
Definition prog_loop_if : lstmt :=
  LSeq (inc 0%nat 1)
       (LLoop (Var 0%nat) (LIf (Var 2%nat) (inc 3%nat 10) (inc 3%nat 1) (Var 2%nat))
              rot (Var 2%nat)).

Example ex_loop_if :
  run_l prog_loop_if = (true, 0, [0; 0; 1; 12; 0; 0; 0], [0; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** An empty S1: the label `from_do` sits on a NOP. *)
Definition prog_loop_skip : lstmt :=
  LSeq (inc 0%nat 1) (LLoop (Var 0%nat) (LBase Skip) rot (Var 2%nat)).

Example ex_loop_skip :
  run_l prog_loop_skip = (true, 0, [0; 0; 1; 0; 0; 0; 0], [0; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

Example ex_loop_skip_nop :
  nth_error (fst (compile_l fin_label prog_loop_skip scratch 1%nat))
            (length (compile_at scratch (Assign 0%nat AAdd (Cst 1)))
             + S (S (length (flag_block (Var 0%nat) scratch))))
  = Some (Some 4%nat, COp nop).
Proof. vm_compute. reflexivity. Qed.

(** Nested loops: the inner [from y0 do d += 1 loop y0 <=> y1 until y1]
    runs twice per outer round (its flag is r4, its bodies at base 5); the
    outer S2 resets [y0, y1].  d = 3 * 2. *)
Definition prog_nested_loop : lstmt :=
  LSeq (inc 0%nat 1)
  (LSeq (inc 4%nat 1)
        (LLoop (Var 0%nat)
               (LLoop (Var 4%nat) (inc 6%nat 1) (LBase (Swap 4%nat 5%nat)) (Var 5%nat))
               (LSeq rot (LBase (Swap 4%nat 5%nat)))
               (Var 2%nat))).

Example ex_nested_loop :
  run_l prog_nested_loop = (true, 0, [0; 0; 1; 0; 0; 1; 6], [0; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** *** Non-Boolean tests (valid Janus; truth is nonzero) *)

(** [x2 += 5 ; if x2 then c += 10 else skip fi c]: test 5, assertion 10. *)
Definition prog_if5 : lstmt :=
  LSeq (inc 2%nat 5) (LIf (Var 2%nat) (inc 3%nat 10) (LBase Skip) (Var 3%nat)).

Example ex_if5 :
  run_l prog_if5 = (true, 0, [0; 0; 5; 10; 0; 0; 0], [0; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** [x1 += 7 ; if x0 then c += 1 else x1 += 1 fi x0 + x1 - x1]: the else
    path with a test of 0 and an assertion that evaluates to 0 through
    arithmetic. *)
Definition prog_if_else : lstmt :=
  LSeq (inc 1%nat 7)
       (LIf (Var 0%nat) (inc 3%nat 1) (inc 1%nat 1)
            (Bin OSub (Bin OAdd (Var 0%nat) (Var 1%nat)) (Var 1%nat))).

Example ex_if_else :
  run_l prog_if_else = (true, 0, [0; 8; 0; 0; 0; 0; 0], [0; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** [x0 += 3 ; from x0 do c += 1 loop rot until x2]: the token is 3, so
    the entry assertion, the exit test and the re-entry assertion all see
    values in [{0, 3}]. *)
Definition prog_loop3 : lstmt :=
  LSeq (inc 0%nat 3) (LLoop (Var 0%nat) (inc 3%nat 1) rot (Var 2%nat)).

Example ex_loop3 :
  run_l prog_loop3 = (true, 0, [0; 0; 3; 3; 0; 0; 0], [0; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** The constants 0 and 1 are left alone by `_as_flag` ([is_bool_const]):
    [if 1 then c += 1 else skip fi 1 ; from 1 do c += 2 loop skip until 1]. *)
Definition prog_const : lstmt :=
  LSeq (LIf (Cst 1) (inc 3%nat 1) (LBase Skip) (Cst 1))
       (LLoop (Cst 1) (inc 3%nat 2) (LBase Skip) (Cst 1)).

Example ex_const :
  run_l prog_const = (true, 0, [0; 0; 0; 3; 0; 0; 0], [0; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** *** Violated assertions are detected

    [prog_v1]: the entry assertion fails ([x0 = 0] on entry).
    [prog_v2]: the re-entry assertion fails ([x0] stays 1 after S2).
    [prog_if_v]: the `fi` assertion fails ([x0 = 0] after the then branch).
    Janus rejects all three (PyJanus: "Assertion failed"); the source
    relation has no execution.  The compiled code jumps to `finish` with
    the flag r3 = 1 — `pisa_interp.py` raises "garbage left in registers"
    there.  (Under the old `XOR rt rt` layout [prog_v1] and [prog_v2] ran
    to the end with every register clean.) *)

Definition zero_store : store := fun _ => 0.

Definition prog_v1 : lstmt :=
  LSeq (inc 2%nat 1) (LLoop (Var 0%nat) (inc 3%nat 1) rot (Var 2%nat)).

Definition prog_v2 : lstmt :=
  LSeq (inc 0%nat 1) (LLoop (Var 0%nat) (inc 3%nat 1) (inc 2%nat 1) (Var 2%nat)).

Definition prog_if_v : lstmt :=
  LSeq (inc 2%nat 5) (LIf (Var 2%nat) (inc 3%nat 1) (LBase Skip) (Var 0%nat)).

Ltac inv H := inversion H; subst; clear H.

Lemma entry_violation_no_exec : forall σ', ~ exec_l prog_v1 zero_store σ'.
Proof.
  intros σ' H; unfold prog_v1, inc in H.
  inv H.
  match goal with H : exec_l (LBase _) _ _ |- _ => inv H end.
  match goal with H : exec (Assign _ _ _) _ _ |- _ => inv H end.
  match goal with H : exec_l (LLoop _ _ _ _) _ _ |- _ => inv H end.
  match goal with H : eval _ (Var 0%nat) <> 0 |- _ => cbv in H; now apply H end.
Qed.

Lemma reentry_violation_no_exec : forall σ', ~ exec_l prog_v2 zero_store σ'.
Proof.
  intros σ' H; unfold prog_v2, inc in H.
  inv H.
  match goal with H : exec_l (LBase _) _ _ |- _ => inv H end.
  match goal with H : exec (Assign _ _ _) _ _ |- _ => inv H end.
  match goal with H : exec_l (LLoop _ _ _ _) _ _ |- _ => inv H end.
  match goal with H : lp_l _ _ _ _ _ _ |- _ => inv H end.
  - (* exit after one S1: but x2 = 0 *)
    match goal with H : exec_l (LBase _) _ _ |- _ => inv H end.
    match goal with H : exec (Assign _ _ _) _ _ |- _ => inv H end.
    match goal with H : eval _ (Var 2%nat) <> 0 |- _ => cbv in H; now apply H end.
  - (* continue: but x0 = 1 at re-entry *)
    repeat match goal with H : exec_l (LBase _) _ _ |- _ => inv H end.
    repeat match goal with H : exec (Assign _ _ _) _ _ |- _ => inv H end.
    match goal with H : eval _ (Var 0%nat) = 0 |- _ => cbv in H; discriminate end.
Qed.

Lemma if_violation_no_exec : forall σ', ~ exec_l prog_if_v zero_store σ'.
Proof.
  intros σ' H; unfold prog_if_v, inc in H.
  inv H.
  match goal with H : exec_l (LBase _) _ _ |- _ => inv H end.
  match goal with H : exec (Assign _ _ _) _ _ |- _ => inv H end.
  match goal with H : exec_l (LIf _ _ _ _) _ _ |- _ => inv H end.
  - (* then path: but x0 = 0 afterwards *)
    match goal with H : exec_l (LBase _) _ _ |- _ => inv H end.
    match goal with H : exec (Assign _ _ _) _ _ |- _ => inv H end.
    match goal with H : eval _ (Var 0%nat) <> 0 |- _ => cbv in H; now apply H end.
  - (* else path: but x2 = 5 *)
    match goal with H : eval _ (Var 2%nat) = 0 |- _ => cbv in H; discriminate end.
Qed.

(** The entry violation: S1 never runs ([c = 0]), r3 = 1 at the end. *)
Example ex_entry_violation_detected :
  run_l prog_v1 = (true, 0, [0; 0; 1; 0; 0; 0; 0], [1; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** The re-entry violation: one round of S1 and S2, then r3 = 1. *)
Example ex_reentry_violation_detected :
  run_l prog_v2 = (true, 0, [1; 0; 1; 1; 0; 0; 0], [1; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

Example ex_if_violation_detected :
  run_l prog_if_v = (true, 0, [0; 0; 5; 1; 0; 0; 0], [1; 0; 0; 0; 0; 0]).
Proof. vm_compute. reflexivity. Qed.

(** The dirty flag is reached through `BNE rt r0 finish`: without a
    `finish` line the three runs get stuck at that branch, while the valid
    programs never touch it. *)
Example ex_violations_branch_to_finish :
  map (fun st => match exec_fuel 5000 (fst (compile_l fin_label st scratch 1%nat))
                                 (mkC 0%nat 0 zero_state) with
                 | None => true | Some _ => false end)
      [prog_v1; prog_v2; prog_if_v; prog_loop; prog_if5; prog_loop3]
  = [true; true; true; false; false; false].
Proof. vm_compute. reflexivity. Qed.

(** ** Axiom footprint *)

Print Assumptions compile_l_spec.
Print Assumptions compile_l_program.
Print Assumptions compile_l_if_violation.
Print Assumptions compile_l_loop_entry_violation.
Print Assumptions compile_l_loop_reentry_violation.
Print Assumptions exec_l_rev.
Print Assumptions steps_relabel.
