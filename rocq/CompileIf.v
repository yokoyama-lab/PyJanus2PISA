(** * CompileIf.v — compiling [If] to paired branches, and its correctness

    This file extends the verified translation of Compile.v with Janus's
    conditional

      if e1 then S1 else S2 fi e2

    and proves it correct on the control-flow machine of PISACtl.v.  The
    emitted code is the layout `_gen_if` in `codegen.py` produces (Axelsen,
    CC 2011, Fig. 9–11):

<<
            <eval e1 -> re> ; XOR rt re ; <uneval e1>      -- rt := e1
   test:    BEQ rt r0 false
            XORI rt 1                                      -- rt := 0 on the then path
            <S1>
            XORI rt 1                                      -- rt := 1 again
   assert:  <eval e2 -> re> ; XOR rt re ; <uneval e2>      -- rt := rt xor e2
   true:    BRA end                 \ Pendulum pair
   false:   BRA test                / partner of the BEQ
            <S2>
            BRA assert
   end:     BRA true                / partner of `true`
>>

    `codegen.py` has since grown two things this model does not have:
    tests and assertions that are not 0/1 are compiled as `e != 0`, and
    `BNE rt r0 finish` follows `end:` so that a violated assertion halts.
    On the fragment proved here [rt] is 0 after `end:` and that branch is a
    fall-through, but the instruction is not part of [compile_c].

    Two facts about this layout drive the proof:

    - The flag register [rt] is allocated *before* the bodies (it is the
      lowest free register, [scratch]), so the bodies are compiled with
      scratch base [S rt].  Compile.v hard-wires the base to [scratch];
      [compile_at] below is the same compiler with the base as a parameter,
      and [compile_at_scratch] shows it coincides with [compile].
    - The exit assertion is checked by XOR-ing [eval e2] into the flag,
      which is [1] on the then path and [0] on the else path.  This is
      garbage-free exactly when the tests are *Boolean-valued* (0 or 1): for
      `if 5 then skip else skip fi 7` — a valid Janus program, truth being
      nonzero — the emitted code leaves [5 xor 7] in [rt].  [exec_c] below
      therefore uses [eval σ e1 = 1] / [= 0] rather than [<> 0] / [= 0]; it
      is a sub-relation of the Janus semantics, and it is precisely the
      fragment on which `codegen.py` is a clean translation.

    Main result: [compile_c_spec] — semantic preservation and a clean
    register file, for a fragment embedded at any position of a larger
    labeled program with fresh labels; [compile_c_program] is the closed
    whole-program corollary with the fuel executor. *)

From Stdlib Require Import ZArith List Lia Bool.
From Stdlib Require Import FunctionalExtensionality.
Require Import PISA Src Compile PISACtl.
Import ListNotations.
Open Scope Z_scope.

(** ** The straight-line compiler with a scratch base

    [gen_expr] already takes its target register; the statement level of
    Compile.v does not.  These are the same definitions with [scratch]
    replaced by a parameter [b]. *)

Definition gen_assign_at (b : reg) (x : var) (o : aop) (e : expr) : code :=
  gen_expr e b
  ++ [ IAddi (S b) (Z.of_nat x)
     ; IExch (S (S b)) (S b)
     ; aop_instr o (S (S b)) b
     ; IExch (S (S b)) (S b)
     ; ISubi (S b) (Z.of_nat x) ]
  ++ invert_code (gen_expr e b).

Definition gen_swap_at (b : reg) (x y : var) : code :=
  [ IAddi b (Z.of_nat x)
  ; IAddi (S b) (Z.of_nat y)
  ; IExch (S (S b)) b
  ; IExch (S (S (S b))) (S b)
  ; IExch (S (S (S b))) b
  ; IExch (S (S b)) (S b)
  ; ISubi b (Z.of_nat x)
  ; ISubi (S b) (Z.of_nat y) ].

Fixpoint compile_at (b : reg) (st : stmt) : code :=
  match st with
  | Skip         => []
  | Assign x o e => gen_assign_at b x o e
  | Swap x y     => gen_swap_at b x y
  | Seq s1 s2    => compile_at b s1 ++ compile_at b s2
  end.

Lemma compile_at_scratch : forall st, compile_at scratch st = compile st.
Proof. induction st; simpl; try reflexivity. now rewrite IHst1, IHst2. Qed.

Lemma wf_compile_at : forall b st, wf_code (compile_at b st).
Proof.
  intros b st; induction st as [| x o e | x y | s1 IH1 s2 IH2]; simpl.
  - apply Forall_nil.
  - unfold gen_assign_at.
    apply wf_code_app; [apply wf_gen_expr |].
    repeat (apply Forall_cons; [destruct o; simpl; first [exact I | lia] |]).
    apply wf_invert_code, wf_gen_expr.
  - unfold gen_swap_at; wf_list.
  - now apply wf_code_app.
Qed.

(** The proofs below are those of Compile.v with [scratch] generalized. *)

Lemma gen_assign_at_spec : forall b x o e s σ,
  occurs x e = false ->
  models s σ -> clean_above b s ->
  run (gen_assign_at b x o e) s
  = mkState (regs s)
            (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) (mem s)).
Proof.
  intros b x o e [R M] σ Hocc Hmod Hcl; cbn [regs mem] in *.
  assert (H0 : R b = 0) by (apply Hcl; lia).
  assert (H1 : R (S b) = 0) by (apply Hcl; lia).
  assert (H2 : R (S (S b)) = 0) by (apply Hcl; lia).
  assert (Hblock :
    run [ IAddi (S b) (Z.of_nat x)
        ; IExch (S (S b)) (S b)
        ; aop_instr o (S (S b)) b
        ; IExch (S (S b)) (S b)
        ; ISubi (S b) (Z.of_nat x) ]
        (mkState (rupd b (eval σ e) R) M)
    = mkState (rupd b (eval σ e) R)
              (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M)).
  { destruct o; simp_state; rewrite ?H1, ?H2, ?Z.add_0_l, ?Hmod;
    f_equal;
    try (apply functional_extensionality; intro r; unfold rupd;
         destruct (Nat.eqb_spec r (S b)) as [E1|E1];
         destruct (Nat.eqb_spec r (S (S b))) as [E2|E2];
         destruct (Nat.eqb_spec r b) as [E3|E3];
         subst; cbn; try lia;
         try (now rewrite H1); try (now rewrite H2); try reflexivity);
    try (rewrite mupd_shadow; reflexivity). }
  unfold gen_assign_at.
  rewrite run_app, (gen_expr_spec e b (mkState R M) σ Hmod Hcl); cbn [regs mem].
  rewrite run_app, Hblock.
  assert (Hmod' : models (mkState R (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M))
                         (update σ x (adenote o (σ x) (eval σ e))))
    by (apply (models_update (mkState R M) σ x (adenote o (σ x) (eval σ e))); exact Hmod).
  assert (Hcl' : clean_above b
                   (mkState R (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M)))
    by exact Hcl.
  assert (Hev : eval (update σ x (adenote o (σ x) (eval σ e))) e = eval σ e)
    by (now apply eval_update_notin).
  assert (Hunc : mkState (rupd b (eval σ e) R)
                         (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M)
                 = run (gen_expr e b)
                       (mkState R (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M))).
  { rewrite (gen_expr_spec e b _ _ Hmod' Hcl'); cbn [regs mem]. now rewrite Hev. }
  rewrite Hunc.
  apply run_invert_code; apply wf_gen_expr.
Qed.

Lemma gen_swap_at_spec : forall b x y s σ,
  x <> y ->
  models s σ -> clean_above b s ->
  run (gen_swap_at b x y) s
  = mkState (regs s)
            (mupd (Z.of_nat y) (σ x) (mupd (Z.of_nat x) (σ y) (mem s))).
Proof.
  intros b x y [R M] σ Hxy Hmod Hcl; cbn [regs mem] in *.
  assert (Hax : Z.of_nat x <> Z.of_nat y) by (intro Hc; apply Hxy, of_nat_inj, Hc).
  assert (Hay : Z.of_nat y <> Z.of_nat x) by (now apply Z.neq_sym).
  assert (H0 : R b = 0) by (apply Hcl; lia).
  assert (H1 : R (S b) = 0) by (apply Hcl; lia).
  assert (H2 : R (S (S b)) = 0) by (apply Hcl; lia).
  assert (H3 : R (S (S (S b))) = 0) by (apply Hcl; lia).
  unfold gen_swap_at; simp_state.
  rewrite H0, H1, H2, H3.
  rewrite !Z.add_0_l, !Hmod.
  f_equal.
  - apply functional_extensionality; intro r; unfold rupd.
    destruct (Nat.eqb_spec r (S b)) as [E1|E1];
    destruct (Nat.eqb_spec r b) as [E2|E2];
    destruct (Nat.eqb_spec r (S (S b))) as [E3|E3];
    destruct (Nat.eqb_spec r (S (S (S b)))) as [E4|E4];
    subst; cbn; try lia;
    try (rewrite ?H0, ?H1, ?H2, ?H3; lia); try reflexivity.
  - apply functional_extensionality; intro a; unfold mupd.
    destruct (Z.eqb_spec a (Z.of_nat y)) as [F1|F1];
    destruct (Z.eqb_spec a (Z.of_nat x)) as [F2|F2];
    subst; cbn; try congruence; try reflexivity.
Qed.

Theorem compile_at_spec : forall b st σ σ' ms,
  exec st σ σ' -> wf_stmt st ->
  models ms σ -> clean_above b ms ->
  models (run (compile_at b st) ms) σ' /\ regs (run (compile_at b st) ms) = regs ms.
Proof.
  intros b st σ σ' ms H; revert ms; induction H; intros ms Hwf Hmod Hcl.
  - split; [exact Hmod | reflexivity].
  - cbn [compile_at].
    rewrite (gen_assign_at_spec b x o e ms s H Hmod Hcl); cbn [regs mem]; split.
    + apply (models_update ms s x). exact Hmod.
    + reflexivity.
  - cbn [wf_stmt] in Hwf; cbn [compile_at].
    rewrite (gen_swap_at_spec b x y ms s Hwf Hmod Hcl); cbn [regs mem]; split;
      [| reflexivity].
    intro z; cbn [mem]; unfold sw, update.
    destruct (Nat.eq_dec y z) as [->|Hyz].
    + rewrite mupd_same, Nat.eqb_refl. reflexivity.
    + rewrite mupd_other by (intro Hc; apply Hyz, of_nat_inj; congruence).
      destruct (Nat.eq_dec x z) as [->|Hxz].
      * rewrite mupd_same.
        rewrite (proj2 (Nat.eqb_neq y z) Hyz), Nat.eqb_refl. reflexivity.
      * rewrite mupd_other by (intro Hc; apply Hxz, of_nat_inj; congruence).
        rewrite (proj2 (Nat.eqb_neq y z) Hyz), (proj2 (Nat.eqb_neq x z) Hxz).
        apply Hmod.
  - cbn [wf_stmt] in Hwf; destruct Hwf as [Hwf1 Hwf2]; cbn [compile_at].
    rewrite run_app.
    destruct (IHexec1 ms Hwf1 Hmod Hcl) as [Hm1 Hr1].
    assert (Hcl1 : clean_above b (run (compile_at b s1) ms))
      by (intros r Hr; rewrite Hr1; now apply Hcl).
    destruct (IHexec2 (run (compile_at b s1) ms) Hwf2 Hm1 Hcl1) as [Hm2 Hr2].
    split; [exact Hm2 | now rewrite Hr2, Hr1].
Qed.

(** ** The test / assertion block: [rt := rt xor e]

    Evaluate [e] into [S rt], XOR it into [rt], unevaluate.  Memory and every
    register other than [rt] are untouched; [rt] may hold anything. *)

Definition xor_block (e : expr) (rt : reg) : code :=
  gen_expr e (S rt) ++ IXor rt (S rt) :: invert_code (gen_expr e (S rt)).

Lemma xor_block_spec : forall e rt s σ,
  models s σ -> clean_above (S rt) s ->
  run (xor_block e rt) s
  = mkState (rupd rt (Z.lxor (regs s rt) (eval σ e)) (regs s)) (mem s).
Proof.
  intros e rt [R M] σ Hmod Hcl; unfold xor_block; cbn [regs mem].
  rewrite run_app, (gen_expr_spec e (S rt) (mkState R M) σ Hmod Hcl); cbn [regs mem].
  rewrite run_cons; cbn [step regs mem].
  rewrite rupd_same, rupd_other by lia.
  assert (HX : mkState (rupd rt (Z.lxor (R rt) (eval σ e)) (rupd (S rt) (eval σ e) R)) M
             = run (gen_expr e (S rt)) (mkState (rupd rt (Z.lxor (R rt) (eval σ e)) R) M)).
  { rewrite (gen_expr_spec e (S rt) _ σ).
    - cbn [regs mem]. now rewrite (rupd_comm rt (S rt)) by lia.
    - exact Hmod.
    - intros r Hr; cbn [regs]; rewrite rupd_other by lia; apply Hcl; lia. }
  rewrite HX, run_invert_code by apply wf_gen_expr. reflexivity.
Qed.

Lemma xor_block_nonempty : forall e rt, xor_block e rt <> [].
Proof. intros e rt; unfold xor_block; destruct (gen_expr e (S rt)); discriminate. Qed.

(** ** Source language with [If] *)

Inductive cstmt :=
| CBase (s : stmt)
| CSeq  (a b : cstmt)
| CIf   (e1 : expr) (a b : cstmt) (e2 : expr).   (** [if e1 then a else b fi e2] *)

Fixpoint wf_cstmt (st : cstmt) : Prop :=
  match st with
  | CBase s     => wf_stmt s
  | CSeq a b    => wf_cstmt a /\ wf_cstmt b
  | CIf _ a b _ => wf_cstmt a /\ wf_cstmt b
  end.

(** Janus's rules, with the tests required to be Boolean-valued (see the
    header).  Replacing [= 1] by [<> 0] gives exactly [Janus.exec]'s rules. *)
Inductive exec_c : cstmt -> store -> store -> Prop :=
| EC_Base : forall s σ σ', exec s σ σ' -> exec_c (CBase s) σ σ'
| EC_Seq : forall a b σ m σ',
    exec_c a σ m -> exec_c b m σ' -> exec_c (CSeq a b) σ σ'
| EC_IfTrue : forall e1 a b e2 σ σ',
    eval σ e1 = 1 -> exec_c a σ σ' -> eval σ' e2 = 1 ->
    exec_c (CIf e1 a b e2) σ σ'
| EC_IfFalse : forall e1 a b e2 σ σ',
    eval σ e1 = 0 -> exec_c b σ σ' -> eval σ' e2 = 0 ->
    exec_c (CIf e1 a b e2) σ σ'.

Fixpoint invert_c (st : cstmt) : cstmt :=
  match st with
  | CBase s       => CBase (invert s)
  | CSeq a b      => CSeq (invert_c b) (invert_c a)
  | CIf e1 a b e2 => CIf e2 (invert_c a) (invert_c b) e1
  end.

(** The fragment stays reversible: the exit assertion of [If] is the entry
    test of its inverse. *)
Theorem exec_c_rev : forall st σ σ', exec_c st σ σ' -> exec_c (invert_c st) σ' σ.
Proof.
  intros st σ σ' H; induction H; simpl.
  - constructor; now apply exec_rev.
  - econstructor; eassumption.
  - apply EC_IfTrue; assumption.
  - apply EC_IfFalse; assumption.
Qed.

(** ** The compiler

    [if_code b n e1 e2 pa pb] is the layout of the header with flag register
    [b], the five labels [n .. n+4] (in `_gen_if`'s allocation order:
    `if_false`, `if_test`, `if_assert`, `if_assert_true`, `if_end`) and the
    already-compiled bodies [pa], [pb]. *)

Definition if_code (b : reg) (n : label) (e1 e2 : expr) (pa pb : lprog) : lprog :=
  ops (xor_block e1 b)
  ++ (Some (S n), CBeq b 0%nat n)
  :: (None, COp (IXori b 1))
  :: pa
  ++ (None, COp (IXori b 1))
  :: ops_l (n + 2)%nat (xor_block e2 b)
  ++ (Some (n + 3)%nat, CBra (n + 4)%nat)
  :: (Some n, CBra (S n))
  :: pb
  ++ (None, CBra (n + 2)%nat)
  :: (Some (n + 4)%nat, CBra (n + 3)%nat)
  :: [].

(** [compile_c st b n]: code for [st] with scratch base [b] and labels from
    [n]; returns the next free label. *)
Fixpoint compile_c (st : cstmt) (b : reg) (n : label) : lprog * label :=
  match st with
  | CBase s => (ops (compile_at b s), n)
  | CSeq a c =>
      let '(p1, n1) := compile_c a b n in
      let '(p2, n2) := compile_c c b n1 in
      (p1 ++ p2, n2)
  | CIf e1 a c e2 =>
      let '(pa, na) := compile_c a (S b) (n + 5)%nat in
      let '(pc, nc) := compile_c c (S b) na in
      (if_code b n e1 e2 pa pc, nc)
  end.

(** ** Label discipline: every label of the code lies in [[n, n')] *)

(** Decompose [H : In l (labels p)] for a concrete layout [p] into a case per
    label and per embedded fragment. *)
Ltac in_labels H :=
  repeat first
    [ rewrite labels_app in H
    | rewrite labels_ops in H
    | progress cbn [labels] in H ];
  repeat match goal with
  | H' : In _ (_ ++ _) |- _ => rewrite in_app_iff in H'
  | H' : In _ (_ :: _) |- _ => apply in_inv in H'
  | H' : In _ [] |- _ => destruct H'
  | H' : _ \/ _ |- _ => destruct H'
  | H' : In _ (labels (ops_l _ _)) |- _ => apply labels_ops_l in H'
  end.

Lemma labels_if_code : forall b n e1 e2 pa pb l,
  In l (labels (if_code b n e1 e2 pa pb)) ->
  l = S n \/ In l (labels pa) \/ l = (n + 2)%nat \/ l = (n + 3)%nat
  \/ l = n \/ In l (labels pb) \/ l = (n + 4)%nat.
Proof.
  intros b n e1 e2 pa pb l H; unfold if_code in H.
  in_labels H; subst; intuition auto.
Qed.

Lemma compile_c_labels : forall st b n p n',
  compile_c st b n = (p, n') ->
  (n <= n')%nat /\ (forall l, In l (labels p) -> (n <= l < n')%nat).
Proof.
  induction st as [s | a IHa c IHc | e1 a IHa c IHc e2]; intros b n p n' Hc; simpl in Hc.
  - injection Hc as <- <-. split; [lia |].
    intros l H; rewrite labels_ops in H; destruct H.
  - destruct (compile_c a b n) as [p1 n1] eqn:E1.
    destruct (compile_c c b n1) as [p2 n2] eqn:E2.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ E1) as [Hle1 Hin1].
    destruct (IHc _ _ _ _ E2) as [Hle2 Hin2].
    split; [lia |].
    intros l H; rewrite labels_app, in_app_iff in H.
    destruct H as [H | H]; [apply Hin1 in H | apply Hin2 in H]; lia.
  - destruct (compile_c a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_c c (S b) na) as [pb nb] eqn:Eb.
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
Qed.

(** ** Running the [If] layout

    Everything about positions is relative to a prefix [pre] and a suffix
    [post] so that the lemma composes: the bodies are themselves such
    fragments, and the whole thing is one for the enclosing statement. *)

Ltac norm_app := repeat first [ rewrite <- app_assoc | progress cbn [app] ].

Section IfLayout.

Variable pre post : lprog.
Variable b : reg.
Variable n : label.
Variable e1 e2 : expr.
Variable pa pb : lprog.

Hypothesis Hb : b <> 0%nat.
Hypothesis Hpre  : forall l, In l (labels pre)  -> ~ (n <= l < n + 5)%nat.
Hypothesis Hpost : forall l, In l (labels post) -> ~ (n <= l < n + 5)%nat.
Hypothesis Hpa   : forall l, In l (labels pa)   -> ~ (n <= l < n + 5)%nat.
Hypothesis Hpb   : forall l, In l (labels pb)   -> ~ (n <= l < n + 5)%nat.

Let lfalse  := n.
Let ltest   := S n.
Let lassert := (n + 2)%nat.
Let ltrue   := (n + 3)%nat.
Let lend    := (n + 4)%nat.
Let T1 := xor_block e1 b.
Let T2 := xor_block e2 b.

Let P := pre ++ if_code b n e1 e2 pa pb ++ post.

(** Positions of the interesting lines. *)
Let n0 := length pre.
Let t1 := length T1.
Let la := length pa.
Let t2 := length T2.
Let lb := length pb.
Let pB := (n0 + t1)%nat.            (* test:   BEQ rt r0 false *)
Let pA := S (S pB).                 (* then body *)
Let pX := (pA + la)%nat.            (* XORI rt 1 after the then body *)
Let pD := S pX.                     (* assert: the assertion block *)
Let pE := (pD + t2)%nat.            (* true:   BRA end *)
Let pF := S pE.                     (* false:  BRA test *)
Let pC := S pF.                     (* else body *)
Let pG := (pC + lb)%nat.            (* BRA assert *)
Let pH := S pG.                     (* end:    BRA true *)

(** The bodies as embedded fragments, in the shape the induction provides. *)
Let preA := pre ++ ops T1
            ++ [(Some ltest, CBeq b 0%nat lfalse); (None, COp (IXori b 1))].
Let postA := (None, COp (IXori b 1))
             :: ops_l lassert T2
             ++ (Some ltrue, CBra lend)
             :: (Some lfalse, CBra ltest)
             :: pb
             ++ (None, CBra lassert)
             :: (Some lend, CBra ltrue)
             :: post.
Let preB := pre ++ ops T1
            ++ (Some ltest, CBeq b 0%nat lfalse)
            :: (None, COp (IXori b 1))
            :: pa
            ++ (None, COp (IXori b 1))
            :: ops_l lassert T2
            ++ [(Some ltrue, CBra lend); (Some lfalse, CBra ltest)].
Let postB := (None, CBra lassert) :: (Some lend, CBra ltrue) :: post.

Ltac len_solve :=
  unfold pH, pG, pC, pF, pE, pD, pX, pA, pB, lb, t2, la, t1, n0, T1, T2 in *;
  repeat first [ rewrite length_app | rewrite length_ops | rewrite length_ops_l
               | progress cbn [length] ];
  lia.

Ltac notin_labels :=
  let H := fresh in intro H; in_labels H;
  repeat match goal with
  | H : In _ (labels pre) |- _ => apply Hpre in H
  | H : In _ (labels post) |- _ => apply Hpost in H
  | H : In _ (labels pa) |- _ => apply Hpa in H
  | H : In _ (labels pb) |- _ => apply Hpb in H
  end;
  unfold lfalse, ltest, lassert, ltrue, lend in *; lia.

Lemma len_if_code : length (if_code b n e1 e2 pa pb) = (t1 + la + t2 + lb + 7)%nat.
Proof. unfold if_code; len_solve. Qed.

Lemma preA_eq : preA ++ pa ++ postA = P.
Proof. unfold P, preA, postA, if_code; norm_app; reflexivity. Qed.

Lemma preB_eq : preB ++ pb ++ postB = P.
Proof. unfold P, preB, postB, if_code; norm_app; reflexivity. Qed.

Lemma len_preA : length preA = pA.
Proof. unfold preA; len_solve. Qed.

Lemma len_preB : length preB = pC.
Proof. unfold preB; len_solve. Qed.

(** *** The lines *)

Lemma nth_B : nth_error P pB = Some (Some ltest, CBeq b 0%nat lfalse).
Proof.
  replace pB with (length (pre ++ ops T1)) by len_solve.
  eapply nth_error_at. unfold P, if_code; norm_app; reflexivity.
Qed.

Lemma nth_X1 : nth_error P (S pB) = Some (None, COp (IXori b 1)).
Proof.
  replace (S pB) with (length (pre ++ ops T1 ++ [(Some ltest, CBeq b 0%nat lfalse)]))
    by len_solve.
  eapply nth_error_at. unfold P, if_code; norm_app; reflexivity.
Qed.

Lemma nth_X2 : nth_error P pX = Some (None, COp (IXori b 1)).
Proof.
  replace pX with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa))
    by len_solve.
  eapply nth_error_at. unfold P, if_code; norm_app; reflexivity.
Qed.

Lemma nth_D : forall i t, T2 = i :: t -> nth_error P pD = Some (Some lassert, COp i).
Proof.
  intros i t HT2.
  replace pD with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))]))
    by len_solve.
  eapply nth_error_at. unfold P, if_code. unfold T2 in HT2. rewrite HT2.
  cbn [ops_l]. norm_app; reflexivity.
Qed.

Lemma nth_E : nth_error P pE = Some (Some ltrue, CBra lend).
Proof.
  replace pE with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2))
    by len_solve.
  eapply nth_error_at. unfold P, if_code; norm_app; reflexivity.
Qed.

Lemma nth_F : nth_error P pF = Some (Some lfalse, CBra ltest).
Proof.
  replace pF with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2 ++ [(Some ltrue, CBra lend)]))
    by len_solve.
  eapply nth_error_at. unfold P, if_code; norm_app; reflexivity.
Qed.

Lemma nth_G : nth_error P pG = Some (None, CBra lassert).
Proof.
  replace pG with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2
      ++ (Some ltrue, CBra lend) :: (Some lfalse, CBra ltest) :: pb))
    by len_solve.
  eapply nth_error_at. unfold P, if_code; norm_app; reflexivity.
Qed.

Lemma nth_H : nth_error P pH = Some (Some lend, CBra ltrue).
Proof.
  replace pH with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2
      ++ (Some ltrue, CBra lend) :: (Some lfalse, CBra ltest) :: pb
      ++ [(None, CBra lassert)]))
    by len_solve.
  eapply nth_error_at. unfold P, if_code; norm_app; reflexivity.
Qed.

(** *** The labels *)

Lemma find_ltest : find_label ltest P = Some pB.
Proof.
  replace pB with (length (pre ++ ops T1)) by len_solve.
  eapply find_label_at; [unfold P, if_code; norm_app; reflexivity | notin_labels].
Qed.

Lemma find_lassert : find_label lassert P = Some pD.
Proof.
  replace pD with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))]))
    by len_solve.
  eapply find_label_ops_l_at with (c := T2);
    [unfold P, if_code; norm_app; reflexivity | apply xor_block_nonempty | notin_labels].
Qed.

Lemma find_ltrue : find_label ltrue P = Some pE.
Proof.
  replace pE with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2))
    by len_solve.
  eapply find_label_at; [unfold P, if_code; norm_app; reflexivity | notin_labels].
Qed.

Lemma find_lfalse : find_label lfalse P = Some pF.
Proof.
  replace pF with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2 ++ [(Some ltrue, CBra lend)]))
    by len_solve.
  eapply find_label_at; [unfold P, if_code; norm_app; reflexivity | notin_labels].
Qed.

Lemma find_lend : find_label lend P = Some pH.
Proof.
  replace pH with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2
      ++ (Some ltrue, CBra lend) :: (Some lfalse, CBra ltest) :: pb
      ++ [(None, CBra lassert)]))
    by len_solve.
  eapply find_label_at; [unfold P, if_code; norm_app; reflexivity | notin_labels].
Qed.

(** *** Which branches are paired *)

Lemma paired_E : paired P pE = true.
Proof. eapply paired_bra_bra; [apply nth_E | apply find_lend | apply nth_H | apply find_ltrue]. Qed.

Lemma paired_H : paired P pH = true.
Proof. eapply paired_bra_bra; [apply nth_H | apply find_ltrue | apply nth_E | apply find_lend]. Qed.

Lemma paired_F : paired P pF = true.
Proof. eapply paired_bra_beq; [apply nth_F | apply find_ltest | apply nth_B | apply find_lfalse]. Qed.

Lemma paired_G : paired P pG = false.
Proof.
  assert (HT2 : exists i t, T2 = i :: t).
  { unfold T2; destruct (xor_block e2 b) as [| i t] eqn:E;
      [exfalso; eapply xor_block_nonempty; eassumption | eauto]. }
  destruct HT2 as [i [t HT2]].
  eapply paired_bra_op; [apply nth_G | apply find_lassert | apply (nth_D i t HT2)].
Qed.

(** *** The join: `true: BRA end` lands on `end: BRA true`, which cancels
    [br] and falls through to the line after the [If]. *)

Lemma join_steps : forall s, steps P (mkC pE 0 s) (mkC (S pH) 0 s).
Proof.
  intro s.
  eapply steps_step.
  { eapply cstep_bra_paired_jump; [apply nth_E | apply find_lend | apply paired_E | | ].
    - unfold pH, pG, pC, pF; lia.
    - lia. }
  rewrite Z.add_0_l, Nat2Z.id.
  apply steps_one.
  eapply cstep_bra_paired_cancel; [apply nth_H | apply find_ltrue | apply paired_H | lia].
Qed.

(** *** The assertion block, from the state the two paths reach it in *)

Lemma assert_steps : forall R M σ' f,
  models (mkState R M) σ' -> clean_above b (mkState R M) ->
  Z.lxor f (eval σ' e2) = 0 ->
  steps P (mkC pD 0 (mkState (rupd b f R) M)) (mkC pE 0 (mkState R M)).
Proof.
  intros R M σ' f Hmod Hcl Hf.
  assert (HT2 : run T2 (mkState (rupd b f R) M) = mkState R M).
  { unfold T2. rewrite (xor_block_spec e2 b _ σ').
    - cbn [regs mem]. rewrite rupd_same, Hf, rupd_shadow.
      rewrite rupd_zero; [reflexivity | apply Hcl; lia].
    - exact Hmod.
    - intros r Hr; cbn [regs]; rewrite rupd_other by lia; apply Hcl; lia. }
  rewrite <- HT2.
  replace pD with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))])) by len_solve.
  replace pE with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))]) + length T2)%nat by len_solve.
  eapply steps_ops_l. unfold P, if_code; norm_app; reflexivity.
Qed.

(** *** The entry test *)

Lemma test_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) ->
  steps P (mkC n0 0 (mkState R M)) (mkC pB 0 (mkState (rupd b (eval σ e1) R) M)).
Proof.
  intros R M σ Hmod Hcl.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT1 : run T1 (mkState R M) = mkState (rupd b (eval σ e1) R) M).
  { unfold T1. rewrite (xor_block_spec e1 b _ σ).
    - cbn [regs mem]. rewrite Hb0. now rewrite Z.lxor_0_l.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia. }
  rewrite <- HT1.
  replace pB with (length pre + length T1)%nat by len_solve.
  eapply steps_ops. unfold P, if_code; norm_app; reflexivity.
Qed.

(** *** The then path *)

Lemma if_true_run : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 1 -> eval σ' e2 = 1 ->
  steps (preA ++ pa ++ postA) (mkC (length preA) 0 (mkState R M))
        (mkC (length preA + length pa) 0 (mkState R' M')) ->
  models (mkState R' M') σ' -> R' = R ->
  steps P (mkC (length pre) 0 (mkState R M))
          (mkC (length pre + length (if_code b n e1 e2 pa pb)) 0 (mkState R M')).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 He2 Hbody Hmod' HR; subst R'.
  rewrite preA_eq, len_preA in Hbody.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  (* test: rt := 1 *)
  eapply steps_trans; [apply (test_steps R M σ Hmod Hcl) |].
  rewrite He1.
  (* BEQ rt r0 false: not taken *)
  eapply steps_step.
  { eapply cstep_beq_direct_not_taken; [apply nth_B |].
    cbn [regs]. rewrite rupd_same, rupd_other by auto. rewrite H0. discriminate. }
  (* XORI rt 1: rt := 0, i.e. the state is [R, M] again *)
  eapply steps_step; [eapply cstep_op; apply nth_X1 |].
  cbn [step regs mem]. rewrite rupd_same, rupd_shadow.
  change (Z.lxor 1 1) with 0. rewrite rupd_zero by exact Hb0.
  (* the then body *)
  eapply steps_trans; [exact Hbody |].
  (* XORI rt 1: rt := 1 *)
  eapply steps_step; [eapply cstep_op; apply nth_X2 |].
  cbn [step regs mem]. rewrite Hb0. change (Z.lxor 0 1) with 1.
  (* assertion: rt := 1 xor e2 = 0 *)
  eapply steps_trans; [apply (assert_steps R M' σ' 1 Hmod' Hcl); now rewrite He2 |].
  (* the join *)
  replace (length pre + length (if_code b n e1 e2 pa pb))%nat with (S pH)
    by (rewrite len_if_code; len_solve).
  apply join_steps.
Qed.

(** *** The else path *)

Lemma if_false_run : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 -> eval σ' e2 = 0 ->
  steps (preB ++ pb ++ postB) (mkC (length preB) 0 (mkState R M))
        (mkC (length preB + length pb) 0 (mkState R' M')) ->
  models (mkState R' M') σ' -> R' = R ->
  steps P (mkC (length pre) 0 (mkState R M))
          (mkC (length pre + length (if_code b n e1 e2 pa pb)) 0 (mkState R M')).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 He2 Hbody Hmod' HR; subst R'.
  rewrite preB_eq, len_preB in Hbody.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  (* test: rt := 0, i.e. the state is unchanged *)
  eapply steps_trans; [apply (test_steps R M σ Hmod Hcl) |].
  rewrite He1, rupd_zero by exact Hb0.
  (* BEQ rt r0 false, taken with br = 0: a direct jump to `false` *)
  eapply steps_step.
  { eapply cstep_beq_direct_taken; [apply nth_B | apply find_lfalse |].
    cbn [regs]. congruence. }
  (* false: BRA test — paired with the BEQ, br := pB - pF, land on the BEQ *)
  eapply steps_step.
  { eapply cstep_bra_paired_jump; [apply nth_F | apply find_ltest | apply paired_F | |].
    - unfold pF, pE, pD; lia.
    - lia. }
  rewrite Z.add_0_l, Nat2Z.id.
  (* the BEQ again, now with br <> 0: cancel and continue after `false` *)
  eapply steps_step.
  { eapply cstep_beq_cancel; [apply nth_B | apply find_lfalse | | |].
    - cbn [regs]. congruence.
    - unfold pF, pE, pD; lia.
    - lia. }
  (* the else body *)
  eapply steps_trans; [exact Hbody |].
  (* BRA assert: an ordinary jump *)
  eapply steps_step.
  { eapply cstep_bra_direct; [apply nth_G | apply find_lassert | apply paired_G]. }
  (* assertion: rt := 0 xor e2 = 0 *)
  assert (HA := assert_steps R M' σ' 0 Hmod' Hcl ltac:(now rewrite He2)).
  rewrite rupd_zero in HA by exact Hb0.
  eapply steps_trans; [exact HA |].
  (* the join *)
  replace (length pre + length (if_code b n e1 e2 pa pb))%nat with (S pH)
    by (rewrite len_if_code; len_solve).
  apply join_steps.
Qed.

End IfLayout.

(** ** Main theorem

    A compiled statement, embedded between any [pre] and [post] that do not
    define its labels, started with [br = 0] at its first line, reaches the
    line after its last with [br = 0], a memory representing the final store
    and the *same* register file.  [b <> 0] and [regs ms 0 = 0] stand for
    the hard-wired zero register `r0` the entry test compares against. *)

Theorem compile_c_spec : forall st σ σ' b n p n' ms pre post,
  exec_c st σ σ' -> wf_cstmt st ->
  compile_c st b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  exists ms',
    steps (pre ++ p ++ post)
          (mkC (length pre) 0 ms) (mkC (length pre + length p) 0 ms')
    /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros st σ σ' b n p n' ms pre post Hex.
  revert b n p n' ms pre post.
  induction Hex as [s σ σ' Hs | a c σ m σ' Ha IHa Hc IHc
                   | e1 a c e2 σ σ' He1 Ha IHa He2 | e1 a c e2 σ σ' He1 Hc IHc He2];
    intros b n p n' ms pre post Hwf Hcomp Hb Hmod Hcl H0 Hpre Hpost; simpl in Hcomp.
  - (* CBase *)
    injection Hcomp as <- <-.
    destruct (compile_at_spec b s σ σ' ms Hs Hwf Hmod Hcl) as [Hm Hr].
    exists (run (compile_at b s) ms). split; [| split; assumption].
    rewrite length_ops. eapply steps_ops. reflexivity.
  - (* CSeq *)
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_c a b n) as [p1 n1] eqn:E1.
    destruct (compile_c c b n1) as [p2 n2] eqn:E2.
    injection Hcomp as <- <-.
    destruct (compile_c_labels _ _ _ _ _ E1) as [Hle1 Hin1].
    destruct (compile_c_labels _ _ _ _ _ E2) as [Hle2 Hin2].
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
  - (* CIf, then path *)
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_c a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_c c (S b) na) as [pb nb] eqn:Eb.
    injection Hcomp as <- <-.
    destruct (compile_c_labels _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_c_labels _ _ _ _ _ Eb) as [Hleb Hinb].
    destruct ms as [R M].
    assert (HclS : clean_above (S b) (mkState R M)) by (intros r Hr; apply Hcl; lia).
    destruct (IHa (S b) (n + 5)%nat pa na (mkState R M)
                  (pre ++ ops (xor_block e1 b)
                       ++ [(Some (S n), CBeq b 0%nat n); (None, COp (IXori b 1))])
                  ((None, COp (IXori b 1))
                   :: ops_l (n + 2)%nat (xor_block e2 b)
                   ++ (Some (n + 3)%nat, CBra (n + 4)%nat)
                   :: (Some n, CBra (S n))
                   :: pb
                   ++ (None, CBra (n + 2)%nat)
                   :: (Some (n + 4)%nat, CBra (n + 3)%nat)
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
    apply (if_true_run pre post b n e1 e2 pa pb Hb) with (σ := σ) (σ' := σ') (R' := R');
      try assumption.
    all: intros l Hl;
         first [apply Hpre in Hl | apply Hpost in Hl | apply Hina in Hl | apply Hinb in Hl];
         lia.
  - (* CIf, else path *)
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_c a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_c c (S b) na) as [pb nb] eqn:Eb.
    injection Hcomp as <- <-.
    destruct (compile_c_labels _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_c_labels _ _ _ _ _ Eb) as [Hleb Hinb].
    destruct ms as [R M].
    assert (HclS : clean_above (S b) (mkState R M)) by (intros r Hr; apply Hcl; lia).
    destruct (IHc (S b) na pb nb (mkState R M)
                  (pre ++ ops (xor_block e1 b)
                       ++ (Some (S n), CBeq b 0%nat n)
                       :: (None, COp (IXori b 1))
                       :: pa
                       ++ (None, COp (IXori b 1))
                       :: ops_l (n + 2)%nat (xor_block e2 b)
                       ++ [(Some (n + 3)%nat, CBra (n + 4)%nat); (Some n, CBra (S n))])
                  ((None, CBra (n + 2)%nat) :: (Some (n + 4)%nat, CBra (n + 3)%nat) :: post)
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
    apply (if_false_run pre post b n e1 e2 pa pb Hb) with (σ := σ) (σ' := σ') (R' := R');
      try assumption.
    all: intros l Hl;
         first [apply Hpre in Hl | apply Hpost in Hl | apply Hina in Hl | apply Hinb in Hl];
         lia.
Qed.

(** ** Whole-program corollary, on the executable machine

    Compile with the real scratch base and labels from 0, start at pc 0 with
    [br = 0]; the fuel interpreter halts by falling off the end. *)

Corollary compile_c_program : forall st σ σ' p n' ms,
  exec_c st σ σ' -> wf_cstmt st ->
  compile_c st scratch 0%nat = (p, n') ->
  models ms σ -> clean_above scratch ms -> regs ms 0%nat = 0 ->
  exists ms' fuel,
    exec_fuel fuel p (mkC 0%nat 0 ms) = Some (mkC (length p) 0 ms')
    /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros st σ σ' p n' ms Hex Hwf Hc Hmod Hcl H0.
  destruct (compile_c_spec st σ σ' scratch 0%nat p n' ms [] [] Hex Hwf Hc)
    as [ms' [Hst [Hm Hr]]];
    [unfold scratch; lia | exact Hmod | exact Hcl | exact H0
    | intros l Hl; destruct Hl | intros l Hl; destruct Hl |].
  exists ms'.
  rewrite app_nil_r in Hst; cbn [app length] in Hst.
  destruct (steps_exec_fuel p _ _ Hst) as [fuel Hf].
  { cbn [cpc]. apply nth_error_None. lia. }
  exists fuel. auto.
Qed.

(** ** Sanity checks: run the compiler and the machine

    Straight-line bodies at base 4 (the flag is r3), both paths. *)

(** [x += 3; if x ^ 2 then y += 1 else y += 2 fi y ^ 0]: with x = 3 the test
    is 1, the then branch runs, and the assertion [y ^ 0 = 1] holds. *)
Definition prog_then : cstmt :=
  CSeq (CBase (Assign 0%nat AAdd (Cst 3)))
       (CIf (Bin OXor (Var 0%nat) (Cst 2))
            (CBase (Assign 1%nat AAdd (Cst 1)))
            (CBase (Assign 1%nat AAdd (Cst 2)))
            (Bin OXor (Var 1%nat) (Cst 0))).

(** [x += 3; if x ^ 3 then y += 1 else y += 2 fi y ^ 2]: the test is 0, the
    else branch runs, and the assertion [y ^ 2 = 0] holds. *)
Definition prog_else : cstmt :=
  CSeq (CBase (Assign 0%nat AAdd (Cst 3)))
       (CIf (Bin OXor (Var 0%nat) (Cst 3))
            (CBase (Assign 1%nat AAdd (Cst 1)))
            (CBase (Assign 1%nat AAdd (Cst 2)))
            (Bin OXor (Var 1%nat) (Cst 2))).

Definition observe (p : lprog) (r : option cstate) :=
  match r with
  | Some c => (Nat.eqb (cpc c) (length p), cbr c, mem (cst c) 0, mem (cst c) 1,
               regs (cst c) 3%nat, regs (cst c) 4%nat, regs (cst c) 5%nat,
               regs (cst c) 6%nat, regs (cst c) 7%nat)
  | None => (false, 1, 0, 0, 0, 0, 0, 0, 0)
  end.

Example ex_then :
  let p := fst (compile_c prog_then scratch 0%nat) in
  observe p (exec_fuel 500 p (mkC 0%nat 0 zero_state)) = (true, 0, 3, 1, 0, 0, 0, 0, 0).
Proof. vm_compute. reflexivity. Qed.

Example ex_else :
  let p := fst (compile_c prog_else scratch 0%nat) in
  observe p (exec_fuel 500 p (mkC 0%nat 0 zero_state)) = (true, 0, 3, 2, 0, 0, 0, 0, 0).
Proof. vm_compute. reflexivity. Qed.

(** A violated exit assertion is *not* clean: [fi y ^ 3] on the then path
    leaves the flag dirty — this is what `pisa_interp.py` reports at FINISH. *)
Example ex_violation_dirty :
  let st := CSeq (CBase (Assign 0%nat AAdd (Cst 3)))
                 (CIf (Bin OXor (Var 0%nat) (Cst 2))
                      (CBase (Assign 1%nat AAdd (Cst 1)))
                      (CBase (Assign 1%nat AAdd (Cst 2)))
                      (Bin OXor (Var 1%nat) (Cst 3))) in
  let p := fst (compile_c st scratch 0%nat) in
  match exec_fuel 500 p (mkC 0%nat 0 zero_state) with
  | Some c => regs (cst c) 3%nat
  | None => 0
  end = 3.
Proof. vm_compute. reflexivity. Qed.

(** A nested [If] with the inner flag in r4 and its bodies at base 5:
    [if x ^ 2 then (if y ^ 1 then y += 5 else skip fi y ^ 4) else skip fi y ^ 4]
    from x = 3, y = 0 takes both then paths and ends with y = 5. *)
Definition prog_nested : cstmt :=
  CSeq (CBase (Assign 0%nat AAdd (Cst 3)))
       (CIf (Bin OXor (Var 0%nat) (Cst 2))
            (CIf (Bin OXor (Var 1%nat) (Cst 1))
                 (CBase (Assign 1%nat AAdd (Cst 5)))
                 (CBase Skip)
                 (Bin OXor (Var 1%nat) (Cst 4)))
            (CBase Skip)
            (Bin OXor (Var 1%nat) (Cst 4))).

Example ex_nested :
  let p := fst (compile_c prog_nested scratch 0%nat) in
  observe p (exec_fuel 500 p (mkC 0%nat 0 zero_state)) = (true, 0, 3, 5, 0, 0, 0, 0, 0).
Proof. vm_compute. reflexivity. Qed.

(** The nested case compiles with distinct labels and the theorem applies to it. *)
Example ex_nested_labels :
  snd (compile_c (CIf (Cst 1) (CIf (Cst 1) (CBase Skip) (CBase Skip) (Cst 1))
                               (CBase Skip) (Cst 1)) scratch 0%nat) = 10%nat.
Proof. reflexivity. Qed.

(** ** Axiom footprint *)

Print Assumptions compile_c_spec.
Print Assumptions compile_c_program.
Print Assumptions exec_c_rev.
Print Assumptions compile_at_scratch.
