(** * CompileIf.v — compiling [If] to paired branches, and its correctness

    This file extends the verified translation of Compile.v with Janus's
    conditional

      if e1 then S1 else S2 fi e2

    and proves it correct on the control-flow machine of PISACtl.v.  The
    emitted code is the layout `_gen_if` in `codegen.py` produces (Axelsen,
    CC 2011, Fig. 9–11, plus the assertion check of PR #6):

<<
            <rt ^= _as_flag(e1)>                           -- rt := (e1 != 0)
   test:    BEQ rt r0 false
            XORI rt 1                                      -- rt := 0 on the then path
            <S1>
            XORI rt 1                                      -- rt := 1 again
   assert:  <rt ^= _as_flag(e2)>                           -- rt := rt xor (e2 != 0)
   true:    BRA end                 \ Pendulum pair
   false:   BRA test                / partner of the BEQ
            <S2>
            BRA assert
   end:     BRA true                / partner of `true`
            BNE rt r0 finish                               -- violated: halt, rt = 1
>>

    [rt ^= _as_flag(e)] is [flag_block]: `_as_flag` leaves comparisons,
    [&&] / [||] and the constants 0 and 1 alone ([is_flag_expr]) and turns
    every other expression of [Src.expr] into [e != 0], computed with two
    [SLTX] against [r0] (see [as_flag], [Compile.nz_code]).

    Three facts about this layout drive the proof:

    - The flag register [rt] is allocated *before* the bodies (it is the
      lowest free register, [scratch]), so the bodies are compiled with
      scratch base [S rt].  Compile.v hard-wires the base to [scratch];
      [compile_at] below is the same compiler with the base as a parameter,
      and [compile_at_scratch] shows it coincides with [compile].
    - Because tests are normalised to 0/1, the flag is exactly the path bit
      and [rt xor (e2 != 0)] is [0] precisely when the exit assertion holds.
      [exec_c] is therefore Janus's own rule set (true = nonzero).
    - A valid run reaches `BNE rt r0 finish` with [rt = 0], so the branch
      falls through and the label [finish] is never looked up; a violated
      assertion reaches it with [rt = 1] and jumps to [finish]
      ([if_true_violation], [if_false_violation]).  [finish] is a label the
      enclosing program provides; the compiler takes it as a parameter.

    Main result: [compile_c_spec] — semantic preservation and a clean
    register file, for a fragment embedded at any position of a larger
    labeled program with fresh labels; [compile_c_program] is the closed
    whole-program corollary with the fuel executor.  The violation
    theorems for whole statements are in CompileLoop.v. *)

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
  ++ ungen_expr e b.

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

Lemma wf_compile_at : forall b st, b <> 0%nat -> wf_code (compile_at b st).
Proof.
  intros b st Hb; induction st as [| x o e | x y | s1 IH1 s2 IH2]; simpl.
  - apply Forall_nil.
  - unfold gen_assign_at, ungen_expr.
    apply wf_code_app; [apply wf_gen_expr, Hb |].
    repeat (apply Forall_cons; [destruct o; simpl; first [exact I | lia] |]).
    apply wf_invert_code, wf_gen_expr, Hb.
  - unfold gen_swap_at; wf_list.
  - apply wf_code_app; auto.
Qed.

(** The proofs below are those of Compile.v with [scratch] generalized. *)

Lemma gen_assign_at_spec : forall b x o e s σ,
  occurs x e = false ->
  models s σ -> clean_above b s -> regs s 0%nat = 0 -> b <> 0%nat ->
  run (gen_assign_at b x o e) s
  = mkState (regs s)
            (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) (mem s)).
Proof.
  intros b x o e [R M] σ Hocc Hmod Hcl Hr0 Hb; cbn [regs mem] in *.
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
  rewrite run_app, (gen_expr_spec e b (mkState R M) σ Hmod Hcl Hr0 Hb); cbn [regs mem].
  rewrite run_app, Hblock.
  assert (Hmod' : models (mkState R (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M))
                         (update σ x (adenote o (σ x) (eval σ e))))
    by (apply (models_update (mkState R M) σ x (adenote o (σ x) (eval σ e))); exact Hmod).
  assert (Hev : eval (update σ x (adenote o (σ x) (eval σ e))) e = eval σ e)
    by (now apply eval_update_notin).
  rewrite <- Hev at 1.
  exact (ungen_expr_spec e b R _ _ Hmod' Hcl Hr0 Hb).
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
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 -> b <> 0%nat ->
  models (run (compile_at b st) ms) σ' /\ regs (run (compile_at b st) ms) = regs ms.
Proof.
  intros b st σ σ' ms H Hwf0 Hmod0 Hcl0 Hr00 Hb; revert ms Hwf0 Hmod0 Hcl0 Hr00.
  induction H; intros ms Hwf Hmod Hcl Hr0.
  - split; [exact Hmod | reflexivity].
  - cbn [compile_at].
    rewrite (gen_assign_at_spec b x o e ms s H Hmod Hcl Hr0 Hb); cbn [regs mem]; split.
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
    destruct (IHexec1 ms Hwf1 Hmod Hcl Hr0) as [Hm1 Hr1].
    assert (Hcl1 : clean_above b (run (compile_at b s1) ms))
      by (intros r Hr; rewrite Hr1; now apply Hcl).
    destruct (IHexec2 (run (compile_at b s1) ms) Hwf2 Hm1 Hcl1 ltac:(now rewrite Hr1))
      as [Hm2 Hr2].
    split; [exact Hm2 | now rewrite Hr2, Hr1].
Qed.

(** ** The test / assertion block: [rt := rt xor e]

    Evaluate [e] into [S rt], XOR it into [rt], unevaluate (`_gen_flag_xor`).
    Memory and every register other than [rt] are untouched; [rt] may hold
    anything. *)

Definition xor_block (e : expr) (rt : reg) : code :=
  gen_expr e (S rt) ++ IXor rt (S rt) :: ungen_expr e (S rt).

Lemma xor_block_spec : forall e rt s σ,
  models s σ -> clean_above (S rt) s -> regs s 0%nat = 0 -> rt <> 0%nat ->
  run (xor_block e rt) s
  = mkState (rupd rt (Z.lxor (regs s rt) (eval σ e)) (regs s)) (mem s).
Proof.
  intros e rt [R M] σ Hmod Hcl H0 Hrt; unfold xor_block; cbn [regs mem] in *.
  rewrite run_app, (gen_expr_spec e (S rt) (mkState R M) σ Hmod Hcl H0) by lia;
    cbn [regs mem].
  rewrite run_cons; cbn [step regs mem].
  rewrite rupd_same, rupd_other by lia.
  rewrite (rupd_comm rt (S rt)) by lia.
  apply (ungen_expr_spec e (S rt) _ M σ).
  - exact Hmod.
  - intros r Hr; cbn [regs]; rewrite rupd_other by lia; apply Hcl; lia.
  - rewrite rupd_other by lia; exact H0.
  - lia.
Qed.

Lemma xor_block_nonempty : forall e rt, xor_block e rt <> [].
Proof. intros e rt; unfold xor_block; destruct (gen_expr e (S rt)); discriminate. Qed.

(** ** Tests are normalised to 0/1: `_as_flag`

    Janus reads a test as true when it is nonzero.  `codegen.py` XORs tests
    and assertions into a 0/1 path flag, so `_as_flag` keeps a test that is
    already 0/1 — a comparison, a logical operator, or the constant 0 or 1
    ([is_flag_expr], Compile.v) — and rewrites every other [e] to [e != 0]
    ([as_flag]).  `_gen_flag_xor` then XORs the value of that expression
    into the flag exactly as [xor_block] does: evaluate, [XOR rt re],
    run the evaluation backwards.  For [e != 0] the evaluation is
    `_gen_nonzero` ([Compile.nz_code]):

<<
     <e → v> ; SLTX re v r0 ; SLTX re r0 v ; <e → v>⁻¹ ; XOR rt re ;
     <e → v> ; SLTX re r0 v ; SLTX re v r0 ; <e → v>⁻¹
>>

    (the two bits [v < 0] and [0 < v] are exclusive, so their XOR is
    [v <> 0]; a literal [k != 0] is folded to the constant by `_gen_binop`,
    [Compile.fold_csts]). *)

Definition truth (v : Z) : Z := if v =? 0 then 0 else 1.

Lemma truth_0 : truth 0 = 0.
Proof. reflexivity. Qed.

Lemma truth_nz : forall v, v <> 0 -> truth v = 1.
Proof. intros v H; unfold truth; destruct (Z.eqb_spec v 0); [contradiction | reflexivity]. Qed.

Lemma truth_b2z : forall v, b2z (negb (v =? 0)) = truth v.
Proof. intro v; unfold truth, b2z; destruct (v =? 0); reflexivity. Qed.

Lemma sltx_pair : forall v,
  Z.lxor (if v <? 0 then 1 else 0) (if 0 <? v then 1 else 0) = truth v.
Proof. intro v; rewrite <- truth_b2z; apply Compile.sltx_pair. Qed.

(** `_as_flag`. *)
Definition as_flag (e : expr) : expr :=
  if is_flag_expr e then e else Bin ONe e (Cst 0).

Lemma eval_as_flag : forall σ e, eval σ (as_flag e) = truth (eval σ e).
Proof.
  intros σ e; unfold as_flag.
  destruct (is_flag_expr e) eqn:Eb.
  - destruct (is_flag_expr_01 σ e Eb) as [E | E]; rewrite E; reflexivity.
  - cbn [eval denote]. apply truth_b2z.
Qed.

(** [rt ^= _as_flag(e)]: the block `_gen_flag_xor(rt, _as_flag(e))` emits. *)
Definition flag_block (e : expr) (rt : reg) : code := xor_block (as_flag e) rt.

Lemma flag_block_spec : forall e rt s σ,
  models s σ -> clean_above (S rt) s -> regs s 0%nat = 0 -> rt <> 0%nat ->
  run (flag_block e rt) s
  = mkState (rupd rt (Z.lxor (regs s rt) (truth (eval σ e))) (regs s)) (mem s).
Proof.
  intros e rt s σ Hmod Hcl H0 Hrt; unfold flag_block.
  rewrite (xor_block_spec (as_flag e) rt s σ Hmod Hcl H0 Hrt).
  now rewrite eval_as_flag.
Qed.

Lemma flag_block_nonempty : forall e rt, flag_block e rt <> [].
Proof. intros e rt; apply xor_block_nonempty. Qed.

(** Every instruction of a test block is well-formed (locally invertible),
    for every test — the old `XOR rt rt` flag clear and the clearing ORX /
    ANDX of comparisons are gone. *)
Lemma wf_xor_block : forall e rt, wf_code (xor_block e rt).
Proof.
  intros e rt; unfold xor_block, ungen_expr.
  apply wf_code_app; [apply wf_gen_expr; lia |].
  apply Forall_cons; [cbn [wf_instr]; lia |].
  apply wf_invert_code, wf_gen_expr; lia.
Qed.

Lemma wf_flag_block : forall e rt, wf_code (flag_block e rt).
Proof. intros; apply wf_xor_block. Qed.

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

(** Janus's rules ([Janus.exec]): a test is true when it is nonzero. *)
Inductive exec_c : cstmt -> store -> store -> Prop :=
| EC_Base : forall s σ σ', exec s σ σ' -> exec_c (CBase s) σ σ'
| EC_Seq : forall a b σ m σ',
    exec_c a σ m -> exec_c b m σ' -> exec_c (CSeq a b) σ σ'
| EC_IfTrue : forall e1 a b e2 σ σ',
    eval σ e1 <> 0 -> exec_c a σ σ' -> eval σ' e2 <> 0 ->
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

    [if_code fin b n e1 e2 pa pb] is the layout of the header with flag
    register [b], the five labels [n .. n+4] (in `_gen_if`'s allocation
    order: `if_false`, `if_test`, `if_assert`, `if_assert_true`, `if_end`)
    and the already-compiled bodies [pa], [pb].  [fin] is the label of the
    program's `finish` (`ASSERT_FAIL_LABEL` in `codegen.py`), where a
    violated assertion jumps.  The compiler only *refers* to [fin]; the
    enclosing program must define it (see [with_finish] below). *)

Definition if_code (fin : label) (b : reg) (n : label) (e1 e2 : expr) (pa pb : lprog)
  : lprog :=
  ops (flag_block e1 b)
  ++ (Some (S n), CBeq b 0%nat n)
  :: (None, COp (IXori b 1))
  :: pa
  ++ (None, COp (IXori b 1))
  :: ops_l (n + 2)%nat (flag_block e2 b)
  ++ (Some (n + 3)%nat, CBra (n + 4)%nat)
  :: (Some n, CBra (S n))
  :: pb
  ++ (None, CBra (n + 2)%nat)
  :: (Some (n + 4)%nat, CBra (n + 3)%nat)
  :: (None, CBne b 0%nat fin)
  :: [].

(** [compile_c fin st b n]: code for [st] with scratch base [b] and labels
    from [n]; returns the next free label. *)
Fixpoint compile_c (fin : label) (st : cstmt) (b : reg) (n : label) : lprog * label :=
  match st with
  | CBase s => (ops (compile_at b s), n)
  | CSeq a c =>
      let '(p1, n1) := compile_c fin a b n in
      let '(p2, n2) := compile_c fin c b n1 in
      (p1 ++ p2, n2)
  | CIf e1 a c e2 =>
      let '(pa, na) := compile_c fin a (S b) (n + 5)%nat in
      let '(pc, nc) := compile_c fin c (S b) na in
      (if_code fin b n e1 e2 pa pc, nc)
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

Lemma labels_if_code : forall fin b n e1 e2 pa pb l,
  In l (labels (if_code fin b n e1 e2 pa pb)) ->
  l = S n \/ In l (labels pa) \/ l = (n + 2)%nat \/ l = (n + 3)%nat
  \/ l = n \/ In l (labels pb) \/ l = (n + 4)%nat.
Proof.
  intros fin b n e1 e2 pa pb l H; unfold if_code in H.
  in_labels H; subst; intuition auto.
Qed.

Lemma compile_c_labels : forall fin st b n p n',
  compile_c fin st b n = (p, n') ->
  (n <= n')%nat /\ (forall l, In l (labels p) -> (n <= l < n')%nat).
Proof.
  intro fin.
  induction st as [s | a IHa c IHc | e1 a IHa c IHc e2]; intros b n p n' Hc; simpl in Hc.
  - injection Hc as <- <-. split; [lia |].
    intros l H; rewrite labels_ops in H; destruct H.
  - destruct (compile_c fin a b n) as [p1 n1] eqn:E1.
    destruct (compile_c fin c b n1) as [p2 n2] eqn:E2.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ E1) as [Hle1 Hin1].
    destruct (IHc _ _ _ _ E2) as [Hle2 Hin2].
    split; [lia |].
    intros l H; rewrite labels_app, in_app_iff in H.
    destruct H as [H | H]; [apply Hin1 in H | apply Hin2 in H]; lia.
  - destruct (compile_c fin a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_c fin c (S b) na) as [pb nb] eqn:Eb.
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
Variable fin : label.

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
Let T1 := flag_block e1 b.
Let T2 := flag_block e2 b.

Let P := pre ++ if_code fin b n e1 e2 pa pb ++ post.

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
Let pN := S pH.                     (* BNE rt r0 finish *)

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
             :: (None, CBne b 0%nat fin)
             :: post.
Let preB := pre ++ ops T1
            ++ (Some ltest, CBeq b 0%nat lfalse)
            :: (None, COp (IXori b 1))
            :: pa
            ++ (None, COp (IXori b 1))
            :: ops_l lassert T2
            ++ [(Some ltrue, CBra lend); (Some lfalse, CBra ltest)].
Let postB := (None, CBra lassert) :: (Some lend, CBra ltrue)
             :: (None, CBne b 0%nat fin) :: post.

Ltac len_solve :=
  unfold pN, pH, pG, pC, pF, pE, pD, pX, pA, pB, lb, t2, la, t1, n0, T1, T2 in *;
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

Lemma len_if_code : length (if_code fin b n e1 e2 pa pb) = (t1 + la + t2 + lb + 8)%nat.
Proof. unfold if_code; len_solve. Qed.

Lemma preA_eq : preA ++ pa ++ postA = P.
Proof. unfold P, preA, postA, if_code; norm_app; reflexivity. Qed.

Lemma preB_eq : preB ++ pb ++ postB = P.
Proof. unfold P, preB, postB, if_code; norm_app; reflexivity. Qed.

Lemma len_preA : length preA = pA.
Proof. unfold preA; len_solve. Qed.

Lemma len_preB : length preB = pC.
Proof. unfold preB; len_solve. Qed.

Lemma end_pos : (length pre + length (if_code fin b n e1 e2 pa pb))%nat = S pN.
Proof. rewrite len_if_code; len_solve. Qed.

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

Lemma nth_N : nth_error P pN = Some (None, CBne b 0%nat fin).
Proof.
  replace pN with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2
      ++ (Some ltrue, CBra lend) :: (Some lfalse, CBra ltest) :: pb
      ++ [(None, CBra lassert); (Some lend, CBra ltrue)]))
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
    [unfold P, if_code; norm_app; reflexivity | apply flag_block_nonempty | notin_labels].
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
  { unfold T2; destruct (flag_block e2 b) as [| i t] eqn:E;
      [exfalso; eapply flag_block_nonempty; eassumption | eauto]. }
  destruct HT2 as [i [t HT2]].
  eapply paired_bra_op; [apply nth_G | apply find_lassert | apply (nth_D i t HT2)].
Qed.

(** *** The join: `true: BRA end` lands on `end: BRA true`, which cancels
    [br] and falls through to the flag check after the [If]. *)

Lemma join_steps : forall s, steps P (mkC pE 0 s) (mkC pN 0 s).
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

(** *** `BNE rt r0 finish`: falls through on a clean flag, jumps to
    [finish] on a dirty one. *)

Lemma check_pass : forall R M, R b = R 0%nat ->
  steps P (mkC pN 0 (mkState R M)) (mkC (S pN) 0 (mkState R M)).
Proof.
  intros R M H; apply steps_one.
  eapply cstep_bne_direct_not_taken; [apply nth_N | exact H].
Qed.

Lemma check_fail : forall R M f, find_label fin P = Some f -> R b <> R 0%nat ->
  steps P (mkC pN 0 (mkState R M)) (mkC f 0 (mkState R M)).
Proof.
  intros R M f Hf H; apply steps_one.
  eapply cstep_bne_direct_taken; [apply nth_N | exact Hf | exact H].
Qed.

(** *** The assertion block, from the state the two paths reach it in *)

Lemma assert_steps : forall R M σ' f,
  models (mkState R M) σ' -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  steps P (mkC pD 0 (mkState (rupd b f R) M))
          (mkC pE 0 (mkState (rupd b (Z.lxor f (truth (eval σ' e2))) R) M)).
Proof.
  intros R M σ' f Hmod Hcl H0.
  assert (HT2 : run T2 (mkState (rupd b f R) M)
                = mkState (rupd b (Z.lxor f (truth (eval σ' e2))) R) M).
  { unfold T2. rewrite (flag_block_spec e2 b _ σ').
    - cbn [regs mem]. now rewrite rupd_same, rupd_shadow.
    - exact Hmod.
    - intros r Hr; cbn [regs]; rewrite rupd_other by lia; apply Hcl; lia.
    - cbn [regs]; rewrite rupd_other by lia; exact H0.
    - exact Hb. }
  rewrite <- HT2.
  replace pD with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))])) by len_solve.
  replace pE with (length (pre ++ ops T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))]) + length T2)%nat by len_solve.
  eapply steps_ops_l. unfold P, if_code; norm_app; reflexivity.
Qed.

(** *** The entry test: [rt := (e1 != 0)] *)

Lemma test_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  steps P (mkC n0 0 (mkState R M))
          (mkC pB 0 (mkState (rupd b (truth (eval σ e1)) R) M)).
Proof.
  intros R M σ Hmod Hcl H0.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT1 : run T1 (mkState R M) = mkState (rupd b (truth (eval σ e1)) R) M).
  { unfold T1. rewrite (flag_block_spec e1 b _ σ).
    - cbn [regs mem]. rewrite Hb0. now rewrite Z.lxor_0_l.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  rewrite <- HT1.
  replace pB with (length pre + length T1)%nat by len_solve.
  eapply steps_ops. unfold P, if_code; norm_app; reflexivity.
Qed.

(** *** The then path, up to the flag check: the flag is [1 xor (e2 != 0)] *)

Lemma then_path : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 ->
  steps (preA ++ pa ++ postA) (mkC (length preA) 0 (mkState R M))
        (mkC (length preA + length pa) 0 (mkState R' M')) ->
  models (mkState R' M') σ' -> R' = R ->
  steps P (mkC (length pre) 0 (mkState R M))
          (mkC pN 0 (mkState (rupd b (Z.lxor 1 (truth (eval σ' e2))) R) M')).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 Hbody Hmod' HR; subst R'.
  rewrite preA_eq, len_preA in Hbody.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  (* test: rt := 1 *)
  eapply steps_trans; [apply (test_steps R M σ Hmod Hcl H0) |].
  rewrite (truth_nz _ He1).
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
  (* assertion: rt := 1 xor (e2 != 0) *)
  eapply steps_trans; [apply (assert_steps R M' σ' 1 Hmod' Hcl H0) |].
  apply join_steps.
Qed.

(** *** The else path, up to the flag check: the flag is [(e2 != 0)] *)

Lemma else_path : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 ->
  steps (preB ++ pb ++ postB) (mkC (length preB) 0 (mkState R M))
        (mkC (length preB + length pb) 0 (mkState R' M')) ->
  models (mkState R' M') σ' -> R' = R ->
  steps P (mkC (length pre) 0 (mkState R M))
          (mkC pN 0 (mkState (rupd b (truth (eval σ' e2)) R) M')).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 Hbody Hmod' HR; subst R'.
  rewrite preB_eq, len_preB in Hbody.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  (* test: rt := 0, i.e. the state is unchanged *)
  eapply steps_trans; [apply (test_steps R M σ Hmod Hcl H0) |].
  rewrite He1, truth_0, rupd_zero by exact Hb0.
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
  (* assertion: rt := 0 xor (e2 != 0) *)
  assert (HA := assert_steps R M' σ' 0 Hmod' Hcl H0).
  rewrite rupd_zero in HA by exact Hb0. rewrite Z.lxor_0_l in HA.
  eapply steps_trans; [exact HA |].
  apply join_steps.
Qed.

(** *** Valid runs: the flag is clean and `BNE rt r0 finish` falls through *)

Lemma if_true_run : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 -> eval σ' e2 <> 0 ->
  steps (preA ++ pa ++ postA) (mkC (length preA) 0 (mkState R M))
        (mkC (length preA + length pa) 0 (mkState R' M')) ->
  models (mkState R' M') σ' -> R' = R ->
  steps P (mkC (length pre) 0 (mkState R M))
          (mkC (length pre + length (if_code fin b n e1 e2 pa pb)) 0 (mkState R M')).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 He2 Hbody Hmod' HR.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply steps_trans; [eapply then_path; eassumption |].
  rewrite (truth_nz _ He2). change (Z.lxor 1 1) with 0. rewrite rupd_zero by exact Hb0.
  rewrite end_pos. apply check_pass. congruence.
Qed.

Lemma if_false_run : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 -> eval σ' e2 = 0 ->
  steps (preB ++ pb ++ postB) (mkC (length preB) 0 (mkState R M))
        (mkC (length preB + length pb) 0 (mkState R' M')) ->
  models (mkState R' M') σ' -> R' = R ->
  steps P (mkC (length pre) 0 (mkState R M))
          (mkC (length pre + length (if_code fin b n e1 e2 pa pb)) 0 (mkState R M')).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 He2 Hbody Hmod' HR.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply steps_trans; [eapply else_path; eassumption |].
  rewrite He2, truth_0, rupd_zero by exact Hb0.
  rewrite end_pos. apply check_pass. congruence.
Qed.

(** *** Violated exit assertion: the flag is [1] and the run jumps to
    [finish] — it does not reach the line after the [If]. *)

Lemma if_true_violation : forall R M R' M' σ σ' f,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 -> eval σ' e2 = 0 ->
  steps (preA ++ pa ++ postA) (mkC (length preA) 0 (mkState R M))
        (mkC (length preA + length pa) 0 (mkState R' M')) ->
  models (mkState R' M') σ' -> R' = R ->
  find_label fin P = Some f ->
  steps P (mkC (length pre) 0 (mkState R M)) (mkC f 0 (mkState (rupd b 1 R) M')).
Proof.
  intros R M R' M' σ σ' f Hmod Hcl H0 He1 He2 Hbody Hmod' HR Hf.
  eapply steps_trans; [eapply then_path; eassumption |].
  rewrite He2, truth_0. change (Z.lxor 1 0) with 1.
  apply check_fail; [exact Hf |]. cbn [regs]. rewrite rupd_same, rupd_other by lia. lia.
Qed.

Lemma if_false_violation : forall R M R' M' σ σ' f,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 -> eval σ' e2 <> 0 ->
  steps (preB ++ pb ++ postB) (mkC (length preB) 0 (mkState R M))
        (mkC (length preB + length pb) 0 (mkState R' M')) ->
  models (mkState R' M') σ' -> R' = R ->
  find_label fin P = Some f ->
  steps P (mkC (length pre) 0 (mkState R M)) (mkC f 0 (mkState (rupd b 1 R) M')).
Proof.
  intros R M R' M' σ σ' f Hmod Hcl H0 He1 He2 Hbody Hmod' HR Hf.
  eapply steps_trans; [eapply else_path; eassumption |].
  rewrite (truth_nz _ He2).
  apply check_fail; [exact Hf |]. cbn [regs]. rewrite rupd_same, rupd_other by lia. lia.
Qed.

End IfLayout.

(** ** Main theorem

    A compiled statement, embedded between any [pre] and [post] that do not
    define its labels, started with [br = 0] at its first line, reaches the
    line after its last with [br = 0], a memory representing the final store
    and the *same* register file.  [b <> 0] and [regs ms 0 = 0] stand for
    the hard-wired zero register `r0` the tests compare against.  Nothing is
    assumed about [fin]: on a valid execution every `BNE rt r0 finish`
    falls through. *)

Theorem compile_c_spec : forall fin st σ σ' b n p n' ms pre post,
  exec_c st σ σ' -> wf_cstmt st ->
  compile_c fin st b n = (p, n') -> b <> 0%nat ->
  models ms σ -> clean_above b ms -> regs ms 0%nat = 0 ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  exists ms',
    steps (pre ++ p ++ post)
          (mkC (length pre) 0 ms) (mkC (length pre + length p) 0 ms')
    /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros fin st σ σ' b n p n' ms pre post Hex.
  revert b n p n' ms pre post.
  induction Hex as [s σ σ' Hs | a c σ m σ' Ha IHa Hc IHc
                   | e1 a c e2 σ σ' He1 Ha IHa He2 | e1 a c e2 σ σ' He1 Hc IHc He2];
    intros b n p n' ms pre post Hwf Hcomp Hb Hmod Hcl H0 Hpre Hpost; simpl in Hcomp.
  - (* CBase *)
    injection Hcomp as <- <-.
    destruct (compile_at_spec b s σ σ' ms Hs Hwf Hmod Hcl H0 Hb) as [Hm Hr].
    exists (run (compile_at b s) ms). split; [| split; assumption].
    rewrite length_ops. eapply steps_ops. reflexivity.
  - (* CSeq *)
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_c fin a b n) as [p1 n1] eqn:E1.
    destruct (compile_c fin c b n1) as [p2 n2] eqn:E2.
    injection Hcomp as <- <-.
    destruct (compile_c_labels _ _ _ _ _ _ E1) as [Hle1 Hin1].
    destruct (compile_c_labels _ _ _ _ _ _ E2) as [Hle2 Hin2].
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
    destruct (compile_c fin a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_c fin c (S b) na) as [pb nb] eqn:Eb.
    injection Hcomp as <- <-.
    destruct (compile_c_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_c_labels _ _ _ _ _ _ Eb) as [Hleb Hinb].
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
  - (* CIf, else path *)
    destruct Hwf as [Hwfa Hwfc].
    destruct (compile_c fin a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_c fin c (S b) na) as [pb nb] eqn:Eb.
    injection Hcomp as <- <-.
    destruct (compile_c_labels _ _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_c_labels _ _ _ _ _ _ Eb) as [Hleb Hinb].
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
Qed.

(** ** Whole-program corollary, on the executable machine

    Compile with the real scratch base and labels from 0, start at pc 0 with
    [br = 0]; the fuel interpreter halts by falling off the end.  The
    program need not even define [fin]. *)

Corollary compile_c_program : forall fin st σ σ' p n' ms,
  exec_c st σ σ' -> wf_cstmt st ->
  compile_c fin st scratch 0%nat = (p, n') ->
  models ms σ -> clean_above scratch ms -> regs ms 0%nat = 0 ->
  exists ms' fuel,
    exec_fuel fuel p (mkC 0%nat 0 ms) = Some (mkC (length p) 0 ms')
    /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros fin st σ σ' p n' ms Hex Hwf Hc Hmod Hcl H0.
  destruct (compile_c_spec fin st σ σ' scratch 0%nat p n' ms [] [] Hex Hwf Hc)
    as [ms' [Hst [Hm Hr]]];
    [unfold scratch; lia | exact Hmod | exact Hcl | exact H0
    | intros l Hl; destruct Hl | intros l Hl; destruct Hl |].
  exists ms'.
  rewrite app_nil_r in Hst; cbn [app length] in Hst.
  destruct (steps_exec_fuel p _ _ Hst) as [fuel Hf].
  { cbn [cpc]. apply nth_error_None. lia. }
  exists fuel. auto.
Qed.

(** ** Every line the compiler emits is well-formed (`pisa.is_wf`) *)

Lemma wf_if_code : forall fin b n e1 e2 pa pb,
  wf_lprog pa -> wf_lprog pb -> wf_lprog (if_code fin b n e1 e2 pa pb).
Proof.
  intros fin b n e1 e2 pa pb Ha Hb.
  pose proof (wf_flag_block e1 b). pose proof (wf_flag_block e2 b).
  unfold if_code; wf_lp.
Qed.

Theorem wf_compile_c : forall fin st b n p n', b <> 0%nat ->
  compile_c fin st b n = (p, n') -> wf_lprog p.
Proof.
  intro fin.
  induction st as [s | a IHa c IHc | e1 a IHa c IHc e2]; intros b n p n' Hb Hc; simpl in Hc.
  - injection Hc as <- <-. now apply wf_ops, wf_compile_at.
  - destruct (compile_c fin a b n) as [p1 n1] eqn:E1.
    destruct (compile_c fin c b n1) as [p2 n2] eqn:E2.
    injection Hc as <- <-.
    apply wf_lprog_app; [exact (IHa b n p1 n1 Hb E1) | exact (IHc b n1 p2 n2 Hb E2)].
  - destruct (compile_c fin a (S b) (n + 5)%nat) as [pa na] eqn:E1.
    destruct (compile_c fin c (S b) na) as [pc nc] eqn:E2.
    injection Hc as <- <-.
    apply wf_if_code; [exact (IHa (S b) _ _ _ ltac:(lia) E1) | exact (IHc (S b) _ _ _ ltac:(lia) E2)].
Qed.

(** ** Sanity checks: run the compiler and the machine

    `finish` is label 0 and the statement's labels start at 1;
    [with_finish] appends the `finish:` line (a NOP here: the fuel
    interpreter halts when the pc falls off the end, which is what `FINISH`
    does).  Straight-line bodies at base 4 (the flag is r3). *)

Definition fin_label : label := 0%nat.

Definition with_finish (p : lprog) : lprog := p ++ [(Some fin_label, COp (IAddi 0%nat 0))].

Definition prog_c (st : cstmt) : lprog := with_finish (fst (compile_c fin_label st scratch 1%nat)).

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

(** Non-Boolean tests, valid Janus: [x += 5; if x then y += 1 else y += 2
    fi y + 6] — the test is 5 and the assertion 7, both true.  Before the
    [e != 0] normalisation this left [5 xor 7] in the flag. *)
Definition prog_nonbool : cstmt :=
  CSeq (CBase (Assign 0%nat AAdd (Cst 5)))
       (CIf (Var 0%nat)
            (CBase (Assign 1%nat AAdd (Cst 1)))
            (CBase (Assign 1%nat AAdd (Cst 2)))
            (Bin OAdd (Var 1%nat) (Cst 6))).

Definition observe (p : lprog) (r : option cstate) :=
  match r with
  | Some c => (Nat.eqb (cpc c) (length p), cbr c, mem (cst c) 0, mem (cst c) 1,
               regs (cst c) 3%nat, regs (cst c) 4%nat, regs (cst c) 5%nat,
               regs (cst c) 6%nat, regs (cst c) 7%nat)
  | None => (false, 1, 0, 0, 0, 0, 0, 0, 0)
  end.

Example ex_then :
  let p := prog_c prog_then in
  observe p (exec_fuel 500 p (mkC 0%nat 0 zero_state)) = (true, 0, 3, 1, 0, 0, 0, 0, 0).
Proof. vm_compute. reflexivity. Qed.

Example ex_else :
  let p := prog_c prog_else in
  observe p (exec_fuel 500 p (mkC 0%nat 0 zero_state)) = (true, 0, 3, 2, 0, 0, 0, 0, 0).
Proof. vm_compute. reflexivity. Qed.

Example ex_nonbool :
  let p := prog_c prog_nonbool in
  observe p (exec_fuel 500 p (mkC 0%nat 0 zero_state)) = (true, 0, 5, 1, 0, 0, 0, 0, 0).
Proof. vm_compute. reflexivity. Qed.

(** A violated exit assertion is detected: [fi y ^ 1] on the then path (y = 1)
    leaves the flag [1], so `BNE rt r0 finish` jumps to `finish` with r3 = 1
    — what `pisa_interp.py` reports as garbage at FINISH.  Without a
    `finish` line the same code gets stuck at that branch (no label). *)
Definition prog_violation : cstmt :=
  CSeq (CBase (Assign 0%nat AAdd (Cst 3)))
       (CIf (Bin OXor (Var 0%nat) (Cst 2))
            (CBase (Assign 1%nat AAdd (Cst 1)))
            (CBase (Assign 1%nat AAdd (Cst 2)))
            (Bin OXor (Var 1%nat) (Cst 1))).

Example ex_violation_detected :
  let p := prog_c prog_violation in
  observe p (exec_fuel 500 p (mkC 0%nat 0 zero_state)) = (true, 0, 3, 1, 1, 0, 0, 0, 0).
Proof. vm_compute. reflexivity. Qed.

Example ex_violation_jumps :
  exec_fuel 500 (fst (compile_c fin_label prog_violation scratch 1%nat))
            (mkC 0%nat 0 zero_state) = None.
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
  let p := prog_c prog_nested in
  observe p (exec_fuel 500 p (mkC 0%nat 0 zero_state)) = (true, 0, 3, 5, 0, 0, 0, 0, 0).
Proof. vm_compute. reflexivity. Qed.

(** The nested case compiles with distinct labels and the theorem applies to it. *)
Example ex_nested_labels :
  snd (compile_c fin_label (CIf (Cst 1) (CIf (Cst 1) (CBase Skip) (CBase Skip) (Cst 1))
                               (CBase Skip) (Cst 1)) scratch 1%nat) = 11%nat.
Proof. reflexivity. Qed.

(** ** Axiom footprint *)

Print Assumptions compile_c_spec.
Print Assumptions compile_c_program.
Print Assumptions exec_c_rev.
Print Assumptions compile_at_scratch.
Print Assumptions flag_block_spec.
Print Assumptions wf_compile_c.
