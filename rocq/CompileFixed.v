(** * CompileFixed.v — Compile.v on the fixed-width machine (PISAFixed)

    EXPERIMENT (see FIXED_WIDTH_REPORT.md).  Same compiler as Compile.v
    (the code generator is width-independent and copied verbatim), same
    source semantics (Src.v: Janus over unbounded [Z]), but the target is
    PISAFixed: every ALU result is wrapped into the signed [b]-bit window.

    The finding this file records: Compile.v's correctness statement
    [models (run (compile st) m) σ'] — memory cell [x] holds EXACTLY the
    Janus value [σ' x] — is false on the fixed-width machine, because Janus
    computes [x += 5] in [Z] while the machine computes it modulo [2^b]
    ([compile_spec_unwrapped_fails]).  What does survive is correctness
    MODULO THE WINDOW: with [models_w s σ := mem s (Z.of_nat x) = wrap (σ x)]
    every theorem of Compile.v goes through ([compile_spec_w]), and the
    original statement is recovered exactly when the FINAL store is
    representable ([compile_spec_in_window]) — intermediate stores may
    overflow freely, because the machine's residues track Janus's residues.

    Two further hypotheses appear that Compile.v never needed:
    - [wf_state m]: the initial machine state is representable (the inverse
      blocks that make the translation clean rely on PISAFixed.step_invert,
      which needs it);
    - variable addresses [Z.of_nat x] must themselves be in the window
      ([expr_vars_ok]/[stmt_vars_ok]): [gen_var] loads the address with
      [ADDI], which wraps, so a variable at address >= 2^(b-1) would be read
      from the wrong cell. *)

From Stdlib Require Import ZArith List Lia Bool.
From Stdlib Require Import FunctionalExtensionality.
Require Import RevSMod PISAFixed Src.
Import ListNotations.
Open Scope Z_scope.

(** ** Width-independent part — verbatim Compile.v *)

Definition models (s : state) (σ : store) : Prop :=
  forall x : var, mem s (Z.of_nat x) = σ x.

Definition clean_above (rt : reg) (s : state) : Prop :=
  forall r : reg, (rt <= r)%nat -> regs s r = 0.

Definition scratch : reg := 3%nat.

Lemma rupd_comm : forall r1 r2 v1 v2 f,
  r1 <> r2 -> rupd r1 v1 (rupd r2 v2 f) = rupd r2 v2 (rupd r1 v1 f).
Proof.
  intros r1 r2 v1 v2 f H; apply functional_extensionality; intro r.
  unfold rupd.
  destruct (Nat.eqb_spec r r1), (Nat.eqb_spec r r2); subst; congruence.
Qed.

Lemma rupd_zero : forall r f, f r = 0 -> rupd r 0 f = f.
Proof. intros r f H; rewrite <- H at 1; apply rupd_id. Qed.

Lemma of_nat_inj : forall x y : nat, Z.of_nat x = Z.of_nat y -> x = y.
Proof. intros; now apply Nat2Z.inj. Qed.

Definition op_instr (o : binop) (rd rs : reg) : instr :=
  match o with
  | OAdd => IAdd rd rs
  | OSub => ISub rd rs
  | OXor => IXor rd rs
  end.

Definition aop_instr (o : aop) (rd rs : reg) : instr :=
  match o with
  | AAdd => IAdd rd rs
  | ASub => ISub rd rs
  | AXor => IXor rd rs
  end.

Definition gen_var (x : var) (rt : reg) : code :=
  [ IAddi (S rt) (Z.of_nat x)
  ; IExch (S (S rt)) (S rt)
  ; IXor  rt (S (S rt))
  ; IExch (S (S rt)) (S rt)
  ; ISubi (S rt) (Z.of_nat x) ].

Fixpoint gen_expr (e : expr) (rt : reg) : code :=
  match e with
  | Cst n => [IAddi rt n]
  | Var x => gen_var x rt
  | Bin o e1 e2 =>
      gen_expr e1 rt
      ++ gen_expr e2 (S rt)
      ++ [op_instr o rt (S rt)]
      ++ invert_code (gen_expr e2 (S rt))
  end.

Definition gen_assign (x : var) (o : aop) (e : expr) : code :=
  gen_expr e scratch
  ++ [ IAddi (S scratch) (Z.of_nat x)
     ; IExch (S (S scratch)) (S scratch)
     ; aop_instr o (S (S scratch)) scratch
     ; IExch (S (S scratch)) (S scratch)
     ; ISubi (S scratch) (Z.of_nat x) ]
  ++ invert_code (gen_expr e scratch).

Definition gen_swap (x y : var) : code :=
  [ IAddi scratch (Z.of_nat x)
  ; IAddi (S scratch) (Z.of_nat y)
  ; IExch (S (S scratch)) scratch
  ; IExch (S (S (S scratch))) (S scratch)
  ; IExch (S (S (S scratch))) scratch
  ; IExch (S (S scratch)) (S scratch)
  ; ISubi scratch (Z.of_nat x)
  ; ISubi (S scratch) (Z.of_nat y) ].

Fixpoint compile (st : stmt) : code :=
  match st with
  | Skip         => []
  | Assign x o e => gen_assign x o e
  | Swap x y     => gen_swap x y
  | Seq s1 s2    => compile s1 ++ compile s2
  end.

(* wf_invert_instr / wf_invert_code / wf_gen_expr / wf_compile: UNCHANGED *)
Lemma wf_invert_instr : forall i, wf_instr i -> wf_instr (invert_instr i).
Proof. destruct i; simpl; auto. Qed.

Lemma wf_invert_code : forall c, wf_code c -> wf_code (invert_code c).
Proof.
  intros c H; unfold invert_code, wf_code.
  apply Forall_rev, Forall_map, Forall_impl with (P := wf_instr); auto.
  apply wf_invert_instr.
Qed.

Ltac wf_list :=
  unfold wf_code;
  repeat (apply Forall_cons; [simpl; first [exact I | lia] |]);
  apply Forall_nil.

Lemma wf_gen_expr : forall e rt, wf_code (gen_expr e rt).
Proof.
  induction e as [n | x | o e1 IH1 e2 IH2]; intro rt; simpl.
  - wf_list.
  - unfold gen_var; wf_list.
  - apply wf_code_app; [apply IH1 |].
    apply wf_code_app; [apply IH2 |].
    apply Forall_cons; [destruct o; simpl; lia |].
    apply wf_invert_code, IH2.
Qed.

Lemma wf_compile : forall st, wf_code (compile st).
Proof.
  induction st as [| x o e | x y | s1 IH1 s2 IH2]; simpl.
  - apply Forall_nil.
  - unfold gen_assign.
    apply wf_code_app; [apply wf_gen_expr |].
    repeat (apply Forall_cons; [destruct o; simpl; first [exact I | lia] |]).
    apply wf_invert_code, wf_gen_expr.
  - unfold gen_swap; wf_list.
  - now apply wf_code_app.
Qed.

(** ** The width-parameterised part *)

Section Fixed.

Variable b : nat.
Hypothesis Hb : (0 < b)%nat.

Local Notation wrap := (wrap b).
Local Notation in_window := (in_window b).
Local Notation step := (step b).
Local Notation run := (run b).
Local Notation wf_state := (wf_state b).

(** *** Correspondence modulo the window *)

(** NEW: the memory holds the Janus store reduced into the window. *)
Definition models_w (s : state) (σ : store) : Prop :=
  forall x : var, mem s (Z.of_nat x) = wrap (σ x).

(** NEW: every variable used lives at a representable address. *)
Fixpoint expr_vars_ok (e : expr) : Prop :=
  match e with
  | Cst _ => True
  | Var x => in_window (Z.of_nat x)
  | Bin _ e1 e2 => expr_vars_ok e1 /\ expr_vars_ok e2
  end.

Fixpoint stmt_vars_ok (st : stmt) : Prop :=
  match st with
  | Skip         => True
  | Assign x _ e => in_window (Z.of_nat x) /\ expr_vars_ok e
  | Swap x y     => in_window (Z.of_nat x) /\ in_window (Z.of_nat y)
  | Seq s1 s2    => stmt_vars_ok s1 /\ stmt_vars_ok s2
  end.

Lemma models_models_w : forall s σ,
  wf_state s -> models s σ -> models_w s σ.
Proof.
  intros s σ [_ HM] Hmod x; rewrite Hmod; symmetry; apply wrap_id; [exact Hb |].
  rewrite <- Hmod; apply HM.
Qed.

Lemma models_w_models : forall s σ,
  models_w s σ -> (forall x, in_window (σ x)) -> models s σ.
Proof. intros s σ H Hσ x; rewrite H; apply wrap_id; [exact Hb | apply Hσ]. Qed.

(** The wrapped ALU agrees with Janus's operators modulo the window. *)
Lemma wrap_denote : forall o a c,
  wrap (denote o (wrap a) (wrap c)) = wrap (denote o a c).
Proof.
  destruct o; simpl; intros.
  - now rewrite wrap_add_l, wrap_add_r.
  - now rewrite wrap_sub_l, wrap_sub_r.
  - now rewrite wrap_lxor_l, wrap_lxor_r.
Qed.

Lemma wrap_adenote : forall o a c,
  wrap (adenote o (wrap a) (wrap c)) = wrap (adenote o a c).
Proof.
  destruct o; simpl; intros.
  - now rewrite wrap_add_l, wrap_add_r.
  - now rewrite wrap_sub_l, wrap_sub_r.
  - now rewrite wrap_lxor_l, wrap_lxor_r.
Qed.

Lemma wf_mkState_rupd : forall R M r v,
  wf_state (mkState R M) -> in_window v -> wf_state (mkState (rupd r v R) M).
Proof.
  intros R M r v [HR HM] Hv; split; simpl; [| exact HM].
  apply rupd_in_window; assumption.
Qed.

Lemma wf_mkState_mupd : forall R M a v,
  wf_state (mkState R M) -> in_window v -> wf_state (mkState R (mupd a v M)).
Proof.
  intros R M a v [HR HM] Hv; split; simpl; [exact HR |].
  apply mupd_in_window; assumption.
Qed.

Ltac simp_state :=
  simpl;
  repeat first
    [ rewrite rupd_same
    | rewrite mupd_same
    | rewrite rupd_other by lia
    | rewrite mupd_other by lia ].

(** *** Expression compilation, modulo the window *)

(** gen_var_spec: MODIFIED — [models_w], target gets [wrap (σ x)], and the
    address must be in the window. *)
Lemma gen_var_spec : forall x rt s σ,
  in_window (Z.of_nat x) ->
  models_w s σ -> clean_above rt s ->
  run (gen_var x rt) s = mkState (rupd rt (wrap (σ x)) (regs s)) (mem s).
Proof.
  intros x rt [R M] σ Hax Hmod Hcl; simpl in *.
  assert (H0 : R rt = 0) by (apply Hcl; lia).
  assert (H1 : R (S rt) = 0) by (apply Hcl; lia).
  assert (H2 : R (S (S rt)) = 0) by (apply Hcl; lia).
  unfold gen_var; simp_state.
  rewrite H0, H1, H2.
  rewrite !Z.add_0_l, Z.lxor_0_l.
  rewrite (wrap_id b Hb (Z.of_nat x)) by exact Hax.
  rewrite Hmod, wrap_idem by exact Hb.
  rewrite Z.sub_diag, (wrap_id b Hb 0) by (apply zero_in_window; exact Hb).
  f_equal.
  - apply functional_extensionality; intro r.
    unfold rupd.
    destruct (Nat.eqb_spec r (S rt)) as [E1|E1];
    destruct (Nat.eqb_spec r (S (S rt))) as [E2|E2];
    destruct (Nat.eqb_spec r rt) as [E3|E3];
    subst; cbn; try lia;
    try (now rewrite H1); try (now rewrite H2); try reflexivity.
  - rewrite mupd_shadow, <- Hmod, mupd_id. reflexivity.
Qed.

(** gen_expr_spec: MODIFIED — needs [wf_state s] (for the inverse block),
    [expr_vars_ok e], and delivers [wrap (eval σ e)]. *)
Theorem gen_expr_spec : forall e rt s σ,
  wf_state s -> expr_vars_ok e ->
  models_w s σ -> clean_above rt s ->
  run (gen_expr e rt) s = mkState (rupd rt (wrap (eval σ e)) (regs s)) (mem s).
Proof.
  induction e as [n | x | o e1 IH1 e2 IH2]; intros rt s σ Hwf Hvars Hmod Hcl.
  - (* Cst: the literal is wrapped — PISA.v had [n] here *)
    destruct s as [R M]; simpl in *.
    assert (H0 : R rt = 0) by (apply Hcl; lia).
    unfold run; simpl. rewrite H0. now rewrite Z.add_0_l.
  - (* Var *) now apply gen_var_spec.
  - (* Bin *)
    destruct s as [R M]; cbn [regs mem eval gen_expr] in *.
    destruct Hvars as [Hv1 Hv2].
    rewrite run_app, (IH1 rt (mkState R M) σ Hwf Hv1 Hmod Hcl); cbn [regs mem].
    assert (Hwf1 : wf_state (mkState (rupd rt (wrap (eval σ e1)) R) M))
      by (apply wf_mkState_rupd; [exact Hwf | apply wrap_range; exact Hb]).
    assert (Hmod1 : models_w (mkState (rupd rt (wrap (eval σ e1)) R) M) σ) by exact Hmod.
    assert (Hcl1 : clean_above (S rt) (mkState (rupd rt (wrap (eval σ e1)) R) M)).
    { intros r Hr; cbn [regs mem]; rewrite rupd_other by lia; apply Hcl; lia. }
    rewrite run_app, (IH2 (S rt) _ σ Hwf1 Hv2 Hmod1 Hcl1); cbn [regs mem].
    rewrite run_app, run_one.
    assert (HwfX : wf_state (mkState (rupd rt (wrap (denote o (eval σ e1) (eval σ e2))) R) M))
      by (apply wf_mkState_rupd; [exact Hwf | apply wrap_range; exact Hb]).
    assert (HmodX : models_w (mkState (rupd rt (wrap (denote o (eval σ e1) (eval σ e2))) R) M) σ)
      by exact Hmod.
    assert (HclX : clean_above (S rt)
                     (mkState (rupd rt (wrap (denote o (eval σ e1) (eval σ e2))) R) M)).
    { intros r Hr; cbn [regs mem]; rewrite rupd_other by lia; apply Hcl; lia. }
    assert (HX : step (op_instr o rt (S rt))
                      (mkState (rupd (S rt) (wrap (eval σ e2)) (rupd rt (wrap (eval σ e1)) R)) M)
                 = run (gen_expr e2 (S rt))
                       (mkState (rupd rt (wrap (denote o (eval σ e1) (eval σ e2))) R) M)).
    { rewrite (IH2 (S rt) _ σ HwfX Hv2 HmodX HclX); cbn [regs mem].
      (* the wrapped operands combine to the wrapped result: wrap_denote *)
      destruct o; simp_state; f_equal.
      - rewrite (rupd_comm rt (S rt)) by lia; rewrite rupd_shadow.
        now rewrite wrap_add_l, wrap_add_r.
      - rewrite (rupd_comm rt (S rt)) by lia; rewrite rupd_shadow.
        now rewrite wrap_sub_l, wrap_sub_r.
      - rewrite (rupd_comm rt (S rt)) by lia; rewrite rupd_shadow.
        now rewrite wrap_lxor_l, wrap_lxor_r. }
    rewrite HX.
    rewrite run_invert_code by (exact Hb || apply wf_gen_expr || exact HwfX).
    reflexivity.
Qed.

(** *** Statement compilation, modulo the window *)

Lemma models_w_update : forall s σ x v,
  models_w s σ ->
  models_w (mkState (regs s) (mupd (Z.of_nat x) (wrap v) (mem s))) (update σ x v).
Proof.
  intros s σ x v H y; simpl.
  destruct (Nat.eq_dec x y) as [->|Hne].
  - rewrite mupd_same, update_eq. reflexivity.
  - rewrite mupd_other by (intro Hc; apply Hne, of_nat_inj; congruence).
    rewrite update_neq by assumption. apply H.
Qed.

(** gen_assign_spec: MODIFIED — the stored value is [wrap (σ x op e)], plus
    the three new hypotheses. *)
Lemma gen_assign_spec : forall x o e s σ,
  occurs x e = false ->
  wf_state s -> in_window (Z.of_nat x) -> expr_vars_ok e ->
  models_w s σ -> clean_above scratch s ->
  run (gen_assign x o e) s
  = mkState (regs s)
            (mupd (Z.of_nat x) (wrap (adenote o (σ x) (eval σ e))) (mem s)).
Proof.
  intros x o e [R M] σ Hocc Hwf Hax Hvars Hmod Hcl; cbn [regs mem] in *.
  assert (H0 : R scratch = 0) by (apply Hcl; lia).
  assert (H1 : R (S scratch) = 0) by (apply Hcl; lia).
  assert (H2 : R (S (S scratch)) = 0) by (apply Hcl; lia).
  assert (Hblock :
    run [ IAddi (S scratch) (Z.of_nat x)
        ; IExch (S (S scratch)) (S scratch)
        ; aop_instr o (S (S scratch)) scratch
        ; IExch (S (S scratch)) (S scratch)
        ; ISubi (S scratch) (Z.of_nat x) ]
        (mkState (rupd scratch (wrap (eval σ e)) R) M)
    = mkState (rupd scratch (wrap (eval σ e)) R)
              (mupd (Z.of_nat x) (wrap (adenote o (σ x) (eval σ e))) M)).
  { destruct o; simp_state; rewrite ?H1, ?H2, ?Z.add_0_l;
    rewrite (wrap_id b Hb (Z.of_nat x)) by exact Hax;
    rewrite ?Hmod;
    rewrite ?Z.sub_diag, ?(wrap_id b Hb 0) by (apply zero_in_window; exact Hb);
    cbn [adenote];
    rewrite ?(wrap_add_l b Hb), ?(wrap_add_r b Hb), ?(wrap_sub_l b Hb), ?(wrap_sub_r b Hb),
            ?(wrap_lxor_l b Hb), ?(wrap_lxor_r b Hb);
    f_equal;
    try (apply functional_extensionality; intro r; unfold rupd;
         destruct (Nat.eqb_spec r (S scratch)) as [E1|E1];
         destruct (Nat.eqb_spec r (S (S scratch))) as [E2|E2];
         destruct (Nat.eqb_spec r scratch) as [E3|E3];
         subst; cbn; try lia;
         try (now rewrite H1); try (now rewrite H2); try reflexivity);
    try (rewrite mupd_shadow; reflexivity). }
  unfold gen_assign.
  rewrite run_app, (gen_expr_spec e scratch (mkState R M) σ Hwf Hvars Hmod Hcl); cbn [regs mem].
  rewrite run_app, Hblock.
  assert (Hmod' : models_w (mkState R (mupd (Z.of_nat x) (wrap (adenote o (σ x) (eval σ e))) M))
                           (update σ x (adenote o (σ x) (eval σ e))))
    by (apply (models_w_update (mkState R M) σ x (adenote o (σ x) (eval σ e))); exact Hmod).
  assert (Hwf' : wf_state (mkState R (mupd (Z.of_nat x) (wrap (adenote o (σ x) (eval σ e))) M)))
    by (apply wf_mkState_mupd; [exact Hwf | apply wrap_range; exact Hb]).
  assert (Hcl' : clean_above scratch
                   (mkState R (mupd (Z.of_nat x) (wrap (adenote o (σ x) (eval σ e))) M)))
    by exact Hcl.
  assert (Hev : eval (update σ x (adenote o (σ x) (eval σ e))) e = eval σ e)
    by (now apply eval_update_notin).
  assert (Hunc : mkState (rupd scratch (wrap (eval σ e)) R)
                         (mupd (Z.of_nat x) (wrap (adenote o (σ x) (eval σ e))) M)
                 = run (gen_expr e scratch)
                       (mkState R (mupd (Z.of_nat x) (wrap (adenote o (σ x) (eval σ e))) M))).
  { rewrite (gen_expr_spec e scratch _ _ Hwf' Hvars Hmod' Hcl'); cbn [regs mem]. now rewrite Hev. }
  rewrite Hunc.
  apply run_invert_code; [exact Hb | apply wf_gen_expr | exact Hwf'].
Qed.

(** gen_swap_spec: MODIFIED — addresses in the window; values are moved, so
    the statement is literally Compile.v's with [σ x] read through [models_w]. *)
Lemma gen_swap_spec : forall x y s σ,
  x <> y ->
  in_window (Z.of_nat x) -> in_window (Z.of_nat y) ->
  models_w s σ -> clean_above scratch s ->
  run (gen_swap x y) s
  = mkState (regs s)
            (mupd (Z.of_nat y) (wrap (σ x)) (mupd (Z.of_nat x) (wrap (σ y)) (mem s))).
Proof.
  intros x y [R M] σ Hxy Hax Hay Hmod Hcl; cbn [regs mem] in *.
  assert (Haxy : Z.of_nat x <> Z.of_nat y) by (intro Hc; apply Hxy, of_nat_inj, Hc).
  assert (Hayx : Z.of_nat y <> Z.of_nat x) by (now apply Z.neq_sym).
  assert (H0 : R scratch = 0) by (apply Hcl; lia).
  assert (H1 : R (S scratch) = 0) by (apply Hcl; lia).
  assert (H2 : R (S (S scratch)) = 0) by (apply Hcl; lia).
  assert (H3 : R (S (S (S scratch))) = 0) by (apply Hcl; lia).
  unfold gen_swap; simp_state.
  rewrite H0, H1, H2, H3.
  rewrite !Z.add_0_l.
  rewrite (wrap_id b Hb (Z.of_nat x)) by exact Hax.
  rewrite (wrap_id b Hb (Z.of_nat y)) by exact Hay.
  rewrite !Hmod.
  (* the memory lookups could not be resolved while the addresses were still
     wrapped; now that they are literal, resolve them *)
  simp_state. rewrite ?Hmod.
  rewrite !Z.sub_diag, (wrap_id b Hb 0) by (apply zero_in_window; exact Hb).
  f_equal.
  - apply functional_extensionality; intro r; unfold rupd.
    destruct (Nat.eqb_spec r (S scratch)) as [E1|E1];
    destruct (Nat.eqb_spec r scratch) as [E2|E2];
    destruct (Nat.eqb_spec r (S (S scratch))) as [E3|E3];
    destruct (Nat.eqb_spec r (S (S (S scratch)))) as [E4|E4];
    subst; cbn; try lia;
    try (rewrite ?H0, ?H1, ?H2, ?H3; lia); try reflexivity.
  - apply functional_extensionality; intro a; unfold mupd.
    destruct (Z.eqb_spec a (Z.of_nat y)) as [F1|F1];
    destruct (Z.eqb_spec a (Z.of_nat x)) as [F2|F2];
    subst; cbn; try congruence; try reflexivity; try apply Hmod.
Qed.

(** *** Main theorem, modulo the window *)

(** compile_spec_w: Compile.v's [compile_spec] with [models] replaced by
    [models_w] and the two new hypotheses.  Intermediate Janus stores may
    leave the window; only residues are tracked. *)
Theorem compile_spec_w : forall st σ σ' ms,
  exec st σ σ' -> wf_stmt st -> stmt_vars_ok st ->
  wf_state ms -> models_w ms σ -> clean_above scratch ms ->
  models_w (run (compile st) ms) σ' /\ regs (run (compile st) ms) = regs ms.
Proof.
  intros st σ σ' ms H; revert ms; induction H; intros ms Hwf Hvars Hst Hmod Hcl.
  - split; [exact Hmod | reflexivity].
  - cbn [compile]; cbn [stmt_vars_ok] in Hvars; destruct Hvars as [Hax Hve].
    rewrite (gen_assign_spec x o e ms s H Hst Hax Hve Hmod Hcl); cbn [regs mem]; split.
    + apply (models_w_update ms s x). exact Hmod.
    + reflexivity.
  - cbn [wf_stmt] in Hwf; cbn [stmt_vars_ok] in Hvars; destruct Hvars as [Hax Hay]; cbn [compile].
    rewrite (gen_swap_spec x y ms s Hwf Hax Hay Hmod Hcl); cbn [regs mem]; split;
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
  - cbn [wf_stmt] in Hwf; destruct Hwf as [Hwf1 Hwf2].
    cbn [stmt_vars_ok] in Hvars; destruct Hvars as [Hv1 Hv2]; cbn [compile].
    rewrite run_app.
    destruct (IHexec1 ms Hwf1 Hv1 Hst Hmod Hcl) as [Hm1 Hr1].
    assert (Hst1 : wf_state (run (compile s1) ms)) by (apply run_wf; [exact Hb | exact Hst]).
    assert (Hcl1 : clean_above scratch (run (compile s1) ms))
      by (intros r Hr; rewrite Hr1; now apply Hcl).
    destruct (IHexec2 (run (compile s1) ms) Hwf2 Hv2 Hst1 Hm1 Hcl1) as [Hm2 Hr2].
    split; [exact Hm2 | now rewrite Hr2, Hr1].
Qed.

(** *** Compile.v's exact statement: recovered when the final store fits *)

(** compile_spec (Compile.v's statement) holds under the extra hypotheses
    [wf_state ms], [stmt_vars_ok st] and [forall x, in_window (σ' x)].  Note
    that only the FINAL store must be representable. *)
Theorem compile_spec_in_window : forall st σ σ' ms,
  exec st σ σ' -> wf_stmt st -> stmt_vars_ok st ->
  wf_state ms -> models ms σ -> clean_above scratch ms ->
  (forall x, in_window (σ' x)) ->
  models (run (compile st) ms) σ' /\ regs (run (compile st) ms) = regs ms.
Proof.
  intros st σ σ' ms Hex Hwf Hvars Hst Hmod Hcl Hfit.
  destruct (compile_spec_w st σ σ' ms Hex Hwf Hvars Hst (models_models_w ms σ Hst Hmod) Hcl)
    as [Hm Hr].
  split; [apply models_w_models; assumption | exact Hr].
Qed.

(** ...and FAILS without it: [x += 2^(b-1)] from [x = 0] gives the Janus value
    [2^(b-1)], which the machine stores as [-2^(b-1)]. *)
Theorem compile_spec_unwrapped_fails :
  exists st σ σ' ms,
    exec st σ σ' /\ wf_stmt st /\ stmt_vars_ok st /\
    wf_state ms /\ models ms σ /\ clean_above scratch ms /\
    ~ models (run (compile st) ms) σ'.
Proof.
  pose proof (half_pos b Hb) as Hh.
  exists (Assign 0%nat AAdd (Cst (half b))), empty,
         (update empty 0%nat (adenote AAdd (empty 0%nat) (eval empty (Cst (half b))))),
         zero_state.
  assert (Hwf0 : wf_state zero_state) by (apply wf_zero_state; exact Hb).
  assert (Hmod0 : models zero_state empty) by (intro x; reflexivity).
  assert (Hcl0 : clean_above scratch zero_state) by (intros r _; reflexivity).
  assert (Hvars : stmt_vars_ok (Assign 0%nat AAdd (Cst (half b))))
    by (split; [apply zero_in_window; exact Hb | exact I]).
  refine (conj _ (conj I (conj Hvars (conj Hwf0 (conj Hmod0 (conj Hcl0 _)))))).
  - apply E_Assign; reflexivity.
  - intro Hm.
    destruct (compile_spec_w _ _ _ zero_state
                (E_Assign 0%nat AAdd (Cst (half b)) empty eq_refl) I Hvars Hwf0
                (models_models_w _ _ Hwf0 Hmod0) Hcl0) as [Hmw _].
    specialize (Hm 0%nat); specialize (Hmw 0%nat).
    rewrite Hm in Hmw.
    rewrite update_eq in Hmw.
    cbn [adenote eval empty] in Hmw.
    rewrite Z.add_0_l, wrap_half in Hmw.
    lia.
Qed.

(** compile_reversible: MODIFIED — needs [wf_state s]. *)
Corollary compile_reversible : forall st s,
  wf_state s -> run (invert_code (compile st)) (run (compile st) s) = s.
Proof. intros; apply run_invert_code; [exact Hb | apply wf_compile | assumption]. Qed.

End Fixed.
