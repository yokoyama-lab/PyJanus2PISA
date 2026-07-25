(** * Compile.v — the Janus → PISA compiler and its correctness proof

    The compiler follows the *clean translation* of

      H. B. Axelsen, "Clean Translation of an Imperative Reversible
      Programming Language", CC 2011, LNCS 6601, pp. 144–163,

    which is also what [codegen.py] in this repository implements: an
    expression is evaluated into a scratch register, used, and then
    *unevaluated* by running its own code backwards, so that no garbage is
    left behind.

    The correctness statement is therefore two properties at once:

      - semantic preservation — the final memory represents the final store;
      - cleanliness — every scratch register is back to 0, i.e. the register
        file is *exactly* the one we started with.

    Both are captured by a single state equation ([gen_expr_spec],
    [compile_spec]).

    Memory layout: variable [x : nat] lives at address [Z.of_nat x], matching
    the [DATA] words emitted by [codegen.py].  Registers 0,1,2 are reserved
    (r0 = 0, r1 = stack pointer, r2 = return offset), so scratch starts at 3. *)

From Stdlib Require Import ZArith List Lia Bool.
From Stdlib Require Import FunctionalExtensionality.
Require Import PISA Src.
Import ListNotations.
Open Scope Z_scope.

(** ** Correspondence between a PISA state and a Janus store *)

Definition models (s : state) (σ : store) : Prop :=
  forall x : var, mem s (Z.of_nat x) = σ x.

(** All scratch registers from [rt] upwards are cleared. *)
Definition clean_above (rt : reg) (s : state) : Prop :=
  forall r : reg, (rt <= r)%nat -> regs s r = 0.

Definition scratch : reg := 3%nat.

(** ** Auxiliary map lemmas *)

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

(** ** The compiler *)

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

(** Load variable [x] into [rt], using [S rt] as the address register and
    [S (S rt)] as the exchange buffer.  The value is copied out with XOR and
    the cell is put straight back, so memory is unchanged. *)
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
      ++ invert_code (gen_expr e2 (S rt))   (* unevaluate the right operand *)
  end.

(** [x op= e]: evaluate [e] into [scratch], apply it to the memory cell of
    [x] through an exchange, then unevaluate [e]. *)
Definition gen_assign (x : var) (o : aop) (e : expr) : code :=
  gen_expr e scratch
  ++ [ IAddi (S scratch) (Z.of_nat x)
     ; IExch (S (S scratch)) (S scratch)
     ; aop_instr o (S (S scratch)) scratch
     ; IExch (S (S scratch)) (S scratch)
     ; ISubi (S scratch) (Z.of_nat x) ]
  ++ invert_code (gen_expr e scratch).

(** [x <=> y]: lift both cells into registers and put them back crosswise. *)
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

(** ** Emitted code is well-formed

    Every instruction the compiler emits has distinct operand registers, so
    [PISA.run_invert_code] applies to it — this is what makes unevaluation
    sound. *)

Lemma wf_invert_instr : forall i, wf_instr i -> wf_instr (invert_instr i).
Proof. destruct i; simpl; auto. Qed.

Lemma wf_invert_code : forall c, wf_code c -> wf_code (invert_code c).
Proof.
  intros c H; unfold invert_code, wf_code.
  apply Forall_rev, Forall_map, Forall_impl with (P := wf_instr); auto.
  apply wf_invert_instr.
Qed.

(** Discharge [wf_code] for a literal list of instructions. *)
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

(** ** Expression compilation is correct and clean *)

(** Reduce a [run] of a literal block to a nest of [rupd]/[mupd] with all
    register and memory lookups resolved. *)
Ltac simp_state :=
  simpl;
  repeat first
    [ rewrite rupd_same
    | rewrite mupd_same
    | rewrite rupd_other by lia
    | rewrite mupd_other by lia ].

Lemma gen_var_spec : forall x rt s σ,
  models s σ -> clean_above rt s ->
  run (gen_var x rt) s = mkState (rupd rt (σ x) (regs s)) (mem s).
Proof.
  intros x rt [R M] σ Hmod Hcl; simpl in *.
  assert (H0 : R rt = 0) by (apply Hcl; lia).
  assert (H1 : R (S rt) = 0) by (apply Hcl; lia).
  assert (H2 : R (S (S rt)) = 0) by (apply Hcl; lia).
  unfold gen_var; simp_state.
  rewrite H0, H1, H2.
  rewrite !Z.add_0_l, Z.lxor_0_l, Hmod.
  f_equal.
  - (* every scratch register is back to its original 0 *)
    apply functional_extensionality; intro r.
    unfold rupd.
    destruct (Nat.eqb_spec r (S rt)) as [E1|E1];
    destruct (Nat.eqb_spec r (S (S rt))) as [E2|E2];
    destruct (Nat.eqb_spec r rt) as [E3|E3];
    subst; cbn; try lia;
    try (now rewrite H1); try (now rewrite H2); try reflexivity.
  - (* memory is restored *)
    rewrite mupd_shadow, <- Hmod, mupd_id. reflexivity.
Qed.

Theorem gen_expr_spec : forall e rt s σ,
  models s σ -> clean_above rt s ->
  run (gen_expr e rt) s = mkState (rupd rt (eval σ e) (regs s)) (mem s).
Proof.
  induction e as [n | x | o e1 IH1 e2 IH2]; intros rt s σ Hmod Hcl.
  - (* Cst *)
    destruct s as [R M]; simpl in *.
    assert (H0 : R rt = 0) by (apply Hcl; lia).
    unfold run; simpl. rewrite H0. now rewrite Z.add_0_l.
  - (* Var *) now apply gen_var_spec.
  - (* Bin *)
    destruct s as [R M]; cbn [regs mem eval gen_expr] in *.
    (* left operand into rt *)
    rewrite run_app, (IH1 rt (mkState R M) σ Hmod Hcl); cbn [regs mem].
    (* right operand into S rt *)
    assert (Hmod1 : models (mkState (rupd rt (eval σ e1) R) M) σ) by exact Hmod.
    assert (Hcl1 : clean_above (S rt) (mkState (rupd rt (eval σ e1) R) M)).
    { intros r Hr; cbn [regs mem]; rewrite rupd_other by lia; apply Hcl; lia. }
    rewrite run_app, (IH2 (S rt) _ σ Hmod1 Hcl1); cbn [regs mem].
    (* combine the two operands in rt *)
    rewrite run_app, run_one.
    (* The combined state is exactly what [gen_expr e2] produces from the
       post-combination state, so the trailing inverse block cancels it. *)
    assert (HmodX : models (mkState (rupd rt (denote o (eval σ e1) (eval σ e2)) R) M) σ)
      by exact Hmod.
    assert (HclX : clean_above (S rt)
                     (mkState (rupd rt (denote o (eval σ e1) (eval σ e2)) R) M)).
    { intros r Hr; cbn [regs mem]; rewrite rupd_other by lia; apply Hcl; lia. }
    assert (HX : step (op_instr o rt (S rt))
                      (mkState (rupd (S rt) (eval σ e2) (rupd rt (eval σ e1) R)) M)
                 = run (gen_expr e2 (S rt))
                       (mkState (rupd rt (denote o (eval σ e1) (eval σ e2)) R) M)).
    { rewrite (IH2 (S rt) _ σ HmodX HclX); cbn [regs mem].
      destruct o; simp_state; f_equal;
        rewrite (rupd_comm rt (S rt)) by lia; now rewrite rupd_shadow. }
    rewrite HX.
    (* running the inverse of a well-formed block undoes it *)
    rewrite run_invert_code by apply wf_gen_expr.
    reflexivity.
Qed.

(** ** Statement compilation is correct and clean *)

Lemma models_update : forall s σ x v,
  models s σ -> models (mkState (regs s) (mupd (Z.of_nat x) v (mem s))) (update σ x v).
Proof.
  intros s σ x v H y; simpl.
  destruct (Nat.eq_dec x y) as [->|Hne].
  - rewrite mupd_same, update_eq. reflexivity.
  - rewrite mupd_other by (intro Hc; apply Hne, of_nat_inj; congruence).
    rewrite update_neq by assumption. apply H.
Qed.

Lemma gen_assign_spec : forall x o e s σ,
  occurs x e = false ->
  models s σ -> clean_above scratch s ->
  run (gen_assign x o e) s
  = mkState (regs s)
            (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) (mem s)).
Proof.
  intros x o e [R M] σ Hocc Hmod Hcl; cbn [regs mem] in *.
  assert (H0 : R scratch = 0) by (apply Hcl; lia).
  assert (H1 : R (S scratch) = 0) by (apply Hcl; lia).
  assert (H2 : R (S (S scratch)) = 0) by (apply Hcl; lia).
  (* the five-instruction store block: read the cell, apply the operator,
     put it back, leaving [scratch] (which holds the value of [e]) untouched *)
  assert (Hblock :
    run [ IAddi (S scratch) (Z.of_nat x)
        ; IExch (S (S scratch)) (S scratch)
        ; aop_instr o (S (S scratch)) scratch
        ; IExch (S (S scratch)) (S scratch)
        ; ISubi (S scratch) (Z.of_nat x) ]
        (mkState (rupd scratch (eval σ e) R) M)
    = mkState (rupd scratch (eval σ e) R)
              (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M)).
  { destruct o; simp_state; rewrite ?H1, ?H2, ?Z.add_0_l, ?Hmod;
    f_equal;
    try (apply functional_extensionality; intro r; unfold rupd;
         destruct (Nat.eqb_spec r (S scratch)) as [E1|E1];
         destruct (Nat.eqb_spec r (S (S scratch))) as [E2|E2];
         destruct (Nat.eqb_spec r scratch) as [E3|E3];
         subst; cbn; try lia;
         try (now rewrite H1); try (now rewrite H2); try reflexivity);
    try (rewrite mupd_shadow; reflexivity). }
  unfold gen_assign.
  rewrite run_app, (gen_expr_spec e scratch (mkState R M) σ Hmod Hcl); cbn [regs mem].
  rewrite run_app, Hblock.
  (* unevaluate [e]; sound because [x] does not occur in [e], so the value of
     [e] is the same before and after the store *)
  assert (Hmod' : models (mkState R (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M))
                         (update σ x (adenote o (σ x) (eval σ e))))
    by (apply (models_update (mkState R M) σ x (adenote o (σ x) (eval σ e))); exact Hmod).
  assert (Hcl' : clean_above scratch
                   (mkState R (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M)))
    by exact Hcl.
  assert (Hev : eval (update σ x (adenote o (σ x) (eval σ e))) e = eval σ e)
    by (now apply eval_update_notin).
  assert (Hunc : mkState (rupd scratch (eval σ e) R)
                         (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M)
                 = run (gen_expr e scratch)
                       (mkState R (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M))).
  { rewrite (gen_expr_spec e scratch _ _ Hmod' Hcl'); cbn [regs mem]. now rewrite Hev. }
  rewrite Hunc.
  apply run_invert_code; apply wf_gen_expr.
Qed.

Lemma gen_swap_spec : forall x y s σ,
  x <> y ->
  models s σ -> clean_above scratch s ->
  run (gen_swap x y) s
  = mkState (regs s)
            (mupd (Z.of_nat y) (σ x) (mupd (Z.of_nat x) (σ y) (mem s))).
Proof.
  intros x y [R M] σ Hxy Hmod Hcl; cbn [regs mem] in *.
  assert (Hax : Z.of_nat x <> Z.of_nat y) by (intro Hc; apply Hxy, of_nat_inj, Hc).
  assert (Hay : Z.of_nat y <> Z.of_nat x) by (now apply Z.neq_sym).
  assert (H0 : R scratch = 0) by (apply Hcl; lia).
  assert (H1 : R (S scratch) = 0) by (apply Hcl; lia).
  assert (H2 : R (S (S scratch)) = 0) by (apply Hcl; lia).
  assert (H3 : R (S (S (S scratch))) = 0) by (apply Hcl; lia).
  unfold gen_swap; simp_state.
  rewrite H0, H1, H2, H3.
  rewrite !Z.add_0_l, !Hmod.
  f_equal.
  - (* all four scratch registers are back to 0 *)
    apply functional_extensionality; intro r; unfold rupd.
    destruct (Nat.eqb_spec r (S scratch)) as [E1|E1];
    destruct (Nat.eqb_spec r scratch) as [E2|E2];
    destruct (Nat.eqb_spec r (S (S scratch))) as [E3|E3];
    destruct (Nat.eqb_spec r (S (S (S scratch)))) as [E4|E4];
    subst; cbn; try lia;
    try (rewrite ?H0, ?H1, ?H2, ?H3; lia); try reflexivity.
  - (* the two cells hold each other's old value *)
    apply functional_extensionality; intro a; unfold mupd.
    destruct (Z.eqb_spec a (Z.of_nat y)) as [F1|F1];
    destruct (Z.eqb_spec a (Z.of_nat x)) as [F2|F2];
    subst; cbn; try congruence; try reflexivity.
Qed.

(** ** Main theorem: semantic preservation with a clean register file *)

Theorem compile_spec : forall st σ σ' ms,
  exec st σ σ' -> wf_stmt st ->
  models ms σ -> clean_above scratch ms ->
  models (run (compile st) ms) σ' /\ regs (run (compile st) ms) = regs ms.
Proof.
  intros st σ σ' ms H; revert ms; induction H; intros ms Hwf Hmod Hcl.
  - (* Skip *) split; [exact Hmod | reflexivity].
  - (* Assign *)
    cbn [compile].
    rewrite (gen_assign_spec x o e ms s H Hmod Hcl); cbn [regs mem]; split.
    + apply (models_update ms s x). exact Hmod.
    + reflexivity.
  - (* Swap *)
    cbn [wf_stmt] in Hwf; cbn [compile].
    rewrite (gen_swap_spec x y ms s Hwf Hmod Hcl); cbn [regs mem]; split;
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
  - (* Seq *)
    cbn [wf_stmt] in Hwf; destruct Hwf as [Hwf1 Hwf2]; cbn [compile].
    rewrite run_app.
    destruct (IHexec1 ms Hwf1 Hmod Hcl) as [Hm1 Hr1].
    assert (Hcl1 : clean_above scratch (run (compile s1) ms))
      by (intros r Hr; rewrite Hr1; now apply Hcl).
    destruct (IHexec2 (run (compile s1) ms) Hwf2 Hm1 Hcl1) as [Hm2 Hr2].
    split; [exact Hm2 | now rewrite Hr2, Hr1].
Qed.

(** ** Corollary: the compiled code is reversible on the machine

    Semantic preservation plus [PISA.run_invert_code] gives the machine-level
    counterpart of [Src.exec_rev]: running the compiled code and then its
    instruction-wise inverse restores the whole machine state. *)

Corollary compile_reversible : forall st s,
  run (invert_code (compile st)) (run (compile st) s) = s.
Proof. intros; apply run_invert_code, wf_compile. Qed.
