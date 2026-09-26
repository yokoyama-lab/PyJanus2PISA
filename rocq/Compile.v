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

    Since 2026-09 (docs/EXPR_LOWERING.md) every expression, comparisons and
    [&&]/[||] included, is compiled in the clean shape

<<
      re ; <e1 → rl> ; <e2 → rr> ; combine re rl rr ; <e2 → rr>⁻¹ ; <e1 → rl>⁻¹
>>

    with the operands uncomputed at once by running their code backwards,
    no [XOR r r] clears, and [ORX] / [ANDX] with Pendulum's 3-operand
    XOR-accumulating semantics (PISA.v).  Hence [ungen_expr e] is
    [invert_code (gen_expr e)] for *every* expression, every emitted
    instruction is well-formed ([wf_gen_expr], [wf_compile]), and
    [compile_reversible] holds for every program.

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

(** [+ - ^] as a register-register instruction ([_INPLACE]). *)
Definition op_instr (o : binop) (rd rs : reg) : instr :=
  match o with
  | OAdd => IAdd rd rs
  | OSub => ISub rd rs
  | _    => IXor rd rs
  end.

Definition aop_instr (o : aop) (rd rs : reg) : instr :=
  match o with
  | AAdd => IAdd rd rs
  | ASub => ISub rd rs
  | AXor => IXor rd rs
  end.

(** A constant into a zero register (`_gen_const`): [ADDI r k] for [k > 0],
    [SUBI r (-k)] for [k < 0], nothing for [k = 0].  In general it adds [n]. *)
Definition cst_code (n : Z) (rt : reg) : code :=
  if n =? 0 then []
  else if 0 <? n then [IAddi rt n] else [ISubi rt (- n)].

(** [rt op= k] for a compile-time constant [k] (`_inplace_imm`). *)
Definition imm_code (o : binop) (k : Z) (rt : reg) : code :=
  match o with
  | OXor => if k =? 0 then [] else [IXori rt k]
  | OSub => cst_code (- k) rt
  | _    => cst_code k rt
  end.

(** `CodeGen._const_value`: the value of an expression built from literals
    by [+ - ^] (the parser's unary minus [0 - e] included); [None] otherwise
    (comparisons and logical operators are not evaluated, as there). *)
Fixpoint const_value (e : expr) : option Z :=
  match e with
  | Cst n => Some n
  | Var _ => None
  | Bin o e1 e2 =>
      if arith_op o then
        match const_value e1, const_value e2 with
        | Some a, Some b => Some (denote o a b)
        | _, _ => None
        end
      else None
  end.

(** The constant folding at the top of `_gen_binop`: both operands are
    literals (every operator of [Src.expr] is folded there). *)
Definition fold_csts (o : binop) (e1 e2 : expr) : option Z :=
  match e1, e2 with
  | Cst a, Cst b => Some (denote o a b)
  | _, _ => None
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

(** *** Comparisons and logical operators

    `_as_flag` keeps an expression that is already 0/1 — a comparison, a
    logical operator, or the constant 0 or 1 — and turns anything else into
    [e != 0].  [is_flag_expr] is the first test. *)
Definition is_flag_expr (e : expr) : bool :=
  match e with
  | Cst k => (k =? 0) || (k =? 1)
  | Var _ => false
  | Bin o _ _ => flag_op o
  end.

(** `_is_nonzero_test`: the shape [e != 0] (literal [0] on the right).
    `codegen.py` compiles it by `_gen_nonzero`, whatever produced it
    (`_as_flag`, `_logical_operands`, or the programmer). *)
Definition is_nz_test (o : binop) (e2 : expr) : bool :=
  match o, e2 with
  | ONe, Cst k => k =? 0
  | _, _ => false
  end.

Definition is_logic (o : binop) : bool :=
  match o with OAnd | OOr => true | _ => false end.

(** `_gen_nonzero`: [rf ^= (e != 0)] with [e]'s value in [S rf] while it is
    needed, [e] uncomputed at once.  [g] is [e]'s code as a function of its
    target register. *)
Definition nz_code (g : reg -> code) (r : reg) : code :=
  g (S r) ++ [ISltx r (S r) 0%nat; ISltx r 0%nat (S r)] ++ invert_code (g (S r)).

(** The operand [_as_flag(e)] of [&&] / [||] (`_logical_operands`), as code:
    [e] itself if it is already 0/1; a literal [k] becomes the literal
    [k != 0] (folded by `_gen_binop`); anything else [e != 0]. *)
Definition flag_gen (e : expr) (g : reg -> code) : reg -> code :=
  if is_flag_expr e then g
  else match e with
       | Cst k => cst_code (b2z (negb (k =? 0)))
       | _ => nz_code g
       end.

(** `_COMBINE`: [re ^= f(rl, rr)] into a zero register [re]. *)
Definition comb_code (o : binop) (re rl rr : reg) : code :=
  match o with
  | OLt  => [ISltx re rl rr]
  | OGt  => [ISltx re rr rl]
  | OLe  => [ISltx re rr rl; IXori re 1]
  | OGe  => [ISltx re rl rr; IXori re 1]
  | ONe  => [ISltx re rl rr; ISltx re rr rl]
  | OEq  => [ISltx re rl rr; ISltx re rr rl; IXori re 1]
  | OAnd => [IAndx re rl rr]
  | OOr  => [IOrx re rl rr]
  | _    => []          (* [+ - ^] are compiled in place *)
  end.

(** `_gen_combine`: result register first, operands above it, operands
    uncomputed right after the combine, right before left. *)
Definition comb_block (o : binop) (g1 g2 : reg -> code) (rt : reg) : code :=
  g1 (S rt) ++ g2 (S (S rt)) ++ comb_code o rt (S rt) (S (S rt))
  ++ invert_code (g2 (S (S rt))) ++ invert_code (g1 (S rt)).

(** [gen_expr e rt] evaluates [e] into [rt], which must be 0, using only
    registers above [rt] as temporaries, and leaves every other register as
    it found it.  The case analysis is `_gen_binop`'s, in its order:
    constant folding, `e != 0` ([nz_code]), [+ - ^] in place (immediate if
    the right operand is a compile-time constant), and the combine of the
    other operators ([&&]/[||] with `_logical_operands`).

    Registers: `codegen.py`'s allocator picks the lowest free register, so
    e.g. the address register of a variable load sits between the target and
    the value register there; here the target is [rt] and the temporaries
    are [S rt], [S (S rt)], ….  The instructions per operator are the same;
    the register numbers are not. *)
Fixpoint gen_expr (e : expr) (rt : reg) {struct e} : code :=
  match e with
  | Cst n => cst_code n rt
  | Var x => gen_var x rt
  | Bin o e1 e2 =>
      match fold_csts o e1 e2 with
      | Some k => cst_code k rt
      | None =>
          if is_nz_test o e2 then nz_code (gen_expr e1) rt
          else if arith_op o then
            gen_expr e1 rt
            ++ match const_value e2 with
               | Some k => imm_code o k rt
               | None => gen_expr e2 (S rt) ++ [op_instr o rt (S rt)]
                         ++ invert_code (gen_expr e2 (S rt))
               end
          else if is_logic o then
            comb_block o (flag_gen e1 (gen_expr e1)) (flag_gen e2 (gen_expr e2)) rt
          else comb_block o (gen_expr e1) (gen_expr e2) rt
      end
  end.

(** `_uneval`: the expression's code run backwards clears [rt] again. *)
Definition ungen_expr (e : expr) (rt : reg) : code := invert_code (gen_expr e rt).

(** [x op= e]: evaluate [e] into [scratch], apply it to the memory cell of
    [x] through an exchange, then unevaluate [e]. *)
Definition gen_assign (x : var) (o : aop) (e : expr) : code :=
  gen_expr e scratch
  ++ [ IAddi (S scratch) (Z.of_nat x)
     ; IExch (S (S scratch)) (S scratch)
     ; aop_instr o (S (S scratch)) scratch
     ; IExch (S (S scratch)) (S scratch)
     ; ISubi (S scratch) (Z.of_nat x) ]
  ++ ungen_expr e scratch.

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

(** ** Inversion lemmas *)

Lemma invert_instr_invol : forall i, invert_instr (invert_instr i) = i.
Proof. destruct i; reflexivity. Qed.

Lemma invert_code_app : forall c1 c2,
  invert_code (c1 ++ c2) = invert_code c2 ++ invert_code c1.
Proof. intros; unfold invert_code; now rewrite map_app, rev_app_distr. Qed.

Lemma invert_code_invol : forall c, invert_code (invert_code c) = c.
Proof.
  intro c; unfold invert_code; rewrite map_rev, rev_involutive, map_map.
  rewrite (map_ext _ (fun i => i)) by apply invert_instr_invol. apply map_id.
Qed.

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

(** ** Tactics for state equations *)

(** Reduce a [run] of a literal block to a nest of [rupd]/[mupd] with all
    register and memory lookups resolved. *)
Ltac simp_state :=
  simpl;
  repeat first
    [ rewrite rupd_same
    | rewrite mupd_same
    | rewrite rupd_other by lia
    | rewrite mupd_other by lia ].

(** Prove two register files equal pointwise, by cases on the register. *)
Ltac regs_ext :=
  apply functional_extensionality; intro r; unfold rupd;
  repeat match goal with
         | |- context [Nat.eqb ?a ?b] =>
             destruct (Nat.eqb_spec a b); try (exfalso; lia)
         end;
  subst.

(** Finish a goal about 0/1 values by cases on the comparisons in it. *)
Ltac zcase :=
  unfold b2z in *; cbn [denote negb andb orb] in *;
  repeat match goal with
         | |- context [Z.ltb ?a ?b] => destruct (Z.ltb_spec a b)
         | |- context [Z.leb ?a ?b] => destruct (Z.leb_spec a b)
         | |- context [Z.eqb ?a ?b] => destruct (Z.eqb_spec a b)
         end;
  cbn; try lia; try reflexivity.

(** ** The building blocks, one lemma each *)

(** [gok g v M]: at every target [r <> 0], [g r] is well-formed, and from
    any register file with [r0 = 0] that is clean from [r] up it puts [v]
    into [r] and changes nothing else. *)
Definition gok (g : reg -> code) (v : Z) (M : addr -> Z) : Prop :=
  forall r, r <> 0%nat ->
    wf_code (g r) /\
    forall R, R 0%nat = 0 -> clean_above r (mkState R M) ->
      run (g r) (mkState R M) = mkState (rupd r v R) M.

(** The clean shape: running the code backwards clears the target again. *)
Lemma gok_uncompute : forall g v M r R, gok g v M -> r <> 0%nat ->
  R 0%nat = 0 -> clean_above r (mkState R M) ->
  run (invert_code (g r)) (mkState (rupd r v R) M) = mkState R M.
Proof.
  intros g v M r R Hg Hr H0 Hcl. destruct (Hg r Hr) as [W G].
  rewrite <- (G R H0 Hcl). now apply run_invert_code.
Qed.

Lemma clean_S : forall r s, clean_above r s -> clean_above (S r) s.
Proof. intros r s H k Hk; apply H; lia. Qed.

Lemma clean_rupd : forall r k a R M, (k < r)%nat ->
  clean_above r (mkState R M) -> clean_above r (mkState (rupd k a R) M).
Proof. intros r k a R M Hk H j Hj; cbn [regs]; rewrite rupd_other by lia; apply H, Hj. Qed.

Lemma zero_rupd : forall k a R, R 0%nat = 0 -> k <> 0%nat -> rupd k a R 0%nat = 0.
Proof. intros; rewrite rupd_other by lia; assumption. Qed.

Lemma wf_cst_code : forall n r, wf_code (cst_code n r).
Proof. intros; unfold cst_code; destruct (n =? 0), (0 <? n); wf_list. Qed.

Lemma cst_code_run : forall n r R M,
  run (cst_code n r) (mkState R M) = mkState (rupd r (R r + n) R) M.
Proof.
  intros n r R M; unfold cst_code.
  destruct (Z.eqb_spec n 0) as [-> | Hn].
  - cbn. rewrite Z.add_0_r, rupd_id. reflexivity.
  - destruct (Z.ltb_spec 0 n); cbn [run fold_left step regs mem]; f_equal; f_equal; lia.
Qed.

Lemma cst_gok : forall n M, gok (cst_code n) n M.
Proof.
  intros n M r Hr; split; [apply wf_cst_code |].
  intros R H0 Hcl. rewrite cst_code_run.
  assert (HR : R r = 0) by (apply Hcl; lia). now rewrite HR, Z.add_0_l.
Qed.

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

Lemma var_gok : forall σ M x, (forall y : var, M (Z.of_nat y) = σ y) ->
  gok (gen_var x) (σ x) M.
Proof.
  intros σ M x Hmod r Hr; split; [unfold gen_var; wf_list |].
  intros R H0 Hcl. exact (gen_var_spec x r (mkState R M) σ Hmod Hcl).
Qed.

Lemma sltx_pair : forall v,
  Z.lxor (if v <? 0 then 1 else 0) (if 0 <? v then 1 else 0) = b2z (negb (v =? 0)).
Proof. intro v; zcase. Qed.

Lemma nz_gok : forall g v M, gok g v M -> gok (nz_code g) (b2z (negb (v =? 0))) M.
Proof.
  intros g v M Hg r Hr. destruct (Hg (S r) ltac:(lia)) as [Wg Gg]. split.
  - unfold nz_code. apply wf_code_app; [exact Wg |].
    apply wf_code_app; [wf_list | now apply wf_invert_code].
  - intros R H0 Hcl.
    assert (HR : R r = 0) by (apply Hcl; lia).
    unfold nz_code. rewrite run_app, (Gg R H0 (clean_S _ _ Hcl)), run_app.
    assert (E : run [ISltx r (S r) 0%nat; ISltx r 0%nat (S r)] (mkState (rupd (S r) v R) M)
                = mkState (rupd (S r) v (rupd r (b2z (negb (v =? 0))) R)) M).
    { rewrite !run_cons, run_nil. cbn [step regs mem].
      repeat first [ rewrite rupd_same | rewrite rupd_other by lia ].
      rewrite H0, HR, rupd_shadow, Z.lxor_0_l, sltx_pair.
      rewrite (rupd_comm r (S r)) by lia. reflexivity. }
    rewrite E. apply (gok_uncompute g v M (S r)); try assumption; try lia.
    + now apply zero_rupd.
    + apply clean_rupd; [lia | now apply clean_S].
Qed.

Lemma is_flag_expr_01 : forall σ e, is_flag_expr e = true ->
  eval σ e = 0 \/ eval σ e = 1.
Proof.
  intros σ [k | x | o e1 e2] H; cbn [is_flag_expr eval] in *.
  - apply orb_true_iff in H as [H | H]; apply Z.eqb_eq in H; auto.
  - discriminate.
  - now apply denote_flag.
Qed.

Lemma flag_gen_gok : forall e g v M, gok g v M ->
  (is_flag_expr e = true -> v = 0 \/ v = 1) -> (forall k, e = Cst k -> v = k) ->
  gok (flag_gen e g) (b2z (negb (v =? 0))) M.
Proof.
  intros e g v M Hg H01 Hk. unfold flag_gen.
  destruct (is_flag_expr e) eqn:E.
  - destruct (H01 eq_refl) as [-> | ->]; exact Hg.
  - destruct e as [k | x | o e1 e2].
    + rewrite (Hk k eq_refl). apply cst_gok.
    + now apply nz_gok.
    + now apply nz_gok.
Qed.

(** The value [comb_code] XORs into its (zero) target. *)
Definition cval (o : binop) (a b : Z) : Z :=
  match o with
  | OAnd => Z.land a b
  | OOr  => Z.lor a b
  | _    => denote o a b
  end.

Lemma wf_comb_code : forall o re rl rr, re <> rl -> re <> rr ->
  wf_code (comb_code o re rl rr).
Proof. intros o re rl rr H1 H2; destruct o; cbn [comb_code]; wf_list. Qed.

Lemma comb_code_run : forall o re rl rr R M, arith_op o = false ->
  re <> rl -> re <> rr -> R re = 0 ->
  run (comb_code o re rl rr) (mkState R M) = mkState (rupd re (cval o (R rl) (R rr)) R) M.
Proof.
  intros o re rl rr R M Ho H1 H2 Hre.
  destruct o; try discriminate; cbn [comb_code cval];
    rewrite ?run_cons, ?run_nil; cbn [step regs mem];
    repeat first [ rewrite rupd_same | rewrite rupd_other by lia ];
    rewrite ?rupd_shadow, Hre, ?Z.lxor_0_l; f_equal; f_equal; zcase.
Qed.

Lemma comb_gok : forall o g1 g2 v1 v2 M, arith_op o = false ->
  gok g1 v1 M -> gok g2 v2 M -> gok (comb_block o g1 g2) (cval o v1 v2) M.
Proof.
  intros o g1 g2 v1 v2 M Ho H1 H2 r Hr.
  destruct (H1 (S r) ltac:(lia)) as [W1 G1].
  destruct (H2 (S (S r)) ltac:(lia)) as [W2 G2].
  split.
  - unfold comb_block.
    repeat apply wf_code_app; try assumption; try (now apply wf_invert_code).
    apply wf_comb_code; lia.
  - intros R H0 Hcl. unfold comb_block.
    assert (HR : R r = 0) by (apply Hcl; lia).
    set (c := cval o v1 v2).
    rewrite run_app, (G1 R H0 (clean_S _ _ Hcl)).
    rewrite run_app, G2.
    2: { now apply zero_rupd. }
    2: { apply clean_rupd; [lia | now apply clean_S, clean_S]. }
    rewrite run_app, comb_code_run by
      (first [exact Ho | lia | rewrite !rupd_other by lia; exact HR]).
    rewrite (rupd_same (S (S r))), rupd_other, rupd_same by lia.
    fold c.
    rewrite (rupd_comm r (S (S r))), (rupd_comm r (S r)) by lia.
    rewrite run_app, (gok_uncompute g2 v2 M (S (S r))); try assumption; try lia.
    + rewrite (gok_uncompute g1 v1 M (S r)); try assumption; try lia.
      * reflexivity.
      * now apply zero_rupd.
      * apply clean_rupd; [lia | now apply clean_S].
    + apply zero_rupd; [now apply zero_rupd | lia].
    + apply clean_rupd; [lia |]. apply clean_rupd; [lia |].
      now apply clean_S, clean_S.
Qed.

Lemma inplace_reg_gok : forall o g1 g2 v1 v2 M, arith_op o = true ->
  gok g1 v1 M -> gok g2 v2 M ->
  gok (fun r => g1 r ++ g2 (S r) ++ [op_instr o r (S r)] ++ invert_code (g2 (S r)))
      (denote o v1 v2) M.
Proof.
  intros o g1 g2 v1 v2 M Ho H1 H2 r Hr.
  destruct (H1 r Hr) as [W1 G1].
  destruct (H2 (S r) ltac:(lia)) as [W2 G2].
  split.
  - repeat apply wf_code_app; try assumption; try (now apply wf_invert_code).
    apply Forall_cons; [destruct o; cbn; lia | apply Forall_nil].
  - intros R H0 Hcl.
    rewrite run_app, (G1 R H0 Hcl), run_app, G2.
    2: { now apply zero_rupd. }
    2: { apply clean_rupd; [lia | now apply clean_S]. }
    rewrite run_app, run_one.
    assert (Hop : step (op_instr o r (S r)) (mkState (rupd (S r) v2 (rupd r v1 R)) M)
                  = mkState (rupd (S r) v2 (rupd r (denote o v1 v2) R)) M).
    { destruct o; try discriminate; simp_state; f_equal;
        rewrite (rupd_comm r (S r)) by lia; now rewrite rupd_shadow. }
    rewrite Hop, (gok_uncompute g2 v2 M (S r)); try assumption; try lia.
    + reflexivity.
    + now apply zero_rupd.
    + apply clean_rupd; [lia | now apply clean_S].
Qed.

Lemma imm_code_run : forall o k r R M, arith_op o = true ->
  run (imm_code o k r) (mkState R M) = mkState (rupd r (denote o (R r) k) R) M.
Proof.
  intros o k r R M Ho; destruct o; try discriminate; cbn [imm_code denote].
  - now rewrite cst_code_run.
  - rewrite cst_code_run. replace (R r - k) with (R r + - k) by lia. reflexivity.
  - destruct (Z.eqb_spec k 0) as [-> | Hk].
    + cbn. now rewrite Z.lxor_0_r, rupd_id.
    + reflexivity.
Qed.

Lemma inplace_imm_gok : forall o k g1 v1 M, arith_op o = true ->
  gok g1 v1 M -> gok (fun r => g1 r ++ imm_code o k r) (denote o v1 k) M.
Proof.
  intros o k g1 v1 M Ho H1 r Hr. destruct (H1 r Hr) as [W1 G1]. split.
  - apply wf_code_app; [exact W1 |].
    destruct o; try discriminate; cbn [imm_code];
      try apply wf_cst_code; destruct (k =? 0); wf_list.
  - intros R H0 Hcl.
    rewrite run_app, (G1 R H0 Hcl), imm_code_run by exact Ho.
    now rewrite rupd_same, rupd_shadow.
Qed.

(** ** Expression compilation is correct, clean and well-formed *)

Lemma const_value_eval : forall σ e k, const_value e = Some k -> eval σ e = k.
Proof.
  intros σ e; induction e as [n | x | o e1 IH1 e2 IH2]; intros k H;
    cbn [const_value eval] in *.
  - congruence.
  - discriminate.
  - destruct (arith_op o); [| discriminate].
    destruct (const_value e1) as [a |]; [| discriminate].
    destruct (const_value e2) as [b |]; [| discriminate].
    injection H as <-. now rewrite (IH1 a), (IH2 b).
Qed.

Lemma fold_csts_eval : forall σ o e1 e2 k, fold_csts o e1 e2 = Some k ->
  denote o (eval σ e1) (eval σ e2) = k.
Proof.
  intros σ o [a | | ] [b | | ] k H; cbn in H; try discriminate.
  injection H as <-. reflexivity.
Qed.

Lemma cval_logic : forall o a b, is_logic o = true ->
  cval o (b2z (negb (a =? 0))) (b2z (negb (b =? 0))) = denote o a b.
Proof.
  intros o a b H; destruct o; try discriminate; cbn;
    destruct (a =? 0), (b =? 0); reflexivity.
Qed.

Lemma cval_cmp : forall o a b, is_logic o = false -> cval o a b = denote o a b.
Proof. intros o a b H; destruct o; try discriminate; reflexivity. Qed.

Lemma is_nz_test_eval : forall o e2 σ v, is_nz_test o e2 = true ->
  denote o v (eval σ e2) = b2z (negb (v =? 0)).
Proof.
  intros o e2 σ v H; destruct o; try discriminate.
  destruct e2 as [k | |]; try discriminate. cbn in H. apply Z.eqb_eq in H. subst k.
  reflexivity.
Qed.

Theorem gen_gok : forall σ M, (forall x : var, M (Z.of_nat x) = σ x) ->
  forall e, gok (gen_expr e) (eval σ e) M.
Proof.
  intros σ M Hmod e.
  induction e as [n | x | o e1 IH1 e2 IH2].
  - apply cst_gok.
  - now apply var_gok.
  - intros r Hr. cbn [gen_expr eval].
    destruct (fold_csts o e1 e2) as [k |] eqn:Ef.
    { rewrite (fold_csts_eval σ o e1 e2 k Ef). exact (cst_gok k M r Hr). }
    destruct (is_nz_test o e2) eqn:Enz.
    { rewrite (is_nz_test_eval o e2 σ _ Enz). exact (nz_gok _ _ M IH1 r Hr). }
    destruct (arith_op o) eqn:Ea.
    { destruct (const_value e2) as [k |] eqn:Ec.
      - rewrite (const_value_eval σ e2 k Ec).
        exact (inplace_imm_gok o k (gen_expr e1) _ M Ea IH1 r Hr).
      - exact (inplace_reg_gok o (gen_expr e1) (gen_expr e2) _ _ M Ea IH1 IH2 r Hr). }
    destruct (is_logic o) eqn:El.
    + rewrite <- (cval_logic o _ _ El).
      apply (comb_gok o _ _ _ _ M Ea); [| | exact Hr].
      * apply flag_gen_gok; [exact IH1 | apply is_flag_expr_01 |].
        intros k ->; reflexivity.
      * apply flag_gen_gok; [exact IH2 | apply is_flag_expr_01 |].
        intros k ->; reflexivity.
    + rewrite <- (cval_cmp o _ _ El).
      exact (comb_gok o _ _ _ _ M Ea IH1 IH2 r Hr).
Qed.

(** Every instruction [gen_expr] emits is well-formed (`pisa.is_wf`), for
    every expression and every target register other than r0. *)
Theorem wf_gen_expr : forall e rt, rt <> 0%nat -> wf_code (gen_expr e rt).
Proof.
  intros e rt Hrt.
  exact (proj1 (gen_gok (fun _ => 0) (fun _ => 0) (fun _ => eq_refl) e rt Hrt)).
Qed.

(** [g] / [u] evaluate a value [v] into register [r] and clear it again,
    touching nothing else, from any register file with [r0 = 0] that is
    clean from [r] up, over the memory [M]. *)
Definition expr_ok (g u : reg -> code) (r : reg) (v : Z) (M : addr -> Z) : Prop :=
  forall R, R 0%nat = 0 -> clean_above r (mkState R M) ->
    run (g r) (mkState R M) = mkState (rupd r v R) M /\
    run (u r) (mkState (rupd r v R) M) = mkState R M.

Theorem gen_ungen_spec : forall σ M, (forall x : var, M (Z.of_nat x) = σ x) ->
  forall e rt, rt <> 0%nat -> expr_ok (gen_expr e) (ungen_expr e) rt (eval σ e) M.
Proof.
  intros σ M Hmod e rt Hrt R H0 Hcl. unfold ungen_expr.
  pose proof (gen_gok σ M Hmod e) as G.
  split; [exact (proj2 (G rt Hrt) R H0 Hcl) |].
  exact (gok_uncompute _ _ M rt R G Hrt H0 Hcl).
Qed.

Theorem gen_expr_spec : forall e rt s σ,
  models s σ -> clean_above rt s -> regs s 0%nat = 0 -> rt <> 0%nat ->
  run (gen_expr e rt) s = mkState (rupd rt (eval σ e) (regs s)) (mem s).
Proof.
  intros e rt [R M] σ Hmod Hcl H0 Hrt.
  exact (proj1 (gen_ungen_spec σ M Hmod e rt Hrt R H0 Hcl)).
Qed.

Theorem ungen_expr_spec : forall e rt R M σ,
  models (mkState R M) σ -> clean_above rt (mkState R M) -> R 0%nat = 0 -> rt <> 0%nat ->
  run (ungen_expr e rt) (mkState (rupd rt (eval σ e) R) M) = mkState R M.
Proof.
  intros e rt R M σ Hmod Hcl H0 Hrt.
  exact (proj2 (gen_ungen_spec σ M Hmod e rt Hrt R H0 Hcl)).
Qed.

(** [nz_code] as a stand-alone block over any correct operand code — used by
    the test normalisation of CompileIf.v. *)
Lemma nz_code_ok : forall σ M e r, (forall x : var, M (Z.of_nat x) = σ x) ->
  r <> 0%nat -> forall R, R 0%nat = 0 -> clean_above r (mkState R M) ->
  run (nz_code (gen_expr e) r) (mkState R M)
  = mkState (rupd r (b2z (negb (eval σ e =? 0))) R) M.
Proof.
  intros σ M e r Hmod Hr R H0 Hcl.
  exact (proj2 (nz_gok _ _ M (gen_gok σ M Hmod e) r Hr) R H0 Hcl).
Qed.

(** ** Statement compilation is well-formed *)

Lemma wf_compile : forall st, wf_code (compile st).
Proof.
  induction st as [| x o e | x y | s1 IH1 s2 IH2]; simpl.
  - apply Forall_nil.
  - unfold gen_assign, ungen_expr.
    apply wf_code_app; [apply wf_gen_expr; unfold scratch; lia |].
    repeat (apply Forall_cons; [destruct o; simpl; first [exact I | lia] |]).
    apply wf_invert_code, wf_gen_expr; unfold scratch; lia.
  - unfold gen_swap; wf_list.
  - apply wf_code_app; auto.
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
  models s σ -> clean_above scratch s -> regs s 0%nat = 0 ->
  run (gen_assign x o e) s
  = mkState (regs s)
            (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) (mem s)).
Proof.
  intros x o e [R M] σ Hocc Hmod Hcl H0; cbn [regs mem] in *.
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
  rewrite run_app, (gen_expr_spec e scratch (mkState R M) σ Hmod Hcl H0) by (unfold scratch; lia);
    cbn [regs mem].
  rewrite run_app, Hblock.
  (* unevaluate [e]; sound because [x] does not occur in [e], so the value of
     [e] is the same before and after the store *)
  assert (Hmod' : models (mkState R (mupd (Z.of_nat x) (adenote o (σ x) (eval σ e)) M))
                         (update σ x (adenote o (σ x) (eval σ e))))
    by (apply (models_update (mkState R M) σ x (adenote o (σ x) (eval σ e))); exact Hmod).
  assert (Hev : eval (update σ x (adenote o (σ x) (eval σ e))) e = eval σ e)
    by (now apply eval_update_notin).
  rewrite <- Hev at 1.
  apply (ungen_expr_spec e scratch R _ _ Hmod' Hcl H0). unfold scratch; lia.
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

(** ** Main theorem: semantic preservation with a clean register file

    The premise [regs ms 0 = 0] is needed because register [r0], hard-wired
    to 0 in `pisa_interp.py`, is an operand of the two [SLTX] of the
    [e != 0] normalisation.  Every control-flow theorem has it too. *)

Theorem compile_spec : forall st σ σ' ms,
  exec st σ σ' -> wf_stmt st ->
  models ms σ -> clean_above scratch ms -> regs ms 0%nat = 0 ->
  models (run (compile st) ms) σ' /\ regs (run (compile st) ms) = regs ms.
Proof.
  intros st σ σ' ms H; revert ms; induction H; intros ms Hwf Hmod Hcl Hr0.
  - (* Skip *) split; [exact Hmod | reflexivity].
  - (* Assign *)
    cbn [compile].
    rewrite (gen_assign_spec x o e ms s H Hmod Hcl Hr0); cbn [regs mem]; split.
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
    destruct (IHexec1 ms Hwf1 Hmod Hcl Hr0) as [Hm1 Hr1].
    assert (Hcl1 : clean_above scratch (run (compile s1) ms))
      by (intros r Hr; rewrite Hr1; now apply Hcl).
    destruct (IHexec2 (run (compile s1) ms) Hwf2 Hm1 Hcl1 ltac:(now rewrite Hr1))
      as [Hm2 Hr2].
    split; [exact Hm2 | now rewrite Hr2, Hr1].
Qed.

(** ** Corollary: the compiled code is reversible, for every program

    Every instruction the compiler emits is well-formed ([wf_compile]), so
    [PISA.run_invert_code] applies: running the compiled code and then its
    instruction-wise inverse restores the whole machine state — from ANY
    state, not only from one that models a store.  This is the machine-level
    counterpart of [Src.exec_rev].  (Until 2026-09 this held for [+ - ^]
    programs only; comparisons brought `XOR r r` clears and the clearing
    ORX / ANDX along, see PISA.v.) *)

Corollary compile_reversible : forall st s,
  run (invert_code (compile st)) (run (compile st) s) = s.
Proof. intros st s; apply run_invert_code, wf_compile. Qed.
