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

    Comparisons and [&&] / [||] (milestone 3) are compiled as `_gen_binop`
    and `_gen_uneval_binop` compile them — [SLTX], [XORI], and the
    compiler pseudo-instructions ORX / ANDX, which zero a source register
    and have no inverse ([PISA.orx_not_injective]).  Their unevaluation is
    therefore not "the code run backwards" but `_gen_uneval_binop`'s
    recomputation ([ungen_expr]), and the operand registers are zeroed by
    [XOR r r] as `_clear_garbage` does.  The state equations still hold
    ([gen_ungen_spec]); instruction-level reversibility
    ([compile_reversible]) holds for [+ - ^] programs only
    ([compile_not_reversible]).

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

(** Only reached for the arithmetic operators ([arith_op]); the others are
    compiled by [cmp_fwd] and the logical cases of [gen_expr]. *)
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

(** Load variable [x] into [rt], using [S rt] as the address register and
    [S (S rt)] as the exchange buffer.  The value is copied out with XOR and
    the cell is put straight back, so memory is unchanged. *)
Definition gen_var (x : var) (rt : reg) : code :=
  [ IAddi (S rt) (Z.of_nat x)
  ; IExch (S (S rt)) (S rt)
  ; IXor  rt (S (S rt))
  ; IExch (S (S rt)) (S rt)
  ; ISubi (S rt) (Z.of_nat x) ].

(** *** Comparisons and logical operators (`_gen_binop` / `_gen_uneval_binop`)

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

(** `_nonzero_into(r, e)`: evaluate [e] into [S r], [r ^= (v < 0)],
    [r ^= (0 < v)] (the bits are exclusive, so [r ^= (v != 0)]),
    unevaluate [e].  Self-inverse, so it is also its own uneval.  [g] / [u]
    are [e]'s evaluation and unevaluation code as functions of the target. *)
Definition nz_code (g u : reg -> code) (r : reg) : code :=
  g (S r) ++ ISltx r (S r) 0%nat :: ISltx r 0%nat (S r) :: u (S r).

(** The operand [_as_flag(e)] of [&&] / [||] (`_logical_operands`): [e]
    itself if [p = is_flag_expr e], else [e != 0]. *)
Definition flag_code (p : bool) (g u : reg -> code) (r : reg) : code :=
  if p then g r else nz_code g u r.

Definition unflag_code (p : bool) (g u : reg -> code) (r : reg) : code :=
  if p then u r else nz_code g u r.

(** The instructions `_gen_binop` emits for a comparison, result in [rd]
    (initially 0), operands in [r1] / [r2], temporary [t] (initially 0,
    zeroed by the ORX).  [_gen_uneval_binop] emits the same block into a
    fresh [sub_re] (and [sub_rt]). *)
Definition cmp_fwd (o : binop) (rd r1 r2 t : reg) : code :=
  match o with
  | OEq => [ISltx rd r1 r2; ISltx t r2 r1; IOrx rd t; IXori rd 1]
  | ONe => [ISltx rd r1 r2; ISltx t r2 r1; IOrx rd t]
  | OLt => [ISltx rd r1 r2]
  | OGt => [ISltx rd r2 r1]
  | OLe => [ISltx rd r2 r1; IXori rd 1]
  | OGe => [ISltx rd r1 r2; IXori rd 1]
  | _   => []
  end.

(** [gen_expr e rt] evaluates [e] into [rt]; [ungen_expr e rt] clears [rt]
    again (`_gen_uneval_expr`).  Both leave every other register as they
    found it.

    - Arithmetic ([+ - ^]): the clean translation of Compile.v's first
      version, unchanged — the right operand is evaluated into [S rt],
      combined, and unevaluated.  On arithmetic expressions
      [ungen_expr e = invert_code (gen_expr e)] ([ungen_arith]), so the code
      is literally the old one.
    - [e != 0]: `_gen_nonzero` ([nz_code]).
    - Comparisons: left into [S rt], right into [S (S rt)], `_gen_binop`'s
      block ([cmp_fwd], result in [rt]), then the two operand registers are
      zeroed by [XOR r r] — `codegen.py` marks them garbage and
      `_clear_garbage` emits exactly these [XOR r r] at the end of the
      statement; here they come right away.  The uneval is
      `_gen_uneval_binop`'s: re-evaluate both operands, recompute the
      comparison into [sub_re], [XOR result sub_re], [XOR sub_re sub_re],
      unevaluate the right, then the left operand.
    - [&&] / [||]: operands through `_logical_operands` ([flag_code]);
      [ANDX rt rl rr] (which zeroes [rl]) and a garbage clear of [rr], resp.
      [ORX rt rl ; ORX rt rr] (which zero both); the uneval copies the
      operands first (`rl_copy`, `rr_copy`), exactly as `_gen_uneval_binop`.

    Registers: `codegen.py` allocates the result register *after* the
    operands; here it is the target [rt] and the operands live above it.
    The values computed are the same (see [gen_ungen_spec]); the register
    numbers are not. *)
Fixpoint gen_expr (e : expr) (rt : reg) {struct e} : code :=
  match e with
  | Cst n => [IAddi rt n]
  | Var x => gen_var x rt
  | Bin o e1 e2 =>
      if arith_op o then
        gen_expr e1 rt ++ gen_expr e2 (S rt) ++ [op_instr o rt (S rt)]
        ++ ungen_expr e2 (S rt)
      else if is_nz_test o e2 then
        nz_code (gen_expr e1) (ungen_expr e1) rt
      else
        match o with
        | OAnd =>
            flag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1) (S rt)
            ++ flag_code (is_flag_expr e2) (gen_expr e2) (ungen_expr e2) (S (S rt))
            ++ [IAndx rt (S rt) (S (S rt)); IXor (S (S rt)) (S (S rt))]
        | OOr =>
            flag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1) (S rt)
            ++ flag_code (is_flag_expr e2) (gen_expr e2) (ungen_expr e2) (S (S rt))
            ++ [IOrx rt (S rt); IOrx rt (S (S rt))]
        | _ =>
            gen_expr e1 (S rt) ++ gen_expr e2 (S (S rt))
            ++ cmp_fwd o rt (S rt) (S (S rt)) (S (S (S rt)))
            ++ [IXor (S rt) (S rt); IXor (S (S rt)) (S (S rt))]
        end
  end
with ungen_expr (e : expr) (rt : reg) {struct e} : code :=
  match e with
  | Cst n => [ISubi rt n]
  | Var x => invert_code (gen_var x rt)
  | Bin o e1 e2 =>
      if arith_op o then
        gen_expr e2 (S rt) ++ [invert_instr (op_instr o rt (S rt))]
        ++ ungen_expr e2 (S rt) ++ ungen_expr e1 rt
      else if is_nz_test o e2 then
        nz_code (gen_expr e1) (ungen_expr e1) rt
      else
        match o with
        | OAnd =>
            flag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1) (S rt)
            ++ flag_code (is_flag_expr e2) (gen_expr e2) (ungen_expr e2) (S (S rt))
            ++ [ IXor (S (S (S rt))) (S rt)
               ; IAndx (S (S (S (S rt)))) (S (S (S rt))) (S (S rt))
               ; IXor rt (S (S (S (S rt))))
               ; IXor (S (S (S (S rt)))) (S (S (S (S rt)))) ]
            ++ unflag_code (is_flag_expr e2) (gen_expr e2) (ungen_expr e2) (S (S rt))
            ++ unflag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1) (S rt)
        | OOr =>
            flag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1) (S rt)
            ++ flag_code (is_flag_expr e2) (gen_expr e2) (ungen_expr e2) (S (S rt))
            ++ [ IXor (S (S (S rt))) (S rt)
               ; IOrx (S (S (S (S (S rt))))) (S (S (S rt)))
               ; IXor (S (S (S (S rt)))) (S (S rt))
               ; IOrx (S (S (S (S (S rt))))) (S (S (S (S rt))))
               ; IXor rt (S (S (S (S (S rt)))))
               ; IXor (S (S (S (S (S rt))))) (S (S (S (S (S rt))))) ]
            ++ unflag_code (is_flag_expr e2) (gen_expr e2) (ungen_expr e2) (S (S rt))
            ++ unflag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1) (S rt)
        | _ =>
            gen_expr e1 (S rt) ++ gen_expr e2 (S (S rt))
            ++ cmp_fwd o (S (S (S rt))) (S rt) (S (S rt)) (S (S (S (S rt))))
            ++ [IXor rt (S (S (S rt))); IXor (S (S (S rt))) (S (S (S rt)))]
            ++ ungen_expr e2 (S (S rt)) ++ ungen_expr e1 (S rt)
        end
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

(** ** On the original fragment the code is the old code, and well-formed

    For arithmetic expressions [ungen_expr] is the instruction-wise inverse
    of [gen_expr] — the clean translation of the first version of this
    file — and every instruction is well-formed, so [PISA.run_invert_code]
    applies.  With comparisons this is no longer so: the operand clears
    [XOR r r] and ORX / ANDX are not invertible ([PISA.orx_not_injective]),
    and [compile_not_reversible] below shows that running the
    instruction-wise inverse of such code does not restore the machine. *)

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

Lemma ungen_arith : forall e rt, arith_expr e = true ->
  ungen_expr e rt = invert_code (gen_expr e rt).
Proof.
  induction e as [n | x | o e1 IH1 e2 IH2]; intros rt H; try reflexivity.
  cbn [arith_expr] in H. apply andb_prop in H as [H H2]. apply andb_prop in H as [Ho H1].
  cbn [gen_expr ungen_expr]. rewrite Ho.
  rewrite !invert_code_app, (IH2 (S rt) H2), invert_code_invol, (IH1 rt H1).
  rewrite <- !app_assoc. reflexivity.
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

Lemma wf_gen_expr : forall e rt, arith_expr e = true -> wf_code (gen_expr e rt).
Proof.
  induction e as [n | x | o e1 IH1 e2 IH2]; intros rt H; simpl.
  - wf_list.
  - unfold gen_var; wf_list.
  - cbn [arith_expr] in H. apply andb_prop in H as [H H2]. apply andb_prop in H as [Ho H1].
    rewrite Ho, (ungen_arith e2 (S rt) H2).
    apply wf_code_app; [apply IH1, H1 |].
    apply wf_code_app; [apply IH2, H2 |].
    apply Forall_cons; [destruct o; simpl; lia |].
    apply wf_invert_code, IH2, H2.
Qed.

Lemma wf_compile : forall st, arith_stmt st = true -> wf_code (compile st).
Proof.
  induction st as [| x o e | x y | s1 IH1 s2 IH2]; simpl; intro H.
  - apply Forall_nil.
  - unfold gen_assign. rewrite (ungen_arith e scratch H).
    apply wf_code_app; [apply wf_gen_expr, H |].
    repeat (apply Forall_cons; [destruct o; simpl; first [exact I | lia] |]).
    apply wf_invert_code, wf_gen_expr, H.
  - unfold gen_swap; wf_list.
  - apply andb_prop in H as [H1 H2]. apply wf_code_app; auto.
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

(** Close one register of a [regs_ext] case split. *)
Ltac vclose :=
  first [ reflexivity
        | apply Z.lxor_nilpotent
        | rewrite Z.lxor_0_l; reflexivity
        | rewrite Z.lor_0_l; reflexivity
        | zcase ].

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

(** [g] / [u] evaluate a value [v] into register [r] and clear it again,
    touching nothing else, from any register file with [r0 = 0] that is
    clean from [r] up, over the memory [M]. *)
Definition expr_ok (g u : reg -> code) (r : reg) (v : Z) (M : addr -> Z) : Prop :=
  forall R, R 0%nat = 0 -> clean_above r (mkState R M) ->
    run (g r) (mkState R M) = mkState (rupd r v R) M /\
    run (u r) (mkState (rupd r v R) M) = mkState R M.

Lemma sltx_pair : forall v,
  Z.lxor (if v <? 0 then 1 else 0) (if 0 <? v then 1 else 0) = b2z (negb (v =? 0)).
Proof. intro v; zcase. Qed.

Lemma nz_code_ok : forall g u r v M R,
  expr_ok g u (S r) v M -> r <> 0%nat ->
  R 0%nat = 0 -> clean_above (S r) (mkState R M) ->
  run (nz_code g u r) (mkState R M)
  = mkState (rupd r (Z.lxor (R r) (b2z (negb (v =? 0)))) R) M.
Proof.
  intros g u r v M R Hok Hr H0 Hcl. unfold nz_code.
  rewrite run_app. destruct (Hok R H0 Hcl) as [Hg _]. rewrite Hg.
  rewrite !run_cons. cbn [step regs mem].
  repeat first [ rewrite rupd_same | rewrite rupd_other by lia ].
  rewrite H0, rupd_shadow, Z.lxor_assoc, sltx_pair.
  rewrite (rupd_comm r (S r)) by lia.
  apply (Hok (rupd r (Z.lxor (R r) (b2z (negb (v =? 0)))) R)).
  - rewrite rupd_other by lia. exact H0.
  - intros k Hk; cbn [regs]. rewrite rupd_other by lia. apply Hcl; lia.
Qed.

Lemma flag_code_ok : forall p g u r v M,
  (forall r', r' <> 0%nat -> expr_ok g u r' v M) ->
  (p = true -> v = 0 \/ v = 1) -> r <> 0%nat ->
  expr_ok (flag_code p g u) (unflag_code p g u) r (b2z (negb (v =? 0))) M.
Proof.
  intros p g u r v M Hok H01 Hr R H0 Hcl. unfold flag_code, unflag_code.
  destruct p.
  - destruct (H01 eq_refl) as [-> | ->]; apply Hok; assumption.
  - assert (HR : R r = 0) by (apply Hcl; cbn; lia).
    assert (Hcl' : clean_above (S r) (mkState R M)) by (intros k Hk; apply Hcl; lia).
    split.
    + rewrite (nz_code_ok g u r v M R (Hok (S r) ltac:(lia)) Hr H0 Hcl').
      now rewrite HR, Z.lxor_0_l.
    + rewrite (nz_code_ok g u r v M _ (Hok (S r) ltac:(lia)) Hr).
      * rewrite rupd_same, Z.lxor_nilpotent, rupd_shadow, rupd_zero by exact HR.
        reflexivity.
      * rewrite rupd_other by lia. exact H0.
      * intros k Hk; cbn [regs]. rewrite rupd_other by lia. apply Hcl; lia.
Qed.

Lemma is_flag_expr_01 : forall σ e, is_flag_expr e = true ->
  eval σ e = 0 \/ eval σ e = 1.
Proof.
  intros σ [k | x | o e1 e2] H; cbn [is_flag_expr eval] in *.
  - apply orb_true_iff in H as [H | H]; apply Z.eqb_eq in H; auto.
  - discriminate.
  - now apply denote_flag.
Qed.

Definition cmp_op (o : binop) : bool :=
  match o with OEq | ONe | OLt | OGt | OLe | OGe => true | _ => false end.

Lemma cmp_fwd_spec : forall o rd r1 r2 t R M,
  cmp_op o = true ->
  rd <> r1 -> rd <> r2 -> rd <> t -> t <> r1 -> t <> r2 ->
  R rd = 0 -> R t = 0 ->
  run (cmp_fwd o rd r1 r2 t) (mkState R M)
  = mkState (rupd rd (denote o (R r1) (R r2)) R) M.
Proof.
  intros o rd r1 r2 t R M Ho H1 H2 H3 H4 H5 Hrd Ht.
  destruct o; try discriminate; simp_state; rewrite ?Hrd, ?Ht; f_equal;
    regs_ext; rewrite ?Hrd, ?Ht; zcase.
Qed.

(** The five shapes of [gen_expr] / [ungen_expr] on [Bin], each proved once
    for arbitrary operand code [g1]/[u1], [g2]/[u2] that is correct at every
    register ([expr_ok]). *)

Section Cases.

Variables (g1 u1 g2 u2 : reg -> code) (v1 v2 : Z) (M : addr -> Z) (rt : reg).
Hypothesis Hrt : rt <> 0%nat.
Hypothesis Hok1 : forall r, r <> 0%nat -> expr_ok g1 u1 r v1 M.
Hypothesis Hok2 : forall r, r <> 0%nat -> expr_ok g2 u2 r v2 M.

(** Starting points of the operand code: [R] with [rt] (and more) updated. *)
Lemma start_ok : forall R a k, R 0%nat = 0 -> (forall j, (k <= j)%nat -> R j = 0) ->
  k <> 0%nat -> rupd k a R 0%nat = 0 /\ clean_above (S k) (mkState (rupd k a R) M).
Proof.
  intros R a k H0 Hz Hk; split; [rewrite rupd_other by lia; exact H0 |].
  intros j Hj; cbn [regs]; rewrite rupd_other by lia; apply Hz; lia.
Qed.

Lemma arith_case_ok : forall o, arith_op o = true ->
  expr_ok (fun r => g1 r ++ g2 (S r) ++ [op_instr o r (S r)] ++ u2 (S r))
          (fun r => g2 (S r) ++ [invert_instr (op_instr o r (S r))] ++ u2 (S r) ++ u1 r)
          rt (denote o v1 v2) M.
Proof.
  intros o Ea R H0 Hcl.
  assert (Hz : forall k, (rt <= k)%nat -> R k = 0) by (intros k Hk; apply (Hcl k Hk)).
  assert (Hop : step (op_instr o rt (S rt)) (mkState (rupd (S rt) v2 (rupd rt v1 R)) M)
                = mkState (rupd (S rt) v2 (rupd rt (denote o v1 v2) R)) M).
  { destruct o; try discriminate; simp_state; f_equal;
      rewrite (rupd_comm rt (S rt)) by lia; now rewrite rupd_shadow. }
  assert (Hinv : step (invert_instr (op_instr o rt (S rt)))
                      (mkState (rupd (S rt) v2 (rupd rt (denote o v1 v2) R)) M)
                 = mkState (rupd (S rt) v2 (rupd rt v1 R)) M).
  { destruct o; try discriminate; simp_state; f_equal;
      rewrite (rupd_comm rt (S rt)), rupd_shadow by lia; repeat f_equal.
    - ring.
    - ring.
    - apply xor_involutive. }
  destruct (Hok1 rt Hrt R H0 Hcl) as [G1 U1].
  destruct (start_ok R v1 rt H0 Hz Hrt) as [H01 C1].
  destruct (start_ok R (denote o v1 v2) rt H0 Hz Hrt) as [H0d Cd].
  destruct (Hok2 (S rt) ltac:(lia) _ H01 C1) as [G2 U2].
  destruct (Hok2 (S rt) ltac:(lia) _ H0d Cd) as [G2d U2d].
  split.
  - rewrite run_app, G1, run_app, G2, run_app, run_one, Hop, U2d. reflexivity.
  - rewrite run_app, G2d, run_app, run_one, Hinv, run_app, U2, U1. reflexivity.
Qed.

Lemma nz_case_ok :
  expr_ok (nz_code g1 u1) (nz_code g1 u1) rt (b2z (negb (v1 =? 0))) M.
Proof.
  intros R H0 Hcl.
  assert (Hz : forall k, (rt <= k)%nat -> R k = 0) by (intros k Hk; apply (Hcl k Hk)).
  assert (HR : R rt = 0) by (apply Hz; lia).
  assert (Hcl' : clean_above (S rt) (mkState R M)) by (intros k Hk; apply Hz; lia).
  split.
  - rewrite (nz_code_ok g1 u1 rt v1 M R (Hok1 (S rt) ltac:(lia)) Hrt H0 Hcl').
    now rewrite HR, Z.lxor_0_l.
  - rewrite (nz_code_ok g1 u1 rt v1 M _ (Hok1 (S rt) ltac:(lia)) Hrt).
    + rewrite rupd_same, Z.lxor_nilpotent, rupd_shadow, rupd_zero by exact HR.
      reflexivity.
    + rewrite rupd_other by lia. exact H0.
    + intros k Hk; cbn [regs]. rewrite rupd_other by lia. apply Hz; lia.
Qed.

Lemma cmp_case_ok : forall o, cmp_op o = true ->
  expr_ok (fun r => g1 (S r) ++ g2 (S (S r))
                    ++ cmp_fwd o r (S r) (S (S r)) (S (S (S r)))
                    ++ [IXor (S r) (S r); IXor (S (S r)) (S (S r))])
          (fun r => g1 (S r) ++ g2 (S (S r))
                    ++ cmp_fwd o (S (S (S r))) (S r) (S (S r)) (S (S (S (S r))))
                    ++ [IXor r (S (S (S r))); IXor (S (S (S r))) (S (S (S r)))]
                    ++ u2 (S (S r)) ++ u1 (S r))
          rt (denote o v1 v2) M.
Proof.
  intros o Ho R H0 Hcl.
  assert (Hz : forall k, (rt <= k)%nat -> R k = 0) by (intros k Hk; apply (Hcl k Hk)).
  assert (Z0 : R rt = 0) by (apply Hz; lia).
  assert (Z1 : R (S rt) = 0) by (apply Hz; lia).
  assert (Z2 : R (S (S rt)) = 0) by (apply Hz; lia).
  assert (Z3 : R (S (S (S rt))) = 0) by (apply Hz; lia).
  assert (Z4 : R (S (S (S (S rt)))) = 0) by (apply Hz; lia).
  set (d := denote o v1 v2).
  (* forward *)
  assert (Hcl1 : clean_above (S rt) (mkState R M)) by (intros k Hk; apply Hz; lia).
  destruct (Hok1 (S rt) ltac:(lia) R H0 Hcl1) as [G1 U1].
  destruct (start_ok R v1 (S rt) H0 ltac:(intros; apply Hz; lia) ltac:(lia)) as [H01 C1].
  destruct (Hok2 (S (S rt)) ltac:(lia) _ H01 C1) as [G2 U2].
  (* backward: the same operand code from [rupd rt d R] *)
  destruct (start_ok R d rt H0 Hz Hrt) as [H0d Cd].
  assert (Cd1 : clean_above (S rt) (mkState (rupd rt d R) M)) by exact Cd.
  destruct (Hok1 (S rt) ltac:(lia) _ H0d Cd1) as [G1d _].
  assert (H0d1 : rupd (S rt) v1 (rupd rt d R) 0%nat = 0)
    by (rewrite !rupd_other by lia; exact H0).
  assert (Cd2 : clean_above (S (S rt)) (mkState (rupd (S rt) v1 (rupd rt d R)) M)).
  { intros k Hk; cbn [regs]; rewrite !rupd_other by lia; apply Hz; lia. }
  destruct (Hok2 (S (S rt)) ltac:(lia) _ H0d1 Cd2) as [G2d _].
  split.
  - rewrite run_app, G1, run_app, G2, run_app, cmp_fwd_spec by
      (first [exact Ho | lia | cbn [regs]; rewrite !rupd_other by lia; assumption]).
    simp_state. f_equal. regs_ext; rewrite ?Z0, ?Z1, ?Z2; unfold d; try reflexivity; try apply Z.lxor_nilpotent; zcase.
  - rewrite run_app, G1d, run_app, G2d, run_app, cmp_fwd_spec by
      (first [exact Ho | lia | cbn [regs]; rewrite !rupd_other by lia; assumption]).
    repeat first [ rewrite rupd_same | rewrite rupd_other by lia ].
    assert (E : run [IXor rt (S (S (S rt))); IXor (S (S (S rt))) (S (S (S rt)))]
                  (mkState (rupd (S (S (S rt))) d (rupd (S (S rt)) v2 (rupd (S rt) v1 (rupd rt d R)))) M)
                = mkState (rupd (S (S rt)) v2 (rupd (S rt) v1 R)) M).
    { simp_state. f_equal. regs_ext; rewrite ?Z0, ?Z3; try reflexivity; try apply Z.lxor_nilpotent. }
    fold d. rewrite run_app, E, run_app, U2, U1. reflexivity.
Qed.

Section Logical.

Variables (p1 p2 : bool).
Hypothesis H01 : p1 = true -> v1 = 0 \/ v1 = 1.
Hypothesis H02 : p2 = true -> v2 = 0 \/ v2 = 1.

Let w1 := b2z (negb (v1 =? 0)).
Let w2 := b2z (negb (v2 =? 0)).

(** The two operand flags, from any register file clean from [S rt] up. *)
Lemma operands_ok : forall R, R 0%nat = 0 -> (forall k, (S rt <= k)%nat -> R k = 0) ->
  run (flag_code p1 g1 u1 (S rt) ++ flag_code p2 g2 u2 (S (S rt))) (mkState R M)
  = mkState (rupd (S (S rt)) w2 (rupd (S rt) w1 R)) M
  /\ run (unflag_code p2 g2 u2 (S (S rt)) ++ unflag_code p1 g1 u1 (S rt))
         (mkState (rupd (S (S rt)) w2 (rupd (S rt) w1 R)) M) = mkState R M.
Proof.
  intros R H0 Hz. unfold w1, w2.
  destruct (flag_code_ok p1 g1 u1 (S rt) v1 M Hok1 H01 ltac:(lia) R H0
              ltac:(intros k Hk; apply Hz; lia)) as [F1 U1].
  destruct (start_ok R (b2z (negb (v1 =? 0))) (S rt) H0 Hz ltac:(lia)) as [H01' C1].
  destruct (flag_code_ok p2 g2 u2 (S (S rt)) v2 M Hok2 H02 ltac:(lia) _ H01' C1) as [F2 U2].
  split.
  - rewrite run_app, F1, F2. reflexivity.
  - rewrite run_app, U2, U1. reflexivity.
Qed.

Lemma land_b2z : forall a b, Z.land (b2z a) (b2z b) = b2z (a && b).
Proof. destruct a, b; reflexivity. Qed.

Lemma lor_b2z : forall a b, Z.lor (b2z a) (b2z b) = b2z (a || b).
Proof. destruct a, b; reflexivity. Qed.

Lemma and_case_ok :
  expr_ok (fun r => flag_code p1 g1 u1 (S r) ++ flag_code p2 g2 u2 (S (S r))
                    ++ [IAndx r (S r) (S (S r)); IXor (S (S r)) (S (S r))])
          (fun r => flag_code p1 g1 u1 (S r) ++ flag_code p2 g2 u2 (S (S r))
                    ++ [ IXor (S (S (S r))) (S r)
                       ; IAndx (S (S (S (S r)))) (S (S (S r))) (S (S r))
                       ; IXor r (S (S (S (S r))))
                       ; IXor (S (S (S (S r)))) (S (S (S (S r)))) ]
                    ++ unflag_code p2 g2 u2 (S (S r)) ++ unflag_code p1 g1 u1 (S r))
          rt (denote OAnd v1 v2) M.
Proof.
  intros R H0 Hcl.
  assert (Hz : forall k, (rt <= k)%nat -> R k = 0) by (intros k Hk; apply (Hcl k Hk)).
  assert (Z0 : R rt = 0) by (apply Hz; lia).
  assert (Z1 : R (S rt) = 0) by (apply Hz; lia).
  assert (Z2 : R (S (S rt)) = 0) by (apply Hz; lia).
  assert (Z3 : R (S (S (S rt))) = 0) by (apply Hz; lia).
  assert (Z4 : R (S (S (S (S rt)))) = 0) by (apply Hz; lia).
  assert (Hd : denote OAnd v1 v2 = Z.land w1 w2) by (unfold w1, w2; now rewrite land_b2z).
  rewrite Hd.
  destruct (operands_ok R H0 ltac:(intros; apply Hz; lia)) as [F U].
  destruct (start_ok R (Z.land w1 w2) rt H0 Hz Hrt) as [H0d Cd].
  destruct (operands_ok (rupd rt (Z.land w1 w2) R) H0d Cd) as [Fd _].
  split.
  - rewrite app_assoc, run_app, F. simp_state. f_equal.
    regs_ext; rewrite ?Z0, ?Z1, ?Z2; vclose.
  - rewrite app_assoc, run_app, Fd, run_app.
    assert (E : run [ IXor (S (S (S rt))) (S rt)
                    ; IAndx (S (S (S (S rt)))) (S (S (S rt))) (S (S rt))
                    ; IXor rt (S (S (S (S rt))))
                    ; IXor (S (S (S (S rt)))) (S (S (S (S rt)))) ]
                  (mkState (rupd (S (S rt)) w2 (rupd (S rt) w1 (rupd rt (Z.land w1 w2) R))) M)
                = mkState (rupd (S (S rt)) w2 (rupd (S rt) w1 R)) M).
    { simp_state. rewrite Z3, Z4, !Z.lxor_0_l. f_equal.
      regs_ext; rewrite ?Z0, ?Z3, ?Z4; vclose. }
    rewrite E, U. reflexivity.
Qed.

Lemma or_case_ok :
  expr_ok (fun r => flag_code p1 g1 u1 (S r) ++ flag_code p2 g2 u2 (S (S r))
                    ++ [IOrx r (S r); IOrx r (S (S r))])
          (fun r => flag_code p1 g1 u1 (S r) ++ flag_code p2 g2 u2 (S (S r))
                    ++ [ IXor (S (S (S r))) (S r)
                       ; IOrx (S (S (S (S (S r))))) (S (S (S r)))
                       ; IXor (S (S (S (S r)))) (S (S r))
                       ; IOrx (S (S (S (S (S r))))) (S (S (S (S r))))
                       ; IXor r (S (S (S (S (S r)))))
                       ; IXor (S (S (S (S (S r))))) (S (S (S (S (S r))))) ]
                    ++ unflag_code p2 g2 u2 (S (S r)) ++ unflag_code p1 g1 u1 (S r))
          rt (denote OOr v1 v2) M.
Proof.
  intros R H0 Hcl.
  assert (Hz : forall k, (rt <= k)%nat -> R k = 0) by (intros k Hk; apply (Hcl k Hk)).
  assert (Z0 : R rt = 0) by (apply Hz; lia).
  assert (Z1 : R (S rt) = 0) by (apply Hz; lia).
  assert (Z2 : R (S (S rt)) = 0) by (apply Hz; lia).
  assert (Z3 : R (S (S (S rt))) = 0) by (apply Hz; lia).
  assert (Z4 : R (S (S (S (S rt)))) = 0) by (apply Hz; lia).
  assert (Z5 : R (S (S (S (S (S rt))))) = 0) by (apply Hz; lia).
  assert (Hd : denote OOr v1 v2 = Z.lor w1 w2) by (unfold w1, w2; now rewrite lor_b2z).
  rewrite Hd.
  destruct (operands_ok R H0 ltac:(intros; apply Hz; lia)) as [F U].
  destruct (start_ok R (Z.lor w1 w2) rt H0 Hz Hrt) as [H0d Cd].
  destruct (operands_ok (rupd rt (Z.lor w1 w2) R) H0d Cd) as [Fd _].
  split.
  - rewrite app_assoc, run_app, F. simp_state. rewrite Z0, Z.lor_0_l. f_equal.
    regs_ext; rewrite ?Z1, ?Z2; vclose.
  - rewrite app_assoc, run_app, Fd, run_app.
    assert (E : run [ IXor (S (S (S rt))) (S rt)
                    ; IOrx (S (S (S (S (S rt))))) (S (S (S rt)))
                    ; IXor (S (S (S (S rt)))) (S (S rt))
                    ; IOrx (S (S (S (S (S rt))))) (S (S (S (S rt))))
                    ; IXor rt (S (S (S (S (S rt)))))
                    ; IXor (S (S (S (S (S rt))))) (S (S (S (S (S rt))))) ]
                  (mkState (rupd (S (S rt)) w2 (rupd (S rt) w1 (rupd rt (Z.lor w1 w2) R))) M)
                = mkState (rupd (S (S rt)) w2 (rupd (S rt) w1 R)) M).
    { simp_state. rewrite Z3, Z4, Z5, !Z.lxor_0_l, Z.lor_0_l. f_equal.
      regs_ext; rewrite ?Z0, ?Z3, ?Z4, ?Z5; vclose. }
    rewrite E, U. reflexivity.
Qed.

End Logical.
End Cases.

Theorem gen_ungen_spec : forall σ M, (forall x : var, M (Z.of_nat x) = σ x) ->
  forall e rt, rt <> 0%nat -> expr_ok (gen_expr e) (ungen_expr e) rt (eval σ e) M.
Proof.
  intros σ M Hmod e.
  induction e as [n | x | o e1 IH1 e2 IH2]; intros rt Hrt.
  - (* Cst *)
    intros R H0 Hcl.
    assert (Hz : forall k, (rt <= k)%nat -> R k = 0) by (intros k Hk; apply (Hcl k Hk)).
    cbn [gen_expr ungen_expr eval]. simp_state. rewrite (Hz rt) by lia.
    rewrite Z.add_0_l, Z.sub_diag, rupd_shadow, rupd_zero by (apply Hz; lia).
    split; reflexivity.
  - (* Var *)
    intros R H0 Hcl.
    pose proof (gen_var_spec x rt (mkState R M) σ Hmod Hcl) as HG; cbn [regs mem] in HG.
    cbn [gen_expr ungen_expr eval]. split; [exact HG |].
    rewrite <- HG. apply run_invert_code. unfold gen_var; wf_list.
  - (* Bin *)
    cbn [eval].
    destruct (arith_op o) eqn:Ea.
    + (* + - ^ : the clean translation of the first version *)
      intros R H0 Hcl. cbn [gen_expr ungen_expr]. rewrite Ea.
      exact (arith_case_ok (gen_expr e1) (ungen_expr e1) (gen_expr e2) (ungen_expr e2)
               _ _ M rt Hrt IH1 IH2 o Ea R H0 Hcl).
    + destruct (is_nz_test o e2) eqn:Enz.
      * (* [e1 != 0]: `_gen_nonzero` *)
        destruct o; try discriminate Enz. destruct e2 as [k | | ]; try discriminate Enz.
        cbn [is_nz_test] in Enz; apply Z.eqb_eq in Enz; subst k.
        intros R H0 Hcl. cbn [gen_expr ungen_expr]. rewrite Ea.
        exact (nz_case_ok (gen_expr e1) (ungen_expr e1) _ M rt Hrt IH1 R H0 Hcl).
      * pose proof (is_flag_expr_01 σ e1) as F1.
        pose proof (is_flag_expr_01 σ e2) as F2.
        intros R H0 Hcl.
        destruct o; try discriminate Ea; cbn [gen_expr ungen_expr]; rewrite Ea, Enz;
          first
            [ refine (cmp_case_ok (gen_expr e1) (ungen_expr e1) (gen_expr e2) (ungen_expr e2)
                       _ _ M rt Hrt IH1 IH2 _ _ R H0 Hcl); reflexivity
            | exact (and_case_ok (gen_expr e1) (ungen_expr e1) (gen_expr e2) (ungen_expr e2)
                       _ _ M rt Hrt IH1 IH2 _ _ F1 F2 R H0 Hcl)
            | exact (or_case_ok (gen_expr e1) (ungen_expr e1) (gen_expr e2) (ungen_expr e2)
                       _ _ M rt Hrt IH1 IH2 _ _ F1 F2 R H0 Hcl) ].
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

    Generalised to comparisons and [&&]/[||]: the one new premise is
    [regs ms 0 = 0] (register [r0], hard-wired to 0 in `pisa_interp.py`,
    is an operand of the two [SLTX] of the [e != 0] normalisation, which
    `_logical_operands` and the programmer's own [e != 0] produce).  Every
    control-flow theorem already had it. *)

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

(** ** Corollary: on the original fragment the compiled code is reversible

    Semantic preservation plus [PISA.run_invert_code] gives the machine-level
    counterpart of [Src.exec_rev]: running the compiled code and then its
    instruction-wise inverse restores the whole machine state — for
    programs over [+ - ^] ([arith_stmt]).  With a comparison this fails
    ([compile_not_reversible]): `codegen.py`'s layout zeroes garbage with
    [XOR r r] and uses ORX, neither of which has a local inverse. *)

Corollary compile_reversible : forall st s, arith_stmt st = true ->
  run (invert_code (compile st)) (run (compile st) s) = s.
Proof. intros st s H; apply run_invert_code, wf_compile, H. Qed.

Example compile_not_reversible :
  let st := Assign 0%nat AAdd (Bin OEq (Var 1%nat) (Var 2%nat)) in
  run (invert_code (compile st)) (run (compile st) zero_state) <> zero_state.
Proof.
  intros st H. apply (f_equal (fun s => mem s 0)) in H. vm_compute in H. discriminate.
Qed.
