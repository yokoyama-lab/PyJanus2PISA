(** * Src.v — the source language (straight-line fragment of Janus)

    Definitions are deliberately kept identical in shape to [Janus.v] of the
    PyJanus development (yokoyama-lab/PyJanus, coq/Janus.v), so that results
    proved there transfer: [var := nat], [store := var -> Z], the same
    [update], the same [aop]/[adenote]/[ainv], the same [occurs] side condition
    on assignment, and an [exec] relation whose rules for the constructors
    below are literally those of [Janus.exec].

    This file covers the constructors the compiler currently handles —
    [Skip], [Assign], [Swap], [Seq].  [If] / [Loop] / [Call] / [Uncall] need
    PISA control flow and are milestone 2 (see MANIFEST.md). *)

From Stdlib Require Import ZArith List Lia Bool.
From Stdlib Require Import FunctionalExtensionality.
Open Scope Z_scope.

(** ** Store *)

Definition var := nat.
Definition store := var -> Z.

Definition update (s : store) (x : var) (v : Z) : store :=
  fun y => if Nat.eqb x y then v else s y.

Lemma update_eq : forall s x v, update s x v x = v.
Proof. intros; unfold update; now rewrite Nat.eqb_refl. Qed.

Lemma update_neq : forall s x v y, x <> y -> update s x v y = s y.
Proof.
  intros s x v y H; unfold update.
  destruct (Nat.eqb_spec x y); [contradiction | reflexivity].
Qed.

(** Janus's simultaneous swap of two cells. *)
Definition sw (s : store) (x y : var) : store :=
  update (update s x (s y)) y (s x).

(** ** Syntax *)

Inductive binop := OAdd | OSub | OXor.

Definition denote (o : binop) (a b : Z) : Z :=
  match o with
  | OAdd => a + b
  | OSub => a - b
  | OXor => Z.lxor a b
  end.

Inductive expr :=
| Cst (n : Z)
| Var (x : var)
| Bin (o : binop) (e1 e2 : expr).

Inductive aop := AAdd | ASub | AXor.

Definition adenote (o : aop) (a b : Z) : Z :=
  match o with
  | AAdd => a + b
  | ASub => a - b
  | AXor => Z.lxor a b
  end.

Definition ainv (o : aop) : aop :=
  match o with AAdd => ASub | ASub => AAdd | AXor => AXor end.

Inductive stmt :=
| Skip
| Assign (x : var) (o : aop) (e : expr)   (** [x op= e], needs [occurs x e = false] *)
| Swap   (x y : var)                      (** [x <=> y] *)
| Seq    (s1 s2 : stmt).

(** Syntactic well-formedness.

    Only [Swap] carries an obligation: swapping a cell with itself must be
    rejected.  Semantically [sw s x x = s], but every reversible lowering of a
    swap (the XOR triple used by vjanus, the paired exchange used here)
    destroys the cell when the two operands alias — the same restriction
    PyJanus and vjanus impose, and the reason [RevLowering.v] proves the
    aliased case unsound. *)
Fixpoint wf_stmt (st : stmt) : Prop :=
  match st with
  | Skip         => True
  | Assign _ _ _ => True
  | Swap x y     => x <> y
  | Seq s1 s2    => wf_stmt s1 /\ wf_stmt s2
  end.

(** ** Expression evaluation and the occurs check *)

Fixpoint eval (s : store) (e : expr) : Z :=
  match e with
  | Cst n => n
  | Var x => s x
  | Bin o e1 e2 => denote o (eval s e1) (eval s e2)
  end.

Fixpoint occurs (x : var) (e : expr) : bool :=
  match e with
  | Cst _ => false
  | Var y => Nat.eqb x y
  | Bin _ e1 e2 => orb (occurs x e1) (occurs x e2)
  end.

Lemma eval_update_notin : forall x v s e,
  occurs x e = false -> eval (update s x v) e = eval s e.
Proof.
  intros x v s e; induction e as [n | y | o e1 IH1 e2 IH2]; simpl; intro H.
  - reflexivity.
  - apply update_neq. intro; subst. now rewrite Nat.eqb_refl in H.
  - apply orb_false_elim in H as [H1 H2]. now rewrite IH1, IH2.
Qed.

(** ** Semantics

    The rules are those of [Janus.exec] restricted to this fragment. *)

Inductive exec : stmt -> store -> store -> Prop :=
| E_Skip   : forall s, exec Skip s s
| E_Assign : forall x o e s,
    occurs x e = false ->
    exec (Assign x o e) s (update s x (adenote o (s x) (eval s e)))
| E_Swap   : forall x y s, exec (Swap x y) s (sw s x y)
| E_Seq    : forall s1 s2 a m b, exec s1 a m -> exec s2 m b -> exec (Seq s1 s2) a b.

(** ** Inversion

    [invert] mirrors [Janus.invert] on this fragment. *)

Fixpoint invert (s : stmt) : stmt :=
  match s with
  | Skip         => Skip
  | Assign x o e => Assign x (ainv o) e
  | Swap x y     => Swap x y
  | Seq s1 s2    => Seq (invert s2) (invert s1)
  end.

Lemma ainv_correct : forall o a b, adenote (ainv o) (adenote o a b) b = a.
Proof.
  destruct o; simpl; intros.
  - ring.
  - ring.
  - rewrite Z.lxor_assoc, Z.lxor_nilpotent; apply Z.lxor_0_r.
Qed.

Lemma update_shadow : forall s x a b, update (update s x a) x b = update s x b.
Proof.
  intros; apply functional_extensionality; intro y.
  unfold update; destruct (Nat.eqb x y); reflexivity.
Qed.

Lemma update_same : forall s x, update s x (s x) = s.
Proof.
  intros; apply functional_extensionality; intro y.
  unfold update; destruct (Nat.eqb_spec x y); subst; reflexivity.
Qed.

Lemma sw_invol : forall s x y, sw (sw s x y) x y = s.
Proof.
  intros s x y; apply functional_extensionality; intro z.
  unfold sw, update.
  destruct (Nat.eqb_spec y z) as [Hyz|Hyz];
  destruct (Nat.eqb_spec x z) as [Hxz|Hxz];
  destruct (Nat.eqb_spec y x) as [Hyx|Hyx];
  subst; repeat rewrite Nat.eqb_refl; try reflexivity; try congruence.
Qed.

(** Running an inverted assignment backwards restores the original store.
    This mirrors [Janus.assign_inv_ok]. *)
Lemma assign_inv_ok : forall s x o e,
  occurs x e = false ->
  exec (Assign x (ainv o) e) (update s x (adenote o (s x) (eval s e))) s.
Proof.
  intros s x o e H.
  pose proof (E_Assign x (ainv o) e (update s x (adenote o (s x) (eval s e))) H) as HH.
  replace (update (update s x (adenote o (s x) (eval s e))) x
             (adenote (ainv o) (update s x (adenote o (s x) (eval s e)) x)
                (eval (update s x (adenote o (s x) (eval s e))) e)))
    with s in HH by
    (rewrite update_eq, eval_update_notin by assumption;
     rewrite ainv_correct, update_shadow, update_same; reflexivity).
  exact HH.
Qed.

Lemma swap_inv_ok : forall s x y, exec (Swap x y) (sw s x y) s.
Proof.
  intros s x y.
  pose proof (E_Swap x y (sw s x y)) as HH.
  now rewrite sw_invol in HH.
Qed.

(** The fragment is reversible: the inverted program runs backwards. *)
Theorem exec_rev : forall st a b, exec st a b -> exec (invert st) b a.
Proof.
  intros st a b H; induction H; simpl.
  - constructor.
  - now apply assign_inv_ok.
  - apply swap_inv_ok.
  - econstructor; eassumption.
Qed.

(** ** Sanity checks *)

Definition empty : store := fun _ => 0.

Example ex_assign :
  exec (Assign 0%nat AAdd (Cst 5)) empty (update empty 0%nat 5).
Proof. replace 5 with (adenote AAdd (empty 0%nat) (eval empty (Cst 5))) at 2 by reflexivity.
       constructor; reflexivity. Qed.

Example ex_occurs_blocks : occurs 0%nat (Bin OAdd (Var 0%nat) (Cst 1)) = true.
Proof. reflexivity. Qed.

Example ex_eval : eval (update empty 1%nat 7) (Bin OAdd (Var 1%nat) (Cst 2)) = 9.
Proof. reflexivity. Qed.
