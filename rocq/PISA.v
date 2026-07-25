(** * PISA.v — machine model for the Pendulum instruction set

    A PISA machine state is a register file plus a memory, both total maps to
    [Z].  Every instruction modelled here is a *local update*: it changes one
    register (or exchanges a register with one memory cell) by a bijection, so
    each instruction has a local inverse.  That is the defining property of the
    architecture, and it is proved here as [step_invert] / [run_invert_code].

    Reversibility is not automatic: [ADD rd rs] with [rd = rs] computes
    [rd := 2*rd], whose inverse is not [SUB rd rd].  Such instructions are
    excluded by [wf_instr], and the compiler must only emit well-formed code
    (see Compile.v, where this is discharged). *)

From Stdlib Require Import ZArith List Lia.
From Stdlib Require Import FunctionalExtensionality.
Import ListNotations.
Open Scope Z_scope.

(** ** State *)

Definition reg  := nat.
Definition addr := Z.

Record state := mkState {
  regs : reg  -> Z;
  mem  : addr -> Z
}.

(** Pointwise update of a total map. *)
Definition rupd (r : reg) (v : Z) (f : reg -> Z) : reg -> Z :=
  fun r' => if Nat.eqb r' r then v else f r'.

Definition mupd (a : addr) (v : Z) (f : addr -> Z) : addr -> Z :=
  fun a' => if Z.eqb a' a then v else f a'.

Lemma rupd_same : forall r v f, rupd r v f r = v.
Proof. intros; unfold rupd; now rewrite Nat.eqb_refl. Qed.

Lemma rupd_other : forall r r' v f, r' <> r -> rupd r v f r' = f r'.
Proof.
  intros r r' v f H; unfold rupd.
  destruct (Nat.eqb_spec r' r); [contradiction | reflexivity].
Qed.

Lemma mupd_same : forall a v f, mupd a v f a = v.
Proof. intros; unfold mupd; now rewrite Z.eqb_refl. Qed.

Lemma mupd_other : forall a a' v f, a' <> a -> mupd a v f a' = f a'.
Proof.
  intros a a' v f H; unfold mupd.
  destruct (Z.eqb_spec a' a); [contradiction | reflexivity].
Qed.

(** Overwriting the same key twice keeps only the last write.  These are the
    only places where functional extensionality is used (see MANIFEST). *)
Lemma rupd_shadow : forall r v w f, rupd r v (rupd r w f) = rupd r v f.
Proof.
  intros; apply functional_extensionality; intro r'.
  unfold rupd; destruct (Nat.eqb r' r); reflexivity.
Qed.

Lemma rupd_id : forall r f, rupd r (f r) f = f.
Proof.
  intros; apply functional_extensionality; intro r'.
  unfold rupd; destruct (Nat.eqb_spec r' r); subst; reflexivity.
Qed.

Lemma mupd_shadow : forall a v w f, mupd a v (mupd a w f) = mupd a v f.
Proof.
  intros; apply functional_extensionality; intro a'.
  unfold mupd; destruct (Z.eqb a' a); reflexivity.
Qed.

Lemma mupd_comm : forall a1 a2 v1 v2 f,
  a1 <> a2 -> mupd a1 v1 (mupd a2 v2 f) = mupd a2 v2 (mupd a1 v1 f).
Proof.
  intros a1 a2 v1 v2 f H; apply functional_extensionality; intro a.
  unfold mupd.
  destruct (Z.eqb_spec a a1), (Z.eqb_spec a a2); subst; congruence.
Qed.

Lemma mupd_id : forall a f, mupd a (f a) f = f.
Proof.
  intros; apply functional_extensionality; intro a'.
  unfold mupd; destruct (Z.eqb_spec a' a); subst; reflexivity.
Qed.

(** ** Instructions

    The straight-line fragment of PISA: the arithmetic/logic updates and the
    memory exchange.  Control flow (BRA/RBRA/BEQ/…) is deliberately absent —
    see MANIFEST.md for the milestone structure. *)

Inductive instr : Type :=
| IAdd  (rd rs : reg)
| ISub  (rd rs : reg)
| IXor  (rd rs : reg)
| IAddi (rd : reg) (c : Z)
| ISubi (rd : reg) (c : Z)
| IXori (rd : reg) (c : Z)
| INeg  (rd : reg)
| IExch (rd ra : reg).   (** swap register [rd] with the memory cell addressed by [ra] *)

Definition code := list instr.

(** ** Semantics

    Every instruction is total, so [step] is a function rather than a relation. *)

Definition step (i : instr) (s : state) : state :=
  match i with
  | IAdd  rd rs => mkState (rupd rd (regs s rd + regs s rs) (regs s)) (mem s)
  | ISub  rd rs => mkState (rupd rd (regs s rd - regs s rs) (regs s)) (mem s)
  | IXor  rd rs => mkState (rupd rd (Z.lxor (regs s rd) (regs s rs)) (regs s)) (mem s)
  | IAddi rd c  => mkState (rupd rd (regs s rd + c) (regs s)) (mem s)
  | ISubi rd c  => mkState (rupd rd (regs s rd - c) (regs s)) (mem s)
  | IXori rd c  => mkState (rupd rd (Z.lxor (regs s rd) c) (regs s)) (mem s)
  | INeg  rd    => mkState (rupd rd (- regs s rd) (regs s)) (mem s)
  | IExch rd ra =>
      let a := regs s ra in
      mkState (rupd rd (mem s a) (regs s)) (mupd a (regs s rd) (mem s))
  end.

Definition run (c : code) (s : state) : state := fold_left (fun st i => step i st) c s.

Lemma run_nil : forall s, run [] s = s.
Proof. reflexivity. Qed.

Lemma run_cons : forall i c s, run (i :: c) s = run c (step i s).
Proof. reflexivity. Qed.

Lemma run_app : forall c1 c2 s, run (c1 ++ c2) s = run c2 (run c1 s).
Proof. intros; unfold run; now rewrite fold_left_app. Qed.

Lemma run_one : forall i s, run [i] s = step i s.
Proof. reflexivity. Qed.

(** ** Local inverses *)

Definition invert_instr (i : instr) : instr :=
  match i with
  | IAdd  rd rs => ISub  rd rs
  | ISub  rd rs => IAdd  rd rs
  | IXor  rd rs => IXor  rd rs      (* self-inverse *)
  | IAddi rd c  => ISubi rd c
  | ISubi rd c  => IAddi rd c
  | IXori rd c  => IXori rd c       (* self-inverse *)
  | INeg  rd    => INeg  rd         (* self-inverse *)
  | IExch rd ra => IExch rd ra      (* self-inverse *)
  end.

Definition invert_code (c : code) : code := rev (map invert_instr c).

(** An instruction is well-formed when its operand registers are distinct.
    [IAdd rd rd] would compute [rd := 2*rd], which [ISub rd rd] does not undo;
    [IXor rd rd] would clear [rd] irreversibly; and [IExch rd rd] would use the
    value being overwritten as its own address. *)
Definition wf_instr (i : instr) : Prop :=
  match i with
  | IAdd  rd rs | ISub rd rs | IXor rd rs | IExch rd rs => rd <> rs
  | IAddi _ _ | ISubi _ _ | IXori _ _ | INeg _ => True
  end.

Definition wf_code (c : code) : Prop := Forall wf_instr c.

Lemma xor_involutive : forall x y, Z.lxor (Z.lxor x y) y = x.
Proof.
  intros; rewrite Z.lxor_assoc, Z.lxor_nilpotent; apply Z.lxor_0_r.
Qed.

(** Each well-formed instruction is undone by its inverse. *)
Theorem step_invert : forall i s, wf_instr i -> step (invert_instr i) (step i s) = s.
Proof.
  intros i s Hwf; destruct s as [R M]; destruct i; simpl in *.
  - (* IAdd *) rewrite rupd_same, rupd_other by (now apply not_eq_sym).
    rewrite rupd_shadow. replace (R rd + R rs - R rs) with (R rd) by ring.
    now rewrite rupd_id.
  - (* ISub *) rewrite rupd_same, rupd_other by (now apply not_eq_sym).
    rewrite rupd_shadow. replace (R rd - R rs + R rs) with (R rd) by ring.
    now rewrite rupd_id.
  - (* IXor *) rewrite rupd_same, rupd_other by (now apply not_eq_sym).
    rewrite rupd_shadow, xor_involutive. now rewrite rupd_id.
  - (* IAddi *) rewrite rupd_same, rupd_shadow.
    replace (R rd + c - c) with (R rd) by ring. now rewrite rupd_id.
  - (* ISubi *) rewrite rupd_same, rupd_shadow.
    replace (R rd - c + c) with (R rd) by ring. now rewrite rupd_id.
  - (* IXori *) rewrite rupd_same, rupd_shadow, xor_involutive. now rewrite rupd_id.
  - (* INeg *) rewrite rupd_same, rupd_shadow.
    replace (- - R rd) with (R rd) by ring. now rewrite rupd_id.
  - (* IExch *) rewrite rupd_other by (now apply not_eq_sym).
    rewrite mupd_same, rupd_same, rupd_shadow, mupd_shadow.
    now rewrite rupd_id, mupd_id.
Qed.

Lemma wf_code_app : forall c1 c2, wf_code c1 -> wf_code c2 -> wf_code (c1 ++ c2).
Proof. intros; now apply Forall_app. Qed.

(** Running a well-formed block and then its inverse is the identity: the
    machine is reversible on all straight-line code the compiler emits. *)
Theorem run_invert_code : forall c s, wf_code c -> run (invert_code c) (run c s) = s.
Proof.
  induction c as [| i c IH]; intros s Hwf.
  - reflexivity.
  - inversion Hwf as [| ? ? Hi Hc]; subst.
    unfold invert_code; simpl; rewrite run_app, run_cons.
    change (rev (map invert_instr c)) with (invert_code c).
    rewrite IH by assumption. simpl. now apply step_invert.
Qed.

(** ** Frame lemmas

    The compiler needs to know which parts of the state a block leaves alone. *)

Definition writes_only (c : code) (P : reg -> Prop) : Prop :=
  forall s r, ~ P r -> regs (run c s) r = regs s r.

Definition preserves_mem (c : code) : Prop :=
  forall s a, mem (run c s) a = mem s a.

Lemma preserves_mem_app : forall c1 c2,
  preserves_mem c1 -> preserves_mem c2 -> preserves_mem (c1 ++ c2).
Proof. intros c1 c2 H1 H2 s a; rewrite run_app, H2; apply H1. Qed.

Lemma writes_only_app : forall c1 c2 P,
  writes_only c1 P -> writes_only c2 P -> writes_only (c1 ++ c2) P.
Proof. intros c1 c2 P H1 H2 s r Hr; rewrite run_app, H2 by exact Hr; now apply H1. Qed.

Lemma writes_only_weaken : forall c (P Q : reg -> Prop),
  writes_only c P -> (forall r, P r -> Q r) -> writes_only c Q.
Proof. intros c P Q H Himp s r Hr; apply H; intro; apply Hr; now apply Himp. Qed.

(** ** Sanity checks (executable tests) *)

Definition zero_state : state := mkState (fun _ => 0) (fun _ => 0).

Example ex_addi : regs (run [IAddi 3%nat 5] zero_state) 3%nat = 5.
Proof. reflexivity. Qed.

Example ex_add_copy :
  (* r4 := 7; r3 := r3 + r4 *)
  regs (run [IAddi 4%nat 7; IAdd 3%nat 4%nat] zero_state) 3%nat = 7.
Proof. reflexivity. Qed.

Example ex_exch_roundtrip :
  (* place 9 in r3, address 2 in r4, swap out and back *)
  let s := run [IAddi 3%nat 9; IAddi 4%nat 2; IExch 3%nat 4%nat; IExch 3%nat 4%nat]
               zero_state in
  regs s 3%nat = 9 /\ mem s 2 = 0.
Proof. split; reflexivity. Qed.

Example ex_exch_stores :
  let s := run [IAddi 3%nat 9; IAddi 4%nat 2; IExch 3%nat 4%nat] zero_state in
  mem s 2 = 9 /\ regs s 3%nat = 0.
Proof. split; reflexivity. Qed.

Example ex_invert_roundtrip : forall s,
  run (invert_code [IAddi 3%nat 5; IAdd 4%nat 3%nat])
      (run [IAddi 3%nat 5; IAdd 4%nat 3%nat] s) = s.
Proof.
  intro s; apply run_invert_code.
  repeat constructor; simpl; discriminate.
Qed.
