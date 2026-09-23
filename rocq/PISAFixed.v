(** * PISAFixed.v — PISA.v with fixed-width ([b]-bit two's-complement) registers

    EXPERIMENT (see FIXED_WIDTH_REPORT.md).  PISA.v models registers and
    memory cells as unbounded [Z]; the real Pendulum machine has [b]-bit
    words.  This file is PISA.v with the arithmetic/logic results normalised
    into the signed window [-2^(b-1), 2^(b-1)) by [RevSMod.wrap], and
    re-proves every lemma of PISA.v, recording which statements survive
    unchanged, which need an extra hypothesis, and which break.

    Design choice (i), least invasive and closest to the machine: the state
    type is unchanged ([regs : reg -> Z], [mem : addr -> Z]); [step] applies
    [wrap] to every value an ALU instruction writes; [IExch] moves words
    without touching them (a move is the identity on words).  The set of
    machine-representable states is the invariant [wf_state]: every register
    and every memory cell is in the window.  [step] preserves it
    ([step_wf]), and reversibility ([step_invert], [run_invert_code]) holds
    on it — and provably NOT outside it ([step_invert_needs_wf]).

    Alternative (ii), a subset type [{z | in_window z}] as codomain, was not
    chosen: it changes the state type, hence every downstream statement, and
    it forces every map lemma (the rupd/mupd family) to be re-proved with proof
    irrelevance, for no gain in fidelity.

    Everything that does not depend on the width ([rupd]/[mupd], the
    instruction syntax, [invert_instr], [wf_instr], [xor_involutive]) is kept
    outside the [Fixed] section and is verbatim PISA.v. *)

From Stdlib Require Import ZArith List Lia.
From Stdlib Require Import FunctionalExtensionality.
Require Import RevSMod.
Import ListNotations.
Open Scope Z_scope.

(** ** State — verbatim PISA.v *)

Definition reg  := nat.
Definition addr := Z.

Record state := mkState {
  regs : reg  -> Z;
  mem  : addr -> Z
}.

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

(** ** Instructions — verbatim PISA.v (immediates stay [Z]; see report) *)

Inductive instr : Type :=
| IAdd  (rd rs : reg)
| ISub  (rd rs : reg)
| IXor  (rd rs : reg)
| IAddi (rd : reg) (c : Z)
| ISubi (rd : reg) (c : Z)
| IXori (rd : reg) (c : Z)
| INeg  (rd : reg)
| IExch (rd ra : reg).

Definition code := list instr.

Definition invert_instr (i : instr) : instr :=
  match i with
  | IAdd  rd rs => ISub  rd rs
  | ISub  rd rs => IAdd  rd rs
  | IXor  rd rs => IXor  rd rs
  | IAddi rd c  => ISubi rd c
  | ISubi rd c  => IAddi rd c
  | IXori rd c  => IXori rd c
  | INeg  rd    => INeg  rd
  | IExch rd ra => IExch rd ra
  end.

Definition invert_code (c : code) : code := rev (map invert_instr c).

Definition wf_instr (i : instr) : Prop :=
  match i with
  | IAdd  rd rs | ISub rd rs | IXor rd rs | IExch rd rs => rd <> rs
  | IAddi _ _ | ISubi _ _ | IXori _ _ | INeg _ => True
  end.

Definition wf_code (c : code) : Prop := Forall wf_instr c.

Lemma wf_code_app : forall c1 c2, wf_code c1 -> wf_code c2 -> wf_code (c1 ++ c2).
Proof. intros; now apply Forall_app. Qed.

Lemma xor_involutive : forall x y, Z.lxor (Z.lxor x y) y = x.
Proof.
  intros; rewrite Z.lxor_assoc, Z.lxor_nilpotent; apply Z.lxor_0_r.
Qed.

Definition zero_state : state := mkState (fun _ => 0) (fun _ => 0).

(** ** The width-parameterised part *)

Section Fixed.

Variable b : nat.
Hypothesis Hb : (0 < b)%nat.

Local Notation wrap := (wrap b).
Local Notation in_window := (in_window b).

(** *** Semantics: every ALU result is a [b]-bit word *)

Definition step (i : instr) (s : state) : state :=
  match i with
  | IAdd  rd rs => mkState (rupd rd (wrap (regs s rd + regs s rs)) (regs s)) (mem s)
  | ISub  rd rs => mkState (rupd rd (wrap (regs s rd - regs s rs)) (regs s)) (mem s)
  | IXor  rd rs => mkState (rupd rd (wrap (Z.lxor (regs s rd) (regs s rs))) (regs s)) (mem s)
  | IAddi rd c  => mkState (rupd rd (wrap (regs s rd + c)) (regs s)) (mem s)
  | ISubi rd c  => mkState (rupd rd (wrap (regs s rd - c)) (regs s)) (mem s)
  | IXori rd c  => mkState (rupd rd (wrap (Z.lxor (regs s rd) c)) (regs s)) (mem s)
  | INeg  rd    => mkState (rupd rd (wrap (- regs s rd)) (regs s)) (mem s)
  | IExch rd ra =>
      let a := regs s ra in
      mkState (rupd rd (mem s a) (regs s)) (mupd a (regs s rd) (mem s))
  end.

Definition run (c : code) (s : state) : state := fold_left (fun st i => step i st) c s.

(* run_nil / run_cons / run_app / run_one: UNCHANGED (they never look inside step). *)
Lemma run_nil : forall s, run [] s = s.
Proof. reflexivity. Qed.

Lemma run_cons : forall i c s, run (i :: c) s = run c (step i s).
Proof. reflexivity. Qed.

Lemma run_app : forall c1 c2 s, run (c1 ++ c2) s = run c2 (run c1 s).
Proof. intros; unfold run; now rewrite fold_left_app. Qed.

Lemma run_one : forall i s, run [i] s = step i s.
Proof. reflexivity. Qed.

(** *** Machine-representable states *)

Definition wf_state (s : state) : Prop :=
  (forall r, in_window (regs s r)) /\ (forall a, in_window (mem s a)).

Lemma wf_zero_state : wf_state zero_state.
Proof. split; intro; apply zero_in_window; exact Hb. Qed.

Lemma rupd_in_window : forall R r v,
  (forall r', in_window (R r')) -> in_window v -> forall r', in_window (rupd r v R r').
Proof.
  intros R r v HR Hv r'; unfold rupd; destruct (Nat.eqb r' r); [exact Hv | apply HR].
Qed.

Lemma mupd_in_window : forall M a v,
  (forall a', in_window (M a')) -> in_window v -> forall a', in_window (mupd a v M a').
Proof.
  intros M a v HM Hv a'; unfold mupd; destruct (Z.eqb a' a); [exact Hv | apply HM].
Qed.

(** [step] preserves representability — the NEW obligation of this model. *)
Lemma step_wf : forall i s, wf_state s -> wf_state (step i s).
Proof.
  intros i [R M] [HR HM]; destruct i; simpl in *; split; simpl;
    try exact HM; try exact HR;
    try (apply rupd_in_window; [exact HR | apply wrap_range; exact Hb]).
  - (* IExch: registers gain a memory word *) apply rupd_in_window; [exact HR | apply HM].
  - (* IExch: memory gains a register word *) apply mupd_in_window; [exact HM | apply HR].
Qed.

Lemma run_wf : forall c s, wf_state s -> wf_state (run c s).
Proof.
  induction c as [| i c IH]; intros s H; [exact H |].
  rewrite run_cons; apply IH, step_wf, H.
Qed.

(** *** Local inverses *)

(** step_invert: MODIFIED STATEMENT — needs [wf_state s].
    PISA.v proves it for every state; here a register outside the window is
    normalised by the first step and cannot be restored by the second
    ([step_invert_needs_wf] below). *)
Theorem step_invert : forall i s,
  wf_instr i -> wf_state s -> step (invert_instr i) (step i s) = s.
Proof.
  intros i s Hwf [HR HM]; destruct s as [R M]; simpl in *; destruct i; simpl in *.
  - (* IAdd: wrap_add_sub_cancel + wrap_id *)
    rewrite rupd_same, rupd_other by (now apply not_eq_sym).
    rewrite rupd_shadow, wrap_add_sub_cancel, wrap_id by (exact Hb || apply HR).
    now rewrite rupd_id.
  - (* ISub: wrap_sub_add_cancel + wrap_id *)
    rewrite rupd_same, rupd_other by (now apply not_eq_sym).
    rewrite rupd_shadow, wrap_sub_add_cancel, wrap_id by (exact Hb || apply HR).
    now rewrite rupd_id.
  - (* IXor: the operands are in the window, so the first wrap is the identity
       (lxor_in_window); then xor_involutive as before *)
    rewrite rupd_same, rupd_other by (now apply not_eq_sym).
    rewrite rupd_shadow.
    rewrite (wrap_id b Hb (Z.lxor (R rd) (R rs)))
      by (apply lxor_in_window; [exact Hb | apply HR | apply HR]).
    rewrite xor_involutive, wrap_id by (exact Hb || apply HR).
    now rewrite rupd_id.
  - (* IAddi: any immediate c : Z, in the window or not *)
    rewrite rupd_same, rupd_shadow, wrap_add_sub_cancel, wrap_id by (exact Hb || apply HR).
    now rewrite rupd_id.
  - (* ISubi *)
    rewrite rupd_same, rupd_shadow, wrap_sub_add_cancel, wrap_id by (exact Hb || apply HR).
    now rewrite rupd_id.
  - (* IXori: the immediate may be outside the window, so lxor_in_window does
       not apply; wrap_lxor_cancel (wrap commutes with xor mod 2^b) does *)
    rewrite rupd_same, rupd_shadow, wrap_lxor_cancel, wrap_id by (exact Hb || apply HR).
    now rewrite rupd_id.
  - (* INeg: wrap_neg_neg — note wrap (-(-2^(b-1))) = -2^(b-1), the classic
       two's-complement fixed point, is handled by the congruence law *)
    rewrite rupd_same, rupd_shadow, wrap_neg_neg, wrap_id by (exact Hb || apply HR).
    now rewrite rupd_id.
  - (* IExch: UNCHANGED proof — a move needs no wrap *)
    rewrite rupd_other by (now apply not_eq_sym).
    rewrite mupd_same, rupd_same, rupd_shadow, mupd_shadow.
    now rewrite rupd_id, mupd_id.
Qed.

(** Why the hypothesis is necessary: with r0 = 2^(b-1) (one past the window),
    [ADDI r0 0 ; SUBI r0 0] leaves r0 = -2^(b-1).  [bad_state] is the witness,
    reused by OptFixed.v. *)
Definition bad_state : state := mkState (fun _ => half b) (fun _ => 0).

Lemma addi_subi_bad :
  regs (step (ISubi 0%nat 0) (step (IAddi 0%nat 0) bad_state)) 0%nat = - half b.
Proof.
  unfold bad_state; simpl. rewrite !rupd_same.
  rewrite Z.add_0_r, Z.sub_0_r, wrap_half.
  apply wrap_id; [exact Hb |].
  pose proof (half_pos b Hb); unfold RevSMod.in_window; lia.
Qed.

Theorem step_invert_needs_wf :
  exists i s, wf_instr i /\ step (invert_instr i) (step i s) <> s.
Proof.
  exists (IAddi 0%nat 0), bad_state.
  split; [exact I |].
  intro H.
  assert (Hr := f_equal (fun st => regs st 0%nat) H).
  cbn [invert_instr] in Hr; rewrite addi_subi_bad in Hr.
  unfold bad_state in Hr; simpl in Hr.
  pose proof (half_pos b Hb); lia.
Qed.

(** run_invert_code: MODIFIED STATEMENT — needs [wf_state s]; the induction
    additionally needs [step_wf] to carry the invariant through the prefix. *)
Theorem run_invert_code : forall c s,
  wf_code c -> wf_state s -> run (invert_code c) (run c s) = s.
Proof.
  induction c as [| i c IH]; intros s Hwf Hs.
  - reflexivity.
  - inversion Hwf as [| ? ? Hi Hc]; subst.
    unfold invert_code; simpl; rewrite run_app, run_cons.
    change (rev (map invert_instr c)) with (invert_code c).
    rewrite IH by (assumption || apply step_wf, Hs). simpl.
    now apply step_invert.
Qed.

(** *** Frame lemmas — UNCHANGED (they never inspect step) *)

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

(** *** Sanity checks — MODIFIED STATEMENTS

    With an abstract width the literals 5, 7, 9 are only in the window when
    the width is large enough, so each example gains a lower bound on [b]
    (and its proof is no longer [reflexivity]).  The concrete instances after
    the section show both the passing (b = 8) and the failing (b = 3) cases. *)

Lemma small_in_window : forall (k : nat) z,
  (0 < k)%nat -> (k <= b)%nat ->
  - 2 ^ (Z.of_nat k - 1) <= z < 2 ^ (Z.of_nat k - 1) -> in_window z.
Proof. intros; eapply in_window_of_bits; eauto. Qed.

Example ex_addi : (4 <= b)%nat -> regs (run [IAddi 3%nat 5] zero_state) 3%nat = 5.
Proof.
  intro H4; simpl; unfold rupd; simpl.
  apply wrap_id; [exact Hb |]. apply (small_in_window 4); [lia | exact H4 | change (2 ^ (Z.of_nat 4 - 1)) with 8; lia].
Qed.

Example ex_add_copy : (4 <= b)%nat ->
  regs (run [IAddi 4%nat 7; IAdd 3%nat 4%nat] zero_state) 3%nat = 7.
Proof.
  intro H4; simpl; unfold rupd; simpl.
  rewrite wrap_idem by exact Hb.
  apply wrap_id; [exact Hb |]. apply (small_in_window 4); [lia | exact H4 | change (2 ^ (Z.of_nat 4 - 1)) with 8; lia].
Qed.

Example ex_exch_roundtrip : (5 <= b)%nat ->
  let s := run [IAddi 3%nat 9; IAddi 4%nat 2; IExch 3%nat 4%nat; IExch 3%nat 4%nat]
               zero_state in
  regs s 3%nat = 9 /\ mem s 2 = 0.
Proof.
  intro H5; simpl.
  rewrite (wrap_id b Hb 9) by (apply (small_in_window 5); [lia | exact H5 | change (2 ^ (Z.of_nat 5 - 1)) with 16; lia]).
  rewrite (wrap_id b Hb 2) by (apply (small_in_window 5); [lia | exact H5 | change (2 ^ (Z.of_nat 5 - 1)) with 16; lia]).
  unfold rupd, mupd; simpl. split; reflexivity.
Qed.

Example ex_exch_stores : (5 <= b)%nat ->
  let s := run [IAddi 3%nat 9; IAddi 4%nat 2; IExch 3%nat 4%nat] zero_state in
  mem s 2 = 9 /\ regs s 3%nat = 0.
Proof.
  intro H5; simpl.
  rewrite (wrap_id b Hb 9) by (apply (small_in_window 5); [lia | exact H5 | change (2 ^ (Z.of_nat 5 - 1)) with 16; lia]).
  rewrite (wrap_id b Hb 2) by (apply (small_in_window 5); [lia | exact H5 | change (2 ^ (Z.of_nat 5 - 1)) with 16; lia]).
  unfold rupd, mupd; simpl. split; reflexivity.
Qed.

(** ex_invert_roundtrip: MODIFIED STATEMENT — [forall s] becomes
    [forall s, wf_state s]. *)
Example ex_invert_roundtrip : forall s, wf_state s ->
  run (invert_code [IAddi 3%nat 5; IAdd 4%nat 3%nat])
      (run [IAddi 3%nat 5; IAdd 4%nat 3%nat] s) = s.
Proof.
  intros s Hs; apply run_invert_code; [| exact Hs].
  repeat constructor; simpl; discriminate.
Qed.

End Fixed.

(** ** Concrete widths: the examples of PISA.v at b = 8 (pass) and b = 3 (fail)

    At 8 bits every literal of PISA.v's examples fits, and the original
    statements hold by computation, exactly as in PISA.v. *)

Example ex_addi_8 : regs (run 8 [IAddi 3%nat 5] zero_state) 3%nat = 5.
Proof. reflexivity. Qed.

Example ex_add_copy_8 :
  regs (run 8 [IAddi 4%nat 7; IAdd 3%nat 4%nat] zero_state) 3%nat = 7.
Proof. reflexivity. Qed.

Example ex_exch_roundtrip_8 :
  let s := run 8 [IAddi 3%nat 9; IAddi 4%nat 2; IExch 3%nat 4%nat; IExch 3%nat 4%nat]
               zero_state in
  regs s 3%nat = 9 /\ mem s 2 = 0.
Proof. split; reflexivity. Qed.

Example ex_exch_stores_8 :
  let s := run 8 [IAddi 3%nat 9; IAddi 4%nat 2; IExch 3%nat 4%nat] zero_state in
  mem s 2 = 9 /\ regs s 3%nat = 0.
Proof. split; reflexivity. Qed.

(** At 3 bits (window [-4,4)) the literal 5 of [ex_addi] wraps to -3: the
    PISA.v statement is FALSE for this width. *)
Example ex_addi_3_wraps : regs (run 3 [IAddi 3%nat 5] zero_state) 3%nat = -3.
Proof. reflexivity. Qed.

Example ex_addi_3_breaks : regs (run 3 [IAddi 3%nat 5] zero_state) 3%nat <> 5.
Proof. discriminate. Qed.

(** Overflow is silent and reversible: 127 + 1 = -128 at 8 bits, and the
    inverse instruction brings it back. *)
Example ex_overflow_8 :
  regs (run 8 [IAddi 3%nat 127; IAddi 3%nat 1] zero_state) 3%nat = -128.
Proof. reflexivity. Qed.

Example ex_overflow_8_undone :
  regs (run 8 [IAddi 3%nat 127; IAddi 3%nat 1; ISubi 3%nat 1] zero_state) 3%nat = 127.
Proof. reflexivity. Qed.
