(** * RevSMod.v — signed fixed-width arithmetic (STAND-IN)

    This file is a stand-in for [RevSMod.v] of the PyJanus development
    (yokoyama-lab/PyJanus, coq/RevSMod.v), written here because that file is
    not in this repository.  It reproduces only what PISAFixed.v needs: the
    signed window [-2^(b-1), 2^(b-1)) of a [b]-bit two's-complement word, the
    normalisation [wrap] into that window, and the algebraic laws that make
    wrapped [+], [-], [neg] and [xor] behave like their unbounded versions
    "modulo the window".  PyJanus validates its RevSMod.v against the output
    of `pyjanus -m 8`; nothing here has been validated against anything but
    the proofs below, so when the real file becomes available it should
    replace this one (see FIXED_WIDTH_REPORT.md, "plugging in the real
    RevSMod.v").

    All definitions are parameterised by the width [b : nat] with [0 < b]. *)

From Stdlib Require Import ZArith Lia Bool.
Open Scope Z_scope.

Section Width.

Variable b : nat.
Hypothesis Hb : (0 < b)%nat.

(** [half = 2^(b-1)], [modulus = 2^b]. *)
Definition half : Z := 2 ^ (Z.of_nat b - 1).
Definition modulus : Z := 2 * half.

(** Normalise into the signed window. *)
Definition wrap (z : Z) : Z := (z + half) mod modulus - half.

Definition in_window (z : Z) : Prop := - half <= z < half.

Lemma half_pos : 0 < half.
Proof. unfold half; apply Z.pow_pos_nonneg; lia. Qed.

Lemma modulus_pos : 0 < modulus.
Proof. unfold modulus; pose proof half_pos; lia. Qed.

Lemma modulus_pow : modulus = 2 ^ Z.of_nat b.
Proof.
  unfold modulus, half.
  replace (Z.of_nat b) with (Z.succ (Z.of_nat b - 1)) at 2 by lia.
  rewrite Z.pow_succ_r by lia. reflexivity.
Qed.

(** ** The window *)

Lemma wrap_range : forall z, in_window (wrap z).
Proof.
  intro z; unfold in_window, wrap.
  pose proof (Z.mod_pos_bound (z + half) modulus modulus_pos).
  unfold modulus in *; lia.
Qed.

Lemma wrap_id : forall z, in_window z -> wrap z = z.
Proof.
  intros z [H1 H2]; unfold wrap.
  rewrite Z.mod_small by (unfold modulus; lia). ring.
Qed.

Lemma wrap_idem : forall z, wrap (wrap z) = wrap z.
Proof. intro z; apply wrap_id, wrap_range. Qed.

Lemma zero_in_window : in_window 0.
Proof. unfold in_window; pose proof half_pos; lia. Qed.

(** One past the top of the window wraps to the bottom (2^(b-1) ↦ -2^(b-1)). *)
Lemma wrap_half : wrap half = - half.
Proof.
  unfold wrap, modulus.
  replace (half + half) with (1 * (2 * half)) by ring.
  rewrite Z_mod_mult. ring.
Qed.

(** ** Congruence laws: [wrap] is a ring homomorphism onto Z/2^b, read back
    into the signed window. *)

Lemma wrap_eq : forall z, wrap z = z - modulus * ((z + half) / modulus).
Proof.
  intro z; unfold wrap.
  rewrite Z.mod_eq by (pose proof modulus_pos; lia). ring.
Qed.

Lemma wrap_add_mult : forall z k, wrap (z + modulus * k) = wrap z.
Proof.
  intros; unfold wrap.
  replace (z + modulus * k + half) with ((z + half) + k * modulus) by ring.
  rewrite Z_mod_plus_full. reflexivity.
Qed.

Lemma wrap_add_l : forall x y, wrap (wrap x + y) = wrap (x + y).
Proof.
  intros. rewrite (wrap_eq x).
  replace (x - modulus * ((x + half) / modulus) + y)
    with ((x + y) + modulus * (- ((x + half) / modulus))) by ring.
  apply wrap_add_mult.
Qed.

Lemma wrap_add_r : forall x y, wrap (x + wrap y) = wrap (x + y).
Proof. intros; rewrite Z.add_comm, wrap_add_l, Z.add_comm; reflexivity. Qed.

Lemma wrap_sub_l : forall x y, wrap (wrap x - y) = wrap (x - y).
Proof.
  intros. rewrite (wrap_eq x).
  replace (x - modulus * ((x + half) / modulus) - y)
    with ((x - y) + modulus * (- ((x + half) / modulus))) by ring.
  apply wrap_add_mult.
Qed.

Lemma wrap_sub_r : forall x y, wrap (x - wrap y) = wrap (x - y).
Proof.
  intros. rewrite (wrap_eq y).
  replace (x - (y - modulus * ((y + half) / modulus)))
    with ((x - y) + modulus * ((y + half) / modulus)) by ring.
  apply wrap_add_mult.
Qed.

Lemma wrap_neg : forall x, wrap (- wrap x) = wrap (- x).
Proof.
  intros. rewrite (wrap_eq x).
  replace (- (x - modulus * ((x + half) / modulus)))
    with ((- x) + modulus * ((x + half) / modulus)) by ring.
  apply wrap_add_mult.
Qed.

(** The cancellation laws PISAFixed.step_invert needs. *)
Lemma wrap_add_sub_cancel : forall x y, wrap (wrap (x + y) - y) = wrap x.
Proof. intros; rewrite wrap_sub_l. f_equal; ring. Qed.

Lemma wrap_sub_add_cancel : forall x y, wrap (wrap (x - y) + y) = wrap x.
Proof. intros; rewrite wrap_add_l. f_equal; ring. Qed.

Lemma wrap_neg_neg : forall x, wrap (- wrap (- x)) = wrap x.
Proof. intros; rewrite wrap_neg. f_equal; ring. Qed.

(** [wrap] only depends on the residue modulo [2^b]. *)
Lemma wrap_mod : forall x, wrap x mod modulus = x mod modulus.
Proof.
  intro x; rewrite (wrap_eq x).
  replace (x - modulus * ((x + half) / modulus))
    with (x + ((- ((x + half) / modulus)) * modulus)) by ring.
  apply Z_mod_plus_full.
Qed.

Lemma wrap_congr : forall x y, x mod modulus = y mod modulus -> wrap x = wrap y.
Proof.
  intros x y H.
  pose proof modulus_pos as Hm.
  rewrite (Z.div_mod x modulus) by lia.
  rewrite (Z.div_mod y modulus) by lia.
  rewrite H.
  rewrite (Z.add_comm (modulus * (x / modulus))), (Z.add_comm (modulus * (y / modulus))).
  rewrite !wrap_add_mult. reflexivity.
Qed.

(** ** Bitwise xor

    Rocq's [Z.lxor] is two's-complement xor on unbounded integers, so on the
    low [b] bits it agrees with the machine's xor.  Two facts are needed:
    (A) xor of two in-window values stays in the window (no [wrap] needed);
    (B) [wrap] commutes with xor modulo the window (needed when one operand,
    e.g. an immediate, is not in the window). *)

(** Auxiliary: xor of two non-negative values below [2^n] stays below [2^n]. *)
Lemma lxor_lt_pow2 : forall n a c,
  0 <= n -> 0 <= a < 2 ^ n -> 0 <= c < 2 ^ n -> 0 <= Z.lxor a c < 2 ^ n.
Proof.
  intros n a c Hn [Ha1 Ha2] [Hc1 Hc2].
  assert (Hnn : 0 <= Z.lxor a c) by (apply Z.lxor_nonneg; tauto).
  split; [exact Hnn |].
  destruct (Z.eq_dec (Z.lxor a c) 0) as [E | E].
  - rewrite E. apply Z.pow_pos_nonneg; lia.
  - assert (Hpos : 0 < Z.lxor a c) by lia.
    assert (Hn0 : 0 < n).
    { destruct (Z.eq_dec n 0) as [-> | Hne]; [| lia].
      simpl in Ha2, Hc2.
      assert (a = 0) by lia. assert (c = 0) by lia. subst.
      now rewrite Z.lxor_0_l in E. }
    apply Z.log2_lt_pow2; [exact Hpos |].
    pose proof (Z.log2_lxor a c Ha1 Hc1) as Hl.
    assert (Hla : Z.log2 a < n).
    { destruct (Z.eq_dec a 0) as [-> | Ha0]; [simpl; lia |].
      apply Z.log2_lt_pow2; lia. }
    assert (Hlc : Z.log2 c < n).
    { destruct (Z.eq_dec c 0) as [-> | Hc0]; [simpl; lia |].
      apply Z.log2_lt_pow2; lia. }
    lia.
Qed.

Lemma lnot_in_window_nonneg : forall x,
  in_window x -> x < 0 -> 0 <= Z.lnot x < half.
Proof. intros x [H1 H2] Hneg; unfold Z.lnot; lia. Qed.

(** (A) *)
Lemma lxor_in_window : forall x y,
  in_window x -> in_window y -> in_window (Z.lxor x y).
Proof.
  intros x y Hx Hy.
  pose proof half_pos as Hh.
  assert (Hn : 0 <= Z.of_nat b - 1) by lia.
  unfold in_window in *.
  destruct (Z_lt_le_dec x 0) as [Hxn | Hxp];
  destruct (Z_lt_le_dec y 0) as [Hyn | Hyp].
  - (* both negative: lxor (lnot x') (lnot y') = lxor x' y' >= 0 *)
    rewrite <- (Z.lnot_involutive x), <- (Z.lnot_involutive y).
    rewrite Z.lxor_lnot_lnot.
    pose proof (lxor_lt_pow2 (Z.of_nat b - 1) (Z.lnot x) (Z.lnot y) Hn
                  (lnot_in_window_nonneg x Hx Hxn)
                  (lnot_in_window_nonneg y Hy Hyn)).
    fold half in *; lia.
  - (* x < 0 <= y: lxor (lnot x') y = lnot (lxor x' y) *)
    rewrite <- (Z.lnot_involutive x).
    rewrite <- Z.lnot_lxor_l.
    pose proof (lxor_lt_pow2 (Z.of_nat b - 1) (Z.lnot x) y Hn
                  (lnot_in_window_nonneg x Hx Hxn) (conj Hyp (proj2 Hy))).
    fold half in *; set (w := Z.lxor (Z.lnot x) y) in *; unfold Z.lnot; lia.
  - (* y < 0 <= x *)
    rewrite <- (Z.lnot_involutive y).
    rewrite <- Z.lnot_lxor_r.
    pose proof (lxor_lt_pow2 (Z.of_nat b - 1) x (Z.lnot y) Hn
                  (conj Hxp (proj2 Hx)) (lnot_in_window_nonneg y Hy Hyn)).
    fold half in *; set (w := Z.lxor x (Z.lnot y)) in *; unfold Z.lnot; lia.
  - (* both non-negative *)
    pose proof (lxor_lt_pow2 (Z.of_nat b - 1) x y Hn
                  (conj Hxp (proj2 Hx)) (conj Hyp (proj2 Hy))).
    fold half in *; lia.
Qed.

(** Residues modulo [2^b] are preserved by xor: only the low [b] bits matter. *)
Lemma lxor_mod_pow2 : forall n a a' y,
  0 <= n ->
  a mod 2 ^ n = a' mod 2 ^ n ->
  (Z.lxor a y) mod 2 ^ n = (Z.lxor a' y) mod 2 ^ n.
Proof.
  intros n a a' y Hn H.
  apply Z.bits_inj'; intros i Hi.
  destruct (Z_lt_le_dec i n) as [Hlt | Hge].
  - rewrite !Z.mod_pow2_bits_low by exact Hlt.
    rewrite !Z.lxor_spec.
    f_equal.
    rewrite <- (Z.mod_pow2_bits_low a n i Hlt), <- (Z.mod_pow2_bits_low a' n i Hlt).
    now rewrite H.
  - rewrite !Z.mod_pow2_bits_high by lia. reflexivity.
Qed.

(** (B) *)
Lemma wrap_lxor_l : forall x y, wrap (Z.lxor (wrap x) y) = wrap (Z.lxor x y).
Proof.
  intros x y. apply wrap_congr.
  rewrite modulus_pow.
  apply lxor_mod_pow2; [lia |].
  rewrite <- modulus_pow. apply wrap_mod.
Qed.

Lemma wrap_lxor_r : forall x y, wrap (Z.lxor x (wrap y)) = wrap (Z.lxor x y).
Proof. intros; rewrite Z.lxor_comm, wrap_lxor_l, Z.lxor_comm; reflexivity. Qed.

Lemma wrap_lxor_cancel : forall x y, wrap (Z.lxor (wrap (Z.lxor x y)) y) = wrap x.
Proof.
  intros; rewrite wrap_lxor_l, Z.lxor_assoc, Z.lxor_nilpotent, Z.lxor_0_r.
  reflexivity.
Qed.

(** ** Facts about the window used by the executable examples *)

(** The window contains [[-2^(k-1), 2^(k-1))] as soon as [k <= b]. *)
Lemma in_window_of_bits : forall (k : nat) z,
  (0 < k)%nat -> (k <= b)%nat ->
  - 2 ^ (Z.of_nat k - 1) <= z < 2 ^ (Z.of_nat k - 1) -> in_window z.
Proof.
  intros k z Hk Hkb [H1 H2]; unfold in_window, half.
  assert (2 ^ (Z.of_nat k - 1) <= 2 ^ (Z.of_nat b - 1))
    by (apply Z.pow_le_mono_r; lia).
  lia.
Qed.

End Width.

(** Concrete sanity checks at 8 bits (the width PyJanus's [-m 8] uses). *)
Example wrap8_127 : wrap 8 127 = 127.   Proof. reflexivity. Qed.
Example wrap8_128 : wrap 8 128 = -128.  Proof. reflexivity. Qed.
Example wrap8_m129 : wrap 8 (-129) = 127. Proof. reflexivity. Qed.
Example wrap8_300 : wrap 8 300 = 44.    Proof. reflexivity. Qed.
Example wrap3_5 : wrap 3 5 = -3.        Proof. reflexivity. Qed.
