(** * LOpt.v — label removal preserves execution

    `remove_unused_labels` in `codegen.py` strips label annotations that no
    branch references (keeping `start`/`finish`, which the runtime looks up).
    Stating that this is sound needs a machine where labels *mean* something,
    so this file introduces the first PC-based model: labeled lines, direct
    branches, and a fuel-bounded executor.  The theorem [strip_exec] shows
    stripping unreferenced labels changes nothing about execution.

    Scope, deliberately: direct branches only (BRA/BEQ/BNE as simple jumps).
    The Pendulum paired-branch mechanism (RBRA, the `br` register, SWAPBR
    call/return) is milestone 1 in MANIFEST.md; nothing here depends on how
    those resolve, because label *lookup* works the same way.  The label
    forwarding done inside `_peephole_pass` (moving a label onto the line
    after a cancelled pair) is also not covered here. *)

From Stdlib Require Import ZArith List Lia Bool.
Require Import PISA.
Import ListNotations.
Open Scope Z_scope.

Definition label := nat.

(** A line: an optional label and either a straight-line op or a branch. *)
Inductive binstr : Type :=
| BOp  (i : instr)
| BBra (l : label)
| BBeq (rd rs : reg) (l : label)
| BBne (rd rs : reg) (l : label).

Definition line  := (option label * binstr)%type.
Definition lprog := list line.

(** ** Label lookup and reference *)

Fixpoint find_label (l : label) (p : lprog) : option nat :=
  match p with
  | [] => None
  | (Some l', _) :: t =>
      if Nat.eqb l l' then Some 0%nat
      else option_map S (find_label l t)
  | (None, _) :: t => option_map S (find_label l t)
  end.

Definition target (b : binstr) : option label :=
  match b with
  | BOp _ => None
  | BBra l | BBeq _ _ l | BBne _ _ l => Some l
  end.

Definition referenced (p : lprog) (l : label) : bool :=
  existsb (fun '(_, b) =>
    match target b with Some l' => Nat.eqb l l' | None => false end) p.

(** ** Execution: program counter + fuel

    Falling off the end halts; a branch to a missing label is an error. *)

Fixpoint exec_fuel (fuel : nat) (p : lprog) (pc : nat) (s : state)
  : option state :=
  match fuel with
  | O => None
  | S f =>
      match nth_error p pc with
      | None => Some s
      | Some (_, BOp i) => exec_fuel f p (S pc) (step i s)
      | Some (_, BBra l) =>
          match find_label l p with
          | Some pc' => exec_fuel f p pc' s
          | None => None
          end
      | Some (_, BBeq rd rs l) =>
          if Z.eqb (regs s rd) (regs s rs)
          then match find_label l p with
               | Some pc' => exec_fuel f p pc' s
               | None => None
               end
          else exec_fuel f p (S pc) s
      | Some (_, BBne rd rs l) =>
          if Z.eqb (regs s rd) (regs s rs)
          then exec_fuel f p (S pc) s
          else match find_label l p with
               | Some pc' => exec_fuel f p pc' s
               | None => None
               end
      end
  end.

(** ** The pass (mirror of `remove_unused_labels`)

    [keep] plays the role of the `start`/`finish` allowlist. *)

Definition strip (keep : list label) (p : lprog) : lprog :=
  map (fun '(lo, b) =>
    (match lo with
     | Some l => if referenced p l || existsb (Nat.eqb l) keep
                 then Some l else None
     | None => None
     end, b)) p.

(** Stripping never changes which instruction sits at an index. *)
Lemma strip_nth : forall keep p pc,
  nth_error (strip keep p) pc
  = option_map (fun '(lo, b) =>
      (match lo with
       | Some l => if referenced p l || existsb (Nat.eqb l) keep
                   then Some l else None
       | None => None
       end, b)) (nth_error p pc).
Proof. intros; unfold strip; apply nth_error_map. Qed.

(** A kept label resolves to the same position.  The generalization over the
    outer program [q] (whose [referenced] decides keeping) is what makes the
    induction go through. *)
Lemma find_label_strip_gen : forall l keep q p,
  referenced q l || existsb (Nat.eqb l) keep = true ->
  find_label l
    (map (fun '(lo, b) =>
      (match lo with
       | Some l' => if referenced q l' || existsb (Nat.eqb l') keep
                    then Some l' else None
       | None => None
       end, b)) p)
  = find_label l p.
Proof.
  intros l keep q p Hkeep; induction p as [| [lo b] t IH]; simpl.
  - reflexivity.
  - destruct lo as [l' |]; simpl.
    + destruct (Nat.eqb l l') eqn:El.
      * apply Nat.eqb_eq in El; subst l'.
        rewrite Hkeep; simpl. now rewrite Nat.eqb_refl.
      * destruct (referenced q l' || existsb (Nat.eqb l') keep); simpl;
          rewrite ?El, IH; reflexivity.
    + now rewrite IH.
Qed.

Corollary find_label_strip : forall l keep p,
  referenced p l || existsb (Nat.eqb l) keep = true ->
  find_label l (strip keep p) = find_label l p.
Proof. intros; now apply find_label_strip_gen. Qed.

(** A branch that occurs in the program makes its target referenced. *)
Lemma branch_referenced : forall p pc lo b l,
  nth_error p pc = Some (lo, b) -> target b = Some l ->
  referenced p l = true.
Proof.
  intros p pc lo b l Hnth Ht; unfold referenced.
  apply existsb_exists.
  exists (lo, b); split.
  - eapply nth_error_In; eauto.
  - now rewrite Ht, Nat.eqb_refl.
Qed.

(** ** Main theorem *)

Theorem strip_exec : forall fuel keep p pc s,
  exec_fuel fuel (strip keep p) pc s = exec_fuel fuel p pc s.
Proof.
  induction fuel as [| f IH]; intros keep p pc s; simpl.
  - reflexivity.
  - rewrite strip_nth.
    destruct (nth_error p pc) as [[lo b] |] eqn:Hnth; simpl; [| reflexivity].
    destruct b as [i | l | rd rs l | rd rs l]; simpl.
    + apply IH.
    + rewrite find_label_strip
        by (rewrite (branch_referenced p pc lo _ l Hnth) by reflexivity;
            reflexivity).
      destruct (find_label l p); [apply IH | reflexivity].
    + destruct (Z.eqb (regs s rd) (regs s rs)).
      * rewrite find_label_strip
          by (rewrite (branch_referenced p pc lo _ l Hnth) by reflexivity;
              reflexivity).
        destruct (find_label l p); [apply IH | reflexivity].
      * apply IH.
    + destruct (Z.eqb (regs s rd) (regs s rs)).
      * apply IH.
      * rewrite find_label_strip
          by (rewrite (branch_referenced p pc lo _ l Hnth) by reflexivity;
              reflexivity).
        destruct (find_label l p); [apply IH | reflexivity].
Qed.

(** ** Sanity checks *)

Example ex_strip_keeps_referenced :
  strip [] [(Some 7%nat, BOp (IAddi 3%nat 1)); (None, BBra 7%nat)]
  = [(Some 7%nat, BOp (IAddi 3%nat 1)); (None, BBra 7%nat)].
Proof. reflexivity. Qed.

Example ex_strip_drops_unreferenced :
  strip [] [(Some 7%nat, BOp (IAddi 3%nat 1))]
  = [(None, BOp (IAddi 3%nat 1))].
Proof. reflexivity. Qed.

Example ex_strip_keeps_allowlisted :
  (* the model's 'start' *)
  strip [0%nat] [(Some 0%nat, BOp (IAddi 3%nat 1))]
  = [(Some 0%nat, BOp (IAddi 3%nat 1))].
Proof. reflexivity. Qed.

Example ex_exec_loop :
  (* r3 += 1; if r3 <> r4 goto 0 — with r4 = 3 this runs three times *)
  exec_fuel 20
    [(Some 0%nat, BOp (IAddi 3%nat 1)); (None, BBne 3%nat 4%nat 0%nat)]
    0%nat (mkState (rupd 4%nat 3 (fun _ => 0)) (fun _ => 0))
  = Some (mkState (rupd 3%nat 3 (rupd 4%nat 3 (fun _ => 0))) (fun _ => 0)).
Proof.
  cbn.
  f_equal; f_equal.
  apply FunctionalExtensionality.functional_extensionality; intro r.
  unfold rupd; destruct (Nat.eqb r 3); destruct (Nat.eqb r 4); reflexivity.
Qed.

Print Assumptions strip_exec.
