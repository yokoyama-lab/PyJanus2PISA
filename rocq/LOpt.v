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

(** * Deleting a cancelling pair, with label forwarding

    The remaining `_peephole_pass` obligation: on labeled code, deleting an
    adjacent cancelling pair — forwarding the pair's label onto the next line
    when it has one — preserves every terminating run.  Deletion shifts all
    subsequent positions by two, so this is a forward simulation with the pc
    map [phi]: positions before the pair are unchanged, positions after it
    move down by two, and both the pair's own position and its successor land
    on the merged line. *)

Require Import Opt.

Definition nopstep (i : instr) : Prop := forall s, step i s = s.

(** [merge lo p2] is the code after the pair: label [lo] forwarded onto the
    head when present.  [mergeable] is the side condition `_peephole_pass`
    (now) enforces: forwarding needs an unlabeled successor, and an unlabeled
    pair needs nothing. *)
Definition merge (lo : option label) (p2 : lprog) : lprog :=
  match lo, p2 with
  | Some l, (None, b) :: t => (Some l, b) :: t
  | _, _ => p2
  end.

Definition mergeable (lo : option label) (p2 : lprog) : Prop :=
  match lo with
  | None => True
  | Some _ => match p2 with (None, _) :: _ => True | _ => False end
  end.

Definition phi (n pc : nat) : nat := if Nat.leb pc n then pc else (pc - 2)%nat.

(** One-step unfolding equation, used instead of [simpl] so that section-local
    abbreviations are not unfolded. *)
Lemma exec_fuel_unfold : forall f q pc s,
  exec_fuel (S f) q pc s =
  match nth_error q pc with
  | None => Some s
  | Some (_, BOp i) => exec_fuel f q (S pc) (step i s)
  | Some (_, BBra l) =>
      match find_label l q with
      | Some pc' => exec_fuel f q pc' s | None => None end
  | Some (_, BBeq rd rs l) =>
      if Z.eqb (regs s rd) (regs s rs)
      then match find_label l q with
           | Some pc' => exec_fuel f q pc' s | None => None end
      else exec_fuel f q (S pc) s
  | Some (_, BBne rd rs l) =>
      if Z.eqb (regs s rd) (regs s rs)
      then exec_fuel f q (S pc) s
      else match find_label l q with
           | Some pc' => exec_fuel f q pc' s | None => None end
  end.
Proof. reflexivity. Qed.

Section DeletePair.

Variable p1 : lprog.
Variable lo : option label.
Variable a b : instr.
Variable p2 : lprog.
Hypothesis Hab  : forall s, step b (step a s) = s.
Hypothesis Hok  : mergeable lo p2.

Let n := length p1.
Let P  : lprog := p1 ++ (lo, BOp a) :: (None, BOp b) :: p2.
Let P' : lprog := p1 ++ merge lo p2.

Lemma merge_len : length (merge lo p2) = length p2.
Proof. destruct lo as [l|]; [destruct p2 as [| [[l'|] b'] t]|]; reflexivity. Qed.

Lemma P'_len : length P' = (length P - 2)%nat.
Proof.
  unfold P, P'; rewrite !length_app, merge_len; simpl; lia.
Qed.

(** Instruction correspondence (labels play no role in execution). *)
Lemma instr_at_lt : forall pc, (pc < n)%nat ->
  nth_error P' pc = nth_error P pc.
Proof.
  intros pc H; unfold P, P'.
  rewrite !nth_error_app1 by (unfold n in H; lia). reflexivity.
Qed.

Lemma instr_at_ge : forall pc, (n + 2 <= pc)%nat ->
  option_map snd (nth_error P' (pc - 2)) = option_map snd (nth_error P pc).
Proof.
  intros pc H; unfold P, P'.
  rewrite nth_error_app2 by (unfold n in *; lia).
  rewrite nth_error_app2 by (unfold n in *; lia).
  replace (pc - length p1)%nat with (S (S (pc - 2 - length p1))) by (unfold n in *; lia).
  simpl.
  destruct lo as [l|]; [destruct p2 as [| [[l'|] b'] t]; try contradiction |].
  - (* Some l, p2 = (None,b')::t *)
    destruct (pc - 2 - length p1)%nat as [| k] eqn:Hk; simpl; reflexivity.
  - reflexivity.
Qed.

(** Label correspondence: a label found in [P] at position k is found in [P']
    at [phi n k], and k is never the pair's second line. *)
Lemma find_label_P : forall l k,
  find_label l P = Some k ->
  (k <> S n /\ find_label l P' = Some (phi n k)).
Proof.
  intros l k.
  unfold P, P'.
  revert k; induction p1 as [| [lo1 b1] t1 IH]; intros k; simpl.
  - (* the interesting part: the middle block *)
    destruct lo as [l0|]; [destruct p2 as [| [[l'|] b'] t]; try contradiction |].
    + (* forwarding case: pair labeled l0, successor unlabeled *)
      simpl. destruct (Nat.eqb l l0) eqn:El.
      * intro H; injection H as <-. split; [unfold n; simpl; lia |].
        unfold phi, n; simpl. reflexivity.
      * (* skip pair (2 lines) in P; skip merged head (1 line) in P' *)
        destruct (find_label l t) as [j|] eqn:Hj; simpl; [| discriminate].
        intro H; injection H as <-. split; [unfold n; simpl; lia |].
        unfold phi, n; simpl.
        destruct (Nat.leb (S (S j)) 0) eqn:E; [apply Nat.leb_le in E; lia |].
        do 2 f_equal; lia.
    + (* unlabeled pair *)
      simpl.
      destruct (find_label l p2) as [j|] eqn:Hj; simpl; [| discriminate].
      intro H; injection H as <-. split; [unfold n; simpl; lia |].
      unfold phi, n; simpl.
      destruct (Nat.leb (S (S j)) 0) eqn:E; [apply Nat.leb_le in E; lia |].
      do 2 f_equal; lia.
  - (* head of p1 *)
    destruct lo1 as [l1|].
    + destruct (Nat.eqb l l1) eqn:El.
      * intro H; injection H as <-. split; [unfold n; simpl; lia |].
        unfold phi; simpl. reflexivity.
      * destruct (find_label l (t1 ++ (lo, BOp a) :: (None, BOp b) :: p2))
          as [j|] eqn:Hj; simpl; [| discriminate].
        intro H; injection H as <-.
        destruct (IH j eq_refl) as [Hne Hfind].
        split.
        { unfold n in *; simpl in *; lia. }
        rewrite Hfind. unfold phi, n in *; simpl.
        destruct (Nat.leb j (length t1)) eqn:E1.
        -- apply Nat.leb_le in E1.
           destruct (Nat.leb (S j) (S (length t1))) eqn:E2;
             [| apply Nat.leb_gt in E2; lia].
           reflexivity.
        -- apply Nat.leb_gt in E1.
           destruct (Nat.leb (S j) (S (length t1))) eqn:E2;
             [apply Nat.leb_le in E2; lia |].
           simpl in Hne. do 2 f_equal; lia.
    + destruct (find_label l (t1 ++ (lo, BOp a) :: (None, BOp b) :: p2))
        as [j|] eqn:Hj; simpl; [| discriminate].
      intro H; injection H as <-.
      destruct (IH j eq_refl) as [Hne Hfind].
      split.
      { unfold n in *; simpl in *; lia. }
      rewrite Hfind. unfold phi, n in *; simpl.
      destruct (Nat.leb j (length t1)) eqn:E1.
      -- apply Nat.leb_le in E1.
         destruct (Nat.leb (S j) (S (length t1))) eqn:E2;
           [| apply Nat.leb_gt in E2; lia].
         reflexivity.
      -- apply Nat.leb_gt in E1.
         destruct (Nat.leb (S j) (S (length t1))) eqn:E2;
           [apply Nat.leb_le in E2; lia |].
         simpl in Hne. do 2 f_equal; lia.
Qed.

(** Fuel monotonicity: more fuel never changes a successful run. *)
Lemma exec_mono : forall f f' q pc s res,
  (f <= f')%nat ->
  exec_fuel f q pc s = Some res -> exec_fuel f' q pc s = Some res.
Proof.
  induction f as [| f IH]; intros f' q pc s res Hle H; simpl in H.
  - discriminate.
  - destruct f' as [| f']; [lia |]; simpl.
    destruct (nth_error q pc) as [[l0 b0] |]; [| exact H].
    destruct b0 as [i | l | rd rs l | rd rs l].
    + apply IH; [lia | exact H].
    + destruct (find_label l q); [apply IH; [lia | exact H] | exact H].
    + destruct (Z.eqb (regs s rd) (regs s rs));
        [destruct (find_label l q); [apply IH; [lia | exact H] | exact H]
        | apply IH; [lia | exact H]].
    + destruct (Z.eqb (regs s rd) (regs s rs));
        [apply IH; [lia | exact H]
        | destruct (find_label l q); [apply IH; [lia | exact H] | exact H]].
Qed.

(** The forward simulation. *)
Theorem delete_pair_fwd : forall fuel pc s res,
  pc <> S n ->
  exec_fuel fuel P pc s = Some res ->
  exec_fuel fuel P' (phi n pc) s = Some res.
Proof.
  induction fuel as [fuel IHf] using lt_wf_ind; intros pc s res Hpc H.
  destruct fuel as [| f]; [discriminate H |].
  rewrite exec_fuel_unfold in H.
  destruct (nth_error P pc) as [[lo0 b0] |] eqn:Hnth.
  - (* a line exists at pc *)
    destruct (Nat.eq_dec pc n) as [-> | Hne].
    + (* the pair: two no-op steps, then continue at n+2 ~ phi = n *)
      assert (Ha : nth_error P n = Some (lo, BOp a))
        by (unfold P, n; rewrite nth_error_app2, Nat.sub_diag by lia; reflexivity).
      rewrite Ha in Hnth; injection Hnth as <- <-.
      cbn beta iota in H.
      (* H : exec_fuel f P (S n) (step a s) = Some res *)
      destruct f as [| f']; [discriminate H |].
      rewrite exec_fuel_unfold in H.
      assert (Hb : nth_error P (S n) = Some (None, BOp b))
        by (unfold P, n; rewrite nth_error_app2 by lia;
            replace (S (length p1) - length p1)%nat with 1%nat by lia; reflexivity).
      rewrite Hb in H; cbn beta iota in H.
      rewrite Hab in H.
      (* H : exec_fuel f' P (S (S n)) s = Some res *)
      assert (Hstep : exec_fuel f' P' (phi n (S (S n))) s = Some res)
        by (apply IHf; [lia | lia | exact H]).
      unfold phi in Hstep.
      destruct (Nat.leb (S (S n)) n) eqn:E; [apply Nat.leb_le in E; lia |].
      replace (S (S n) - 2)%nat with n in Hstep by lia.
      eapply exec_mono with (f := f'); [lia |].
      unfold phi; rewrite Nat.leb_refl. exact Hstep.
    + (* not the pair *)
      assert (Hcase : (pc < n)%nat \/ (n + 2 <= pc)%nat) by lia.
      assert (Hinstr : option_map snd (nth_error P' (phi n pc)) = Some b0).
      { destruct Hcase as [Hlt | Hge].
        - unfold phi; destruct (Nat.leb pc n) eqn:E;
            [| apply Nat.leb_gt in E; lia].
          rewrite instr_at_lt, Hnth by exact Hlt. reflexivity.
        - unfold phi; destruct (Nat.leb pc n) eqn:E;
            [apply Nat.leb_le in E; lia |].
          rewrite instr_at_ge, Hnth by exact Hge. reflexivity. }
      rewrite exec_fuel_unfold.
      destruct (nth_error P' (phi n pc)) as [[lo1 b1] |] eqn:Hnth';
        simpl in Hinstr; [| discriminate].
      injection Hinstr as ->.
      assert (Hsucc : phi n (S pc) = S (phi n pc)).
      { unfold phi.
        destruct (Nat.leb (S pc) n) eqn:E1; destruct (Nat.leb pc n) eqn:E2;
        [ apply Nat.leb_le in E1; apply Nat.leb_le in E2
        | apply Nat.leb_le in E1; apply Nat.leb_gt in E2
        | apply Nat.leb_gt in E1; apply Nat.leb_le in E2
        | apply Nat.leb_gt in E1; apply Nat.leb_gt in E2 ]; lia. }
      destruct b0 as [i | l | rd rs l | rd rs l]; cbn beta iota in H |- *.
      * (* BOp *)
        apply IHf in H; [| lia | lia].
        rewrite Hsucc in H. exact H.
      * (* BBra *)
        destruct (find_label l P) as [k|] eqn:Hf; [| discriminate].
        destruct (find_label_P l k Hf) as [Hk Hf'].
        rewrite Hf'.
        apply IHf; [lia | exact Hk | exact H].
      * (* BBeq *)
        destruct (Z.eqb (regs s rd) (regs s rs)).
        -- destruct (find_label l P) as [k|] eqn:Hf; [| discriminate].
           destruct (find_label_P l k Hf) as [Hk Hf'].
           rewrite Hf'.
           apply IHf; [lia | exact Hk | exact H].
        -- apply IHf in H; [| lia | lia].
           rewrite Hsucc in H. exact H.
      * (* BBne *)
        destruct (Z.eqb (regs s rd) (regs s rs)).
        -- apply IHf in H; [| lia | lia].
           rewrite Hsucc in H. exact H.
        -- destruct (find_label l P) as [k|] eqn:Hf; [| discriminate].
           destruct (find_label_P l k Hf) as [Hk Hf'].
           rewrite Hf'.
           apply IHf; [lia | exact Hk | exact H].
  - (* off the end: halt *)
    injection H as <-.
    assert (Hlen : (length P <= pc)%nat)
      by (apply nth_error_None; exact Hnth).
    assert (HlenP : (n + 2 <= length P)%nat)
      by (unfold P, n; rewrite length_app; simpl; lia).
    rewrite exec_fuel_unfold.
    assert (Hnone : nth_error P' (phi n pc) = None).
    { apply nth_error_None. rewrite P'_len.
      unfold phi; destruct (Nat.leb pc n) eqn:E;
        [apply Nat.leb_le in E; lia | apply Nat.leb_gt in E; lia]. }
    rewrite Hnone. reflexivity.
Qed.

End DeletePair.

(** Instantiate the no-op hypothesis with a genuine cancelling pair. *)
Corollary delete_cancelling_pair_fwd :
  forall p1 lo a b p2 fuel pc s res,
  cancels a b = true ->
  mergeable lo p2 ->
  pc <> S (length p1) ->
  exec_fuel fuel (p1 ++ (lo, BOp a) :: (None, BOp b) :: p2) pc s = Some res ->
  exec_fuel fuel (p1 ++ merge lo p2) (phi (length p1) pc) s = Some res.
Proof.
  intros; eapply delete_pair_fwd; eauto.
  intro s0; now apply cancels_undo.
Qed.

Print Assumptions delete_cancelling_pair_fwd.
