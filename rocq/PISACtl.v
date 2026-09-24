(** * PISACtl.v — PISA with control flow: pc, the [br] register, paired branches

    [PISA.v] models straight-line code: a state is a register file and a
    memory, and a block is run by folding [step].  This file adds the control
    flow that `if` (and later `from/loop/until`) compile to, as a separate
    module so that PISA.v itself is untouched: a *labeled* program, a program
    counter, and the Pendulum branch register [br].

    The semantics is that of `pisa_interp.py`, the interpreter the Python
    compiler is tested against, restricted to what a procedure body uses (no
    software call stack, no START/FINISH/DATA).  In particular:

    - A `BRA`/`RBRA` at position [a] is *paired* when its target line [t]
      branches back to [a] — either an unconditional branch to [a], or a
      conditional branch whose label is [a] (`_detect_paired_branches`).
    - A paired branch does [br += t - a]; if [br] becomes nonzero the pc
      moves by [br] (landing on the partner), otherwise it falls through.
      So the second branch of a pair *consumes* the [br] its partner set and
      execution continues after it.
    - A conditional branch with [br = 0] is a direct jump.  With [br <> 0]
      (arrived at via a paired branch) a taken branch does [br += t - pc];
      when that cancels to 0 the pc continues *after the partner* ([S t]).
    - `RBRA` is executed exactly like `BRA` (the interpreter tracks a
      direction bit but never uses it), and `SWAPBR` exchanges a register
      with [br] and advances the pc.

    Everything else — data instructions — is delegated to [PISA.step].
    See MANIFEST.md for what is not modelled. *)

From Stdlib Require Import ZArith List Lia Bool.
Require Import PISA.
Import ListNotations.
Open Scope Z_scope.

Definition label := nat.

(** ** Labeled programs *)

Inductive cinstr : Type :=
| COp     (i : instr)                     (** any straight-line instruction *)
| CBra    (l : label)
| CRbra   (l : label)
| CBeq    (rd rs : reg) (l : label)
| CBne    (rd rs : reg) (l : label)
| CBgez   (rd : reg) (l : label)
| CSwapbr (rd : reg).

Definition line  := (option label * cinstr)%type.
Definition lprog := list line.

Fixpoint find_label (l : label) (p : lprog) : option nat :=
  match p with
  | [] => None
  | (Some l', _) :: t =>
      if Nat.eqb l l' then Some 0%nat
      else option_map S (find_label l t)
  | (None, _) :: t => option_map S (find_label l t)
  end.

(** The labels a program defines, in order. *)
Fixpoint labels (p : lprog) : list label :=
  match p with
  | [] => []
  | (Some l, _) :: t => l :: labels t
  | (None, _) :: t => labels t
  end.

(** Straight-line code as unlabeled lines, and with a label on its first line. *)
Definition ops (c : code) : lprog := map (fun i => (None, COp i)) c.

Definition ops_l (l : label) (c : code) : lprog :=
  match c with
  | [] => []
  | i :: t => (Some l, COp i) :: ops t
  end.

(** ** Machine state *)

Record cstate := mkC { cpc : nat; cbr : Z; cst : state }.

(** [pc + d], undefined when negative (`PC out of range` in the interpreter). *)
Definition jump (pc : nat) (d : Z) : option nat :=
  let z := Z.of_nat pc + d in
  if z <? 0 then None else Some (Z.to_nat z).

(** ** Paired-branch detection (`_detect_paired_branches`) *)

Definition bra_target (b : cinstr) : option label :=
  match b with CBra l | CRbra l => Some l | _ => None end.

Definition cond_target (b : cinstr) : option label :=
  match b with CBeq _ _ l | CBne _ _ l | CBgez _ l => Some l | _ => None end.

(** Does line [b] branch (unconditionally or conditionally) to position [a]? *)
Definition points_back (p : lprog) (b : cinstr) (a : nat) : bool :=
  match bra_target b with
  | Some l' =>
      match find_label l' p with Some a' => Nat.eqb a' a | None => false end
  | None =>
      match cond_target b with
      | Some l' =>
          match find_label l' p with Some a' => Nat.eqb a' a | None => false end
      | None => false
      end
  end.

Definition paired (p : lprog) (a : nat) : bool :=
  match nth_error p a with
  | Some (_, b) =>
      match bra_target b with
      | Some l =>
          match find_label l p with
          | Some t =>
              match nth_error p t with
              | Some (_, b') => points_back p b' a
              | None => false
              end
          | None => false
          end
      | None => false
      end
  | None => false
  end.

(** ** One step *)

Definition bra_step (p : lprog) (pc : nat) (br : Z) (s : state) (l : label)
  : option cstate :=
  match find_label l p with
  | None => None
  | Some t =>
      if paired p pc then
        let br' := br + (Z.of_nat t - Z.of_nat pc) in
        if br' =? 0 then Some (mkC (S pc) 0 s)
        else match jump pc br' with
             | Some pc' => Some (mkC pc' br' s)
             | None => None
             end
      else Some (mkC t br s)
  end.

Definition cond_step (p : lprog) (pc : nat) (br : Z) (s : state)
                     (taken : bool) (l : label) : option cstate :=
  if br =? 0 then
    if taken
    then match find_label l p with Some t => Some (mkC t 0 s) | None => None end
    else Some (mkC (S pc) 0 s)
  else if taken then
    match find_label l p with
    | None => None
    | Some t =>
        let br' := br + (Z.of_nat t - Z.of_nat pc) in
        if br' =? 0 then Some (mkC (S t) 0 s)
        else match jump pc br' with
             | Some pc' => Some (mkC pc' br' s)
             | None => None
             end
    end
  else match jump pc br with Some pc' => Some (mkC pc' br s) | None => None end.

Definition cstep (p : lprog) (c : cstate) : option cstate :=
  let '(mkC pc br s) := c in
  match nth_error p pc with
  | None => None
  | Some (_, COp i)       => Some (mkC (S pc) br (step i s))
  | Some (_, CBra l)      => bra_step p pc br s l
  | Some (_, CRbra l)     => bra_step p pc br s l
  | Some (_, CBeq rd rs l) => cond_step p pc br s (regs s rd =? regs s rs) l
  | Some (_, CBne rd rs l) => cond_step p pc br s (negb (regs s rd =? regs s rs)) l
  | Some (_, CBgez rd l)  => cond_step p pc br s (0 <=? regs s rd) l
  | Some (_, CSwapbr rd)  =>
      Some (mkC (S pc) (regs s rd) (mkState (rupd rd br (regs s)) (mem s)))
  end.

(** ** Runs *)

Inductive steps (p : lprog) : cstate -> cstate -> Prop :=
| steps_refl : forall c, steps p c c
| steps_step : forall c c' c'',
    cstep p c = Some c' -> steps p c' c'' -> steps p c c''.

Lemma steps_one : forall p c c', cstep p c = Some c' -> steps p c c'.
Proof. intros; eapply steps_step; [eassumption | constructor]. Qed.

Lemma steps_trans : forall p c1 c2 c3,
  steps p c1 c2 -> steps p c2 c3 -> steps p c1 c3.
Proof.
  intros p c1 c2 c3 H12; induction H12 as [| c c' c'' Hs Hst IH]; intros H23;
    [exact H23 |].
  eapply steps_step; [eassumption | now apply IH].
Qed.

(** The executable reading: run with fuel until the pc falls off the end
    (the fragment's `finish`); a stuck step or missing label is [None]. *)
Fixpoint exec_fuel (fuel : nat) (p : lprog) (c : cstate) : option cstate :=
  match fuel with
  | O => None
  | S f =>
      match nth_error p (cpc c) with
      | None => Some c
      | Some _ =>
          match cstep p c with
          | Some c' => exec_fuel f p c'
          | None => None
          end
      end
  end.

Lemma steps_exec_fuel : forall p c c',
  steps p c c' -> nth_error p (cpc c') = None ->
  exists f, exec_fuel f p c = Some c'.
Proof.
  intros p c c' H; induction H; intro Hend.
  - exists 1%nat; simpl; now rewrite Hend.
  - destruct (IHsteps Hend) as [f Hf].
    exists (S f); simpl.
    destruct (nth_error p (cpc c)) as [x |] eqn:Hn.
    + now rewrite H.
    + destruct c as [pc br s]; simpl in *; rewrite Hn in H; discriminate.
Qed.

(** ** Position and label lemmas *)

Lemma nth_error_at : forall (p1 : lprog) x p2 P,
  P = p1 ++ x :: p2 -> nth_error P (length p1) = Some x.
Proof.
  intros p1 x p2 P ->.
  rewrite nth_error_app2 by apply Nat.le_refl.
  now rewrite Nat.sub_diag.
Qed.

Lemma labels_app : forall p q, labels (p ++ q) = labels p ++ labels q.
Proof.
  induction p as [| [[l|] i] t IH]; intro q; simpl; [reflexivity | now rewrite IH | apply IH].
Qed.

Lemma labels_ops : forall c, labels (ops c) = [].
Proof. induction c; simpl; auto. Qed.

Lemma labels_ops_l : forall l c l', In l' (labels (ops_l l c)) -> l' = l.
Proof.
  intros l [| i t] l' H; simpl in H; [contradiction |].
  rewrite labels_ops in H; simpl in H; intuition.
Qed.

Lemma length_ops : forall c, length (ops c) = length c.
Proof. intros; apply length_map. Qed.

Lemma length_ops_l : forall l c, length (ops_l l c) = length c.
Proof. intros l [| i t]; simpl; [reflexivity | now rewrite length_ops]. Qed.

Lemma snd_ops : forall c, map snd (ops c) = map COp c.
Proof. induction c; simpl; congruence. Qed.

Lemma snd_ops_l : forall l c, map snd (ops_l l c) = map COp c.
Proof. intros l [| i t]; simpl; [reflexivity | now rewrite snd_ops]. Qed.

Lemma find_label_notin : forall l p, ~ In l (labels p) -> find_label l p = None.
Proof.
  intros l p; induction p as [| [[l'|] i] t IH]; simpl; intro H.
  - reflexivity.
  - destruct (Nat.eqb_spec l l') as [->|Hne]; [exfalso; apply H; now left |].
    rewrite IH; [reflexivity | intro; apply H; now right].
  - now rewrite IH.
Qed.

Lemma find_label_app_none : forall l p q,
  find_label l p = None ->
  find_label l (p ++ q) = option_map (fun k => (length p + k)%nat) (find_label l q).
Proof.
  intros l p q; induction p as [| [[l'|] i] t IH]; simpl; intro H.
  - destruct (find_label l q); reflexivity.
  - destruct (Nat.eqb l l'); [discriminate |].
    destruct (find_label l t); [discriminate |].
    rewrite IH by reflexivity. destruct (find_label l q); reflexivity.
  - destruct (find_label l t); [discriminate |].
    rewrite IH by reflexivity. destruct (find_label l q); reflexivity.
Qed.

(** A label sitting right after a prefix that does not define it is found
    at the prefix's length. *)
Lemma find_label_at : forall l p1 i p2 P,
  P = p1 ++ (Some l, i) :: p2 -> ~ In l (labels p1) ->
  find_label l P = Some (length p1).
Proof.
  intros l p1 i p2 P -> Hnot.
  rewrite find_label_app_none by (now apply find_label_notin).
  simpl; rewrite Nat.eqb_refl; simpl. now rewrite Nat.add_0_r.
Qed.

Lemma find_label_ops_l_at : forall l p1 c p2 P,
  P = p1 ++ ops_l l c ++ p2 -> c <> [] -> ~ In l (labels p1) ->
  find_label l P = Some (length p1).
Proof.
  intros l p1 [| i t] p2 P -> Hne Hnot; [contradiction |].
  simpl. eapply find_label_at; [reflexivity | exact Hnot].
Qed.

(** ** Step lemmas, one per situation the compiled code creates *)

Lemma cstep_op : forall (p : lprog) pc br s lo i,
  nth_error p pc = Some (lo, COp i) ->
  cstep p (mkC pc br s) = Some (mkC (S pc) br (step i s)).
Proof. intros; simpl; now rewrite H. Qed.

Lemma cstep_bra_direct : forall (p : lprog) pc br s lo l t,
  nth_error p pc = Some (lo, CBra l) -> find_label l p = Some t ->
  paired p pc = false ->
  cstep p (mkC pc br s) = Some (mkC t br s).
Proof. intros; simpl; rewrite H; unfold bra_step; now rewrite H0, H1. Qed.

(** A paired branch whose [br] does not cancel jumps by [br], i.e. lands at
    absolute position [br + t]. *)
Lemma cstep_bra_paired_jump : forall (p : lprog) pc br s lo l t,
  nth_error p pc = Some (lo, CBra l) -> find_label l p = Some t ->
  paired p pc = true ->
  br + (Z.of_nat t - Z.of_nat pc) <> 0 -> 0 <= br + Z.of_nat t ->
  cstep p (mkC pc br s)
  = Some (mkC (Z.to_nat (br + Z.of_nat t)) (br + (Z.of_nat t - Z.of_nat pc)) s).
Proof.
  intros p pc br s lo l t Hn Hf Hp Hnz Hge; simpl; rewrite Hn; unfold bra_step.
  rewrite Hf, Hp.
  destruct (Z.eqb_spec (br + (Z.of_nat t - Z.of_nat pc)) 0); [contradiction |].
  unfold jump.
  destruct (Z.ltb_spec (Z.of_nat pc + (br + (Z.of_nat t - Z.of_nat pc))) 0);
    [lia |].
  do 3 f_equal. lia.
Qed.

(** A paired branch whose [br] cancels falls through. *)
Lemma cstep_bra_paired_cancel : forall (p : lprog) pc br s lo l t,
  nth_error p pc = Some (lo, CBra l) -> find_label l p = Some t ->
  paired p pc = true ->
  br + (Z.of_nat t - Z.of_nat pc) = 0 ->
  cstep p (mkC pc br s) = Some (mkC (S pc) 0 s).
Proof.
  intros p pc br s lo l t Hn Hf Hp Hz; simpl; rewrite Hn; unfold bra_step.
  rewrite Hf, Hp, Hz. reflexivity.
Qed.

Lemma cstep_beq_direct_taken : forall (p : lprog) pc s lo rd rs l t,
  nth_error p pc = Some (lo, CBeq rd rs l) -> find_label l p = Some t ->
  regs s rd = regs s rs ->
  cstep p (mkC pc 0 s) = Some (mkC t 0 s).
Proof.
  intros p pc s lo rd rs l t Hn Hf He; simpl; rewrite Hn; unfold cond_step.
  rewrite He, !Z.eqb_refl; simpl. now rewrite Hf.
Qed.

Lemma cstep_beq_direct_not_taken : forall (p : lprog) pc s lo rd rs l,
  nth_error p pc = Some (lo, CBeq rd rs l) ->
  regs s rd <> regs s rs ->
  cstep p (mkC pc 0 s) = Some (mkC (S pc) 0 s).
Proof.
  intros p pc s lo rd rs l Hn He; simpl; rewrite Hn; unfold cond_step.
  destruct (Z.eqb_spec (regs s rd) (regs s rs)); [contradiction | reflexivity].
Qed.

Lemma cstep_bne_direct_taken : forall (p : lprog) pc s lo rd rs l t,
  nth_error p pc = Some (lo, CBne rd rs l) -> find_label l p = Some t ->
  regs s rd <> regs s rs ->
  cstep p (mkC pc 0 s) = Some (mkC t 0 s).
Proof.
  intros p pc s lo rd rs l t Hn Hf He; simpl; rewrite Hn; unfold cond_step.
  destruct (Z.eqb_spec (regs s rd) (regs s rs)); [contradiction |].
  simpl. now rewrite Hf.
Qed.

(** A conditional branch that is not taken needs no label: the target of
    `BNE rt r0 finish` may be anywhere (or nowhere) when it falls through. *)
Lemma cstep_bne_direct_not_taken : forall (p : lprog) pc s lo rd rs l,
  nth_error p pc = Some (lo, CBne rd rs l) ->
  regs s rd = regs s rs ->
  cstep p (mkC pc 0 s) = Some (mkC (S pc) 0 s).
Proof.
  intros p pc s lo rd rs l Hn He; simpl; rewrite Hn; unfold cond_step.
  rewrite He, !Z.eqb_refl. reflexivity.
Qed.

(** The conditional partner of a paired branch: [br] cancels and execution
    resumes after the branch that sent us here. *)
Lemma cstep_beq_cancel : forall (p : lprog) pc br s lo rd rs l t,
  nth_error p pc = Some (lo, CBeq rd rs l) -> find_label l p = Some t ->
  regs s rd = regs s rs -> br <> 0 ->
  br + (Z.of_nat t - Z.of_nat pc) = 0 ->
  cstep p (mkC pc br s) = Some (mkC (S t) 0 s).
Proof.
  intros p pc br s lo rd rs l t Hn Hf He Hnz Hz; simpl; rewrite Hn; unfold cond_step.
  destruct (Z.eqb_spec br 0); [contradiction |].
  rewrite He, !Z.eqb_refl, Hf, Hz. reflexivity.
Qed.

(** ** Straight-line segments behave like [PISA.run]

    Labels on the lines play no role in execution, hence [map snd]. *)
Lemma steps_ops_gen : forall c P pre p post br s,
  P = pre ++ p ++ post -> map snd p = map COp c ->
  steps P (mkC (length pre) br s) (mkC (length pre + length c) br (run c s)).
Proof.
  induction c as [| i c IH]; intros P pre p post br s HP Hp.
  - destruct p; [| discriminate]. simpl. rewrite Nat.add_0_r. constructor.
  - destruct p as [| [lo x] p']; [discriminate |].
    simpl in Hp; injection Hp as Hx Hp'. subst x.
    eapply steps_step.
    + apply (cstep_op P (length pre) br s lo i).
      eapply nth_error_at. rewrite HP. reflexivity.
    + rewrite run_cons.
      replace (S (length pre)) with (length (pre ++ [(lo, COp i)]))
        by (rewrite length_app; simpl; lia).
      replace (length pre + length (i :: c))%nat
        with (length (pre ++ [(lo, COp i)]) + length c)%nat
        by (rewrite length_app; simpl; lia).
      apply (IH P (pre ++ [(lo, COp i)]) p' post br (step i s)).
      * rewrite HP, <- app_assoc. reflexivity.
      * exact Hp'.
Qed.

Corollary steps_ops : forall c P pre post br s,
  P = pre ++ ops c ++ post ->
  steps P (mkC (length pre) br s) (mkC (length pre + length c) br (run c s)).
Proof. intros; eapply steps_ops_gen; [eassumption | apply snd_ops]. Qed.

Corollary steps_ops_l : forall l c P pre post br s,
  P = pre ++ ops_l l c ++ post ->
  steps P (mkC (length pre) br s) (mkC (length pre + length c) br (run c s)).
Proof. intros; eapply steps_ops_gen; [eassumption | apply snd_ops_l]. Qed.

(** ** Deciding [paired] for the shapes the compiler emits *)

Lemma paired_bra_bra : forall (p : lprog) a b lo lo' l l',
  nth_error p a = Some (lo, CBra l) -> find_label l p = Some b ->
  nth_error p b = Some (lo', CBra l') -> find_label l' p = Some a ->
  paired p a = true.
Proof.
  intros p a b lo lo' l l' Ha Hl Hb Hl'; unfold paired.
  rewrite Ha; simpl; rewrite Hl, Hb; unfold points_back; simpl.
  rewrite Hl'. apply Nat.eqb_refl.
Qed.

Lemma paired_bra_beq : forall (p : lprog) a b lo lo' l rd rs l',
  nth_error p a = Some (lo, CBra l) -> find_label l p = Some b ->
  nth_error p b = Some (lo', CBeq rd rs l') -> find_label l' p = Some a ->
  paired p a = true.
Proof.
  intros p a b lo lo' l rd rs l' Ha Hl Hb Hl'; unfold paired.
  rewrite Ha; simpl; rewrite Hl, Hb; unfold points_back; simpl.
  rewrite Hl'. apply Nat.eqb_refl.
Qed.

Lemma paired_bra_op : forall (p : lprog) a b lo lo' l i,
  nth_error p a = Some (lo, CBra l) -> find_label l p = Some b ->
  nth_error p b = Some (lo', COp i) ->
  paired p a = false.
Proof.
  intros p a b lo lo' l i Ha Hl Hb; unfold paired.
  rewrite Ha; simpl; rewrite Hl, Hb. reflexivity.
Qed.

(** ** Sanity checks

    A minimal pair: `0: BRA 1 / r3 += 1 / 1: BRA 0 / r4 += 1`.  The first
    branch sets [br = 2] and lands on its partner, which cancels [br] and
    falls through, so [r3 += 1] is skipped and [r4 += 1] runs. *)
Definition pair_prog : lprog :=
  [ (Some 0%nat, CBra 1%nat)
  ; (None, COp (IAddi 3%nat 1))
  ; (Some 1%nat, CBra 0%nat)
  ; (None, COp (IAddi 4%nat 1)) ].

Example ex_pair_detected : paired pair_prog 0 = true /\ paired pair_prog 2 = true.
Proof. split; reflexivity. Qed.

Example ex_pair_run :
  match exec_fuel 10 pair_prog (mkC 0 0 zero_state) with
  | Some c => (cpc c, cbr c, regs (cst c) 3%nat, regs (cst c) 4%nat)
  | None => (0%nat, 1, 1, 1)
  end = (4%nat, 0, 0, 1).
Proof. reflexivity. Qed.

(** An unpaired branch is a plain jump: here the target is a data line. *)
Example ex_direct_bra :
  match exec_fuel 10 [ (None, CBra 5%nat); (None, COp (IAddi 3%nat 1))
                     ; (Some 5%nat, COp (IAddi 4%nat 1)) ]
                  (mkC 0 0 zero_state) with
  | Some c => (cpc c, regs (cst c) 3%nat, regs (cst c) 4%nat)
  | None => (0%nat, 1, 1)
  end = (3%nat, 0, 1).
Proof. reflexivity. Qed.
