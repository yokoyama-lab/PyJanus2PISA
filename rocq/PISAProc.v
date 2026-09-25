(** * PISAProc.v — the control-flow machine of PISACtl.v plus procedure calls

    `pisa_interp.py` does not return from a procedure through the Pendulum
    [br] mechanism: it keeps a *software call stack* and recognises calls and
    returns by their labels.  This file adds exactly that to the machine of
    PISACtl.v (which it reuses unchanged for every other line):

    - A label [f] is a *procedure name* when [f], [f_top] and [f_bot] are all
      defined (`_proc_names`).  Labels are numbers here, so the naming
      convention is replaced by a table [T : ptable] of triples
      [(f, f_top, f_bot)]; [is_proc T P l] holds when [l] is the first
      component of a triple all of whose labels are defined in [P].
    - `BRA`/`RBRA` at a line labeled [f_bot] (of such a triple) with a
      non-empty call stack is a RETURN: the six lines starting at label [f]
      (the prologue `SUBI r1 1; EXCH r2 r1; SWAPBR r2; NEG r2; EXCH r2 r1;
      ADDI r1 1`) are executed once more as data instructions (`SWAPBR`
      included, which changes [br]), then [(pc, br)] is popped.
    - Otherwise a `BRA`/`RBRA` whose target is a procedure name is a CALL:
      [(pc + 1, br)] is pushed, [br := 0], and control goes to the target.
    - `BRA`/`RBRA` at an [f_bot] line with an empty stack is an error
      (`RETURN without CALL`).
    - Every other line is executed by [PISACtl.cstep] with the stack
      untouched — in particular `RBRA` is still `BRA` (the interpreter's
      direction bit is never used), and the paired-branch logic is that of
      PISACtl.v.

    The order of the tests is the interpreter's: the target label is looked
    up first (a missing label is a `KeyError`), then RETURN, then CALL, then
    the error, then paired/direct branching.  `FINISH` is modelled as in
    PISACtl.v: the fuel executor halts when the pc leaves the program. *)

From Stdlib Require Import ZArith List Lia Bool.
Require Import PISA PISACtl.
Import ListNotations.
Open Scope Z_scope.

(** ** Procedure table *)

Definition ptable := list (label * label * label).   (** [(f, f_top, f_bot)] *)

Definition defined (P : lprog) (l : label) : bool :=
  match find_label l P with Some _ => true | None => false end.

Definition entry_ok (P : lprog) (x : label * label * label) : bool :=
  let '(e, t, b) := x in defined P e && defined P t && defined P b.

(** `target_label in self._proc_names` *)
Definition is_proc (T : ptable) (P : lprog) (l : label) : bool :=
  existsb (fun x => let '(e, _, _) := x in Nat.eqb e l && entry_ok P x) T.

(** `cur_label.endswith('_bot') and cur_label[:-4] in self._proc_names`:
    the procedure whose [f_bot] label the line carries. *)
Fixpoint bot_lookup (T : ptable) (P : lprog) (l : label) : option label :=
  match T with
  | [] => None
  | (e, t, b) :: T' =>
      if Nat.eqb b l && entry_ok P (e, t, b) then Some e else bot_lookup T' P l
  end.

Definition bot_of (T : ptable) (P : lprog) (lo : option label) : option label :=
  match lo with Some l => bot_lookup T P l | None => None end.

(** ** The epilogue: `_exec_data` on the six lines at the procedure label *)

Definition data_step (x : cinstr) (br : Z) (s : state) : option (Z * state) :=
  match x with
  | COp i => Some (br, step i s)
  | CSwapbr rd => Some (regs s rd, mkState (rupd rd br (regs s)) (mem s))
  | _ => None                                   (** "Unknown instruction" *)
  end.

Fixpoint data_run (P : lprog) (pc cnt : nat) (br : Z) (s : state) : option (Z * state) :=
  match cnt with
  | O => Some (br, s)
  | S c =>
      match nth_error P pc with
      | Some (_, x) =>
          match data_step x br s with
          | Some (br', s') => data_run P (S pc) c br' s'
          | None => None
          end
      | None => None
      end
  end.

(** ** One step *)

Definition pstate := (cstate * list (nat * Z))%type.

Definition pstep (T : ptable) (P : lprog) (c : pstate) : option pstate :=
  let '(mkC pc br s, k) := c in
  match nth_error P pc with
  | None => None
  | Some (lo, x) =>
      match bra_target x with
      | None => option_map (fun c' => (c', k)) (cstep P (mkC pc br s))
      | Some l =>
          match find_label l P with
          | None => None
          | Some t =>
              match bot_of T P lo, k with
              | Some e, (ret, br0) :: k' =>                        (* RETURN *)
                  match find_label e P with
                  | Some ep =>
                      match data_run P ep 6 br s with
                      | Some (_, s') => Some (mkC ret br0 s', k')
                      | None => None
                      end
                  | None => None
                  end
              | bo, _ =>
                  if is_proc T P l then Some (mkC t 0 s, (S pc, br) :: k)   (* CALL *)
                  else match bo with
                       | Some _ => None                        (* RETURN without CALL *)
                       | None => option_map (fun c' => (c', k)) (bra_step P pc br s l)
                       end
              end
          end
      end
  end.

(** ** Runs *)

Inductive psteps (T : ptable) (P : lprog) : pstate -> pstate -> Prop :=
| psteps_refl : forall c, psteps T P c c
| psteps_step : forall c c' c'',
    pstep T P c = Some c' -> psteps T P c' c'' -> psteps T P c c''.

Lemma psteps_one : forall T P c c', pstep T P c = Some c' -> psteps T P c c'.
Proof. intros; eapply psteps_step; [eassumption | constructor]. Qed.

Lemma psteps_trans : forall T P c1 c2 c3,
  psteps T P c1 c2 -> psteps T P c2 c3 -> psteps T P c1 c3.
Proof.
  intros T P c1 c2 c3 H; induction H; intro H'; [exact H' |].
  eapply psteps_step; [eassumption | auto].
Qed.

Fixpoint pexec_fuel (fuel : nat) (T : ptable) (P : lprog) (c : pstate) : option pstate :=
  match fuel with
  | O => None
  | S f =>
      match nth_error P (cpc (fst c)) with
      | None => Some c
      | Some _ =>
          match pstep T P c with
          | Some c' => pexec_fuel f T P c'
          | None => None
          end
      end
  end.

Lemma psteps_exec_fuel : forall T P c c',
  psteps T P c c' -> nth_error P (cpc (fst c')) = None ->
  exists f, pexec_fuel f T P c = Some c'.
Proof.
  intros T P c c' H; induction H as [c | c c' c'' Hs Hst IH]; intro Hend.
  - exists 1%nat; simpl; now rewrite Hend.
  - destruct (IH Hend) as [f Hf]. exists (S f); simpl.
    destruct (nth_error P (cpc (fst c))) as [x |] eqn:Hn.
    + now rewrite Hs.
    + destruct c as [[pc br s] k]; simpl in *. rewrite Hn in Hs. discriminate.
Qed.

(** The fuel executor is deterministic: two successful runs agree. *)
Lemma pexec_fuel_det : forall f1 f2 T P c c1 c2,
  pexec_fuel f1 T P c = Some c1 -> pexec_fuel f2 T P c = Some c2 -> c1 = c2.
Proof.
  induction f1 as [| f1 IH]; intros f2 T P c c1 c2 H1 H2; [discriminate |].
  destruct f2 as [| f2]; [discriminate |].
  simpl in H1, H2.
  destruct (nth_error P (cpc (fst c))); [| congruence].
  destruct (pstep T P c); [| discriminate].
  eapply IH; eassumption.
Qed.

(** ** Steps that do not involve the call stack *)

Lemma pstep_lift : forall T (P : lprog) pc br s k lo x c',
  nth_error P pc = Some (lo, x) -> bra_target x = None ->
  cstep P (mkC pc br s) = Some c' ->
  pstep T P (mkC pc br s, k) = Some (c', k).
Proof.
  intros T P pc br s k lo x c' Hn Hx Hc; unfold pstep; rewrite Hn, Hx.
  now rewrite Hc.
Qed.

Lemma pstep_bra : forall T (P : lprog) pc br s k lo l c',
  nth_error P pc = Some (lo, CBra l) ->
  is_proc T P l = false -> bot_of T P lo = None ->
  cstep P (mkC pc br s) = Some c' ->
  pstep T P (mkC pc br s, k) = Some (c', k).
Proof.
  intros T P pc br s k lo l c' Hn Hp Hb Hc.
  cbn [cstep] in Hc; rewrite Hn in Hc.
  unfold pstep; rewrite Hn; cbn [bra_target].
  pose proof Hc as Hc'. unfold bra_step in Hc'.
  destruct (find_label l P) as [t |] eqn:Hf; [| discriminate].
  rewrite Hb, Hp. destruct k as [| [ret br0] k']; now rewrite Hc.
Qed.

Lemma pstep_op : forall T (P : lprog) pc br s k lo i,
  nth_error P pc = Some (lo, COp i) ->
  pstep T P (mkC pc br s, k) = Some (mkC (S pc) br (step i s), k).
Proof.
  intros; eapply pstep_lift; [eassumption | reflexivity | now apply cstep_op with lo].
Qed.

Lemma pstep_swapbr : forall T (P : lprog) pc br s k lo rd,
  nth_error P pc = Some (lo, CSwapbr rd) ->
  pstep T P (mkC pc br s, k)
  = Some (mkC (S pc) (regs s rd) (mkState (rupd rd br (regs s)) (mem s)), k).
Proof.
  intros T P pc br s k lo rd H.
  eapply pstep_lift; [exact H | reflexivity |]. cbn [cstep]. now rewrite H.
Qed.

(** A CALL: push the return address and the caller's [br], clear [br]. *)
Lemma pstep_call : forall T (P : lprog) pc br s k lo l t,
  nth_error P pc = Some (lo, CBra l) -> find_label l P = Some t ->
  bot_of T P lo = None -> is_proc T P l = true ->
  pstep T P (mkC pc br s, k) = Some (mkC t 0 s, (S pc, br) :: k).
Proof.
  intros T P pc br s k lo l t Hn Hf Hb Hp; unfold pstep; rewrite Hn; cbn [bra_target].
  rewrite Hf, Hb, Hp. now destruct k as [| [ret br0] k'].
Qed.

(** A RETURN: the epilogue, then pop. *)
Lemma pstep_return : forall T (P : lprog) pc br s ret br0 k lb l t e ep br' s',
  nth_error P pc = Some (Some lb, CBra l) -> find_label l P = Some t ->
  bot_of T P (Some lb) = Some e -> find_label e P = Some ep ->
  data_run P ep 6 br s = Some (br', s') ->
  pstep T P (mkC pc br s, (ret, br0) :: k) = Some (mkC ret br0 s', k).
Proof.
  intros T P pc br s ret br0 k lb l t e ep br' s' Hn Hf Hb He Hd; unfold pstep; rewrite Hn.
  cbn [bra_target]. rewrite Hf, Hb, He, Hd. reflexivity.
Qed.

(** Straight-line segments, with any labels and any stack. *)
Lemma psteps_ops_gen : forall T c P pre p post br s k,
  P = pre ++ p ++ post -> map snd p = map COp c ->
  psteps T P (mkC (length pre) br s, k) (mkC (length pre + length c) br (run c s), k).
Proof.
  intros T; induction c as [| i c IH]; intros P pre p post br s k HP Hp.
  - rewrite Nat.add_0_r. constructor.
  - destruct p as [| [lo x] p']; [discriminate |].
    simpl in Hp; injection Hp as Hx Hp'. subst x.
    eapply psteps_step.
    + apply (pstep_op T P (length pre) br s k lo i).
      eapply nth_error_at. rewrite HP. reflexivity.
    + rewrite run_cons.
      replace (S (length pre)) with (length (pre ++ [(lo, COp i)]))
        by (rewrite length_app; simpl; lia).
      replace (length pre + length (i :: c))%nat
        with (length (pre ++ [(lo, COp i)]) + length c)%nat
        by (rewrite length_app; simpl; lia).
      apply (IH P (pre ++ [(lo, COp i)]) p' post br (step i s) k).
      * rewrite HP, <- app_assoc. reflexivity.
      * exact Hp'.
Qed.

Corollary psteps_ops : forall T c P pre post br s k,
  P = pre ++ ops c ++ post ->
  psteps T P (mkC (length pre) br s, k) (mkC (length pre + length c) br (run c s), k).
Proof. intros; eapply psteps_ops_gen; [eassumption | apply snd_ops]. Qed.

Corollary psteps_ops_l : forall T l c P pre post br s k,
  P = pre ++ ops_l l c ++ post ->
  psteps T P (mkC (length pre) br s, k) (mkC (length pre + length c) br (run c s), k).
Proof. intros; eapply psteps_ops_gen; [eassumption | apply snd_ops_l]. Qed.

(** ** Facts about the table *)

Lemma is_proc_true : forall T P l t b,
  In (l, t, b) T -> entry_ok P (l, t, b) = true -> is_proc T P l = true.
Proof.
  intros T P l t b Hin Hok; unfold is_proc; apply existsb_exists.
  exists (l, t, b); split; [exact Hin |]. now rewrite Nat.eqb_refl, Hok.
Qed.

Lemma is_proc_false : forall T P l,
  (forall e t b, In (e, t, b) T -> e <> l) -> is_proc T P l = false.
Proof.
  intros T P l H; unfold is_proc.
  apply not_true_is_false; intro Hx; apply existsb_exists in Hx.
  destruct Hx as [[[e t] b] [Hin Hx]].
  apply andb_true_iff in Hx as [Hx _]. apply Nat.eqb_eq in Hx.
  exact (H e t b Hin Hx).
Qed.

Lemma bot_lookup_none : forall T P l,
  (forall e t b, In (e, t, b) T -> b <> l) -> bot_lookup T P l = None.
Proof.
  induction T as [| [[e t] b] T IH]; intros P l H; [reflexivity |].
  cbn [bot_lookup]. destruct (Nat.eqb_spec b l) as [-> | Hne].
  - exfalso; eapply H; [now left | reflexivity].
  - cbn [andb]. apply IH. intros; eapply H; right; eassumption.
Qed.

Lemma bot_lookup_some : forall T P l e0 t0,
  In (e0, t0, l) T ->
  (forall e t, In (e, t, l) T -> e = e0 /\ entry_ok P (e, t, l) = true) ->
  bot_lookup T P l = Some e0.
Proof.
  induction T as [| [[e t] b] T IH]; intros P l e0 t0 Hin H; [destruct Hin |].
  cbn [bot_lookup]. destruct (Nat.eqb_spec b l) as [-> | Hne].
  - destruct (H e t (or_introl eq_refl)) as [-> Hok]. now rewrite Hok.
  - cbn [andb]. destruct Hin as [Heq | Hin]; [injection Heq; intros; subst; contradiction |].
    eapply IH; [exact Hin |]. intros; apply H; now right.
Qed.

(** ** Sanity checks *)

(** `f` adds 1 to r3 and returns; `main` (the unlabeled code after
    `start`) calls it twice.  Lines: 0 f_top, 1 f, …, 7 body, 8 f_bot,
    9 start. *)
Definition call_demo : lprog :=
  [ (Some 2%nat, CBra 3%nat)                 (* f_top: BRA f_bot *)
  ; (Some 1%nat, COp (ISubi 1%nat 1))        (* f:     SUBI r1 1 *)
  ; (None, COp (IExch 2%nat 1%nat))
  ; (None, CSwapbr 2%nat)
  ; (None, COp (INeg 2%nat))
  ; (None, COp (IExch 2%nat 1%nat))
  ; (None, COp (IAddi 1%nat 1))
  ; (None, COp (IAddi 3%nat 1))              (*        body *)
  ; (Some 3%nat, CBra 2%nat)                 (* f_bot: BRA f_top *)
  ; (None, COp (IAddi 1%nat 10))             (* start: ADDI r1 10 *)
  ; (None, CBra 1%nat)                       (*        BRA f *)
  ; (None, CBra 1%nat) ].                    (*        BRA f *)

Example ex_call_demo :
  match pexec_fuel 100 [(1%nat, 2%nat, 3%nat)] call_demo (mkC 9 0 zero_state, []) with
  | Some (c, k) => (cpc c, cbr c, regs (cst c) 1%nat, regs (cst c) 2%nat,
                    regs (cst c) 3%nat, mem (cst c) 9, length k)
  | None => (0%nat, 1, 0, 0, 0, 0, 1%nat)
  end = (12%nat, 0, 10, 0, 2, 0, 0%nat).
Proof. reflexivity. Qed.

(** Falling into `f_bot` with an empty stack is the interpreter's
    `RETURN without CALL`. *)
Example ex_return_without_call :
  pexec_fuel 100 [(1%nat, 2%nat, 3%nat)] call_demo (mkC 1 0 zero_state, []) = None.
Proof. reflexivity. Qed.
