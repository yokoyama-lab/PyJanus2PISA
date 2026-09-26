(** * CompileProc.v — compiling [call f] / [uncall f], and its correctness

    The compiler of CompileLoop.v extended with procedure calls, proved
    correct on the machine of PISAProc.v (PISACtl.v plus the software call
    stack of `pisa_interp.py`).  The emitted code is `codegen.py`'s for a
    program whose procedures are **not inlined** (see "Scope" below):

<<
   f_top:  BRA f_bot                  -- gen_proc, one per procedure
   f:      SUBI r1 1                  -- prologue
           EXCH r2 r1
           SWAPBR r2
           NEG r2
           EXCH r2 r1
           ADDI r1 1
           <body of f>                -- at scratch base r3, labels fresh
   f_bot:  BRA f_top
   ...
   f_inv_top: ...                     -- the inverted companion (gen_program
   f_inv:     ...                        step 3b): body = invert(body of f)
   ...
   start:  START
           ADDI r1 k                  -- the stack pointer
           BRA main
   finish: FINISH
>>

    and `call f` is `BRA f`, `uncall f` is `BRA f_inv` (`_gen_call`,
    `_gen_uncall`).  So `uncall` is realised by a second, inverted copy of
    the body, executed forwards; `RBRA` is never emitted (and would behave
    like `BRA` in `pisa_interp.py` anyway).  [if]/[from] are `_gen_if` /
    `_gen_from` exactly as in CompileIf.v / CompileLoop.v ([if_code],
    [loop_code] are reused verbatim).

    What the interpreter does at a call, and why the code is clean: the call
    pushes the return address and clears [br]; the prologue, run with
    [br = 0] and the stack cell [M[r1 - 1] = 0], is the identity on the
    whole state (it exchanges [r2] through the cell and back, and the
    [SWAPBR] moves 0 into [br] and 0 out of it — [prologue_id]); the body
    runs; at `f_bot` the interpreter executes the same six instructions once
    more as data and pops.  [r1] is never changed across a statement, so
    every call uses the same cell, which is why the only memory requirement
    is that the cell holds 0 and no statement writes it.

    Main results:
    - [compile_p_spec]: semantic preservation + cleanliness for a compiled
      statement anywhere in a program whose procedures are laid out as
      above ([procs_in]), at any call-stack depth — recursion included (the
      proof is an induction on the derivation, which contains the callee's).
    - [compile_p_program]: the whole program, run by the fuel executor from
      `start`, halts at the end with memory representing the final store,
      the registers of the start state except [r1 = k], and an empty call
      stack.

    Calls may occur anywhere, including the [loop] part (S2) of a [from]
    loop: `_gen_from` runs both S1 and S2 with its flag register at 0, and a
    callee's body assumes r3, r4, … are 0.  (An older `_gen_from` ran S2
    with the flag at 1; that layout miscompiled a call in S2, and a
    restriction in [wf_p] excluded it.)  Inlining is not modelled. *)

From Stdlib Require Import ZArith List Lia Bool.
From Stdlib Require Import FunctionalExtensionality.
Require Import PISA Src Compile PISACtl CompileIf CompileLoop PISAProc SrcProc.
Import ListNotations.
Open Scope Z_scope.

(** ** Labels

    `finish` is [fin_label] = 0.  Procedure [f]'s labels `f`, `f_top`,
    `f_bot` are [pe f false], [pt f false], [pbt f false]; those of
    `f_inv` are [pe f true], … .  They are all below [L0 Γ]; statement
    labels are allocated from [L0 Γ] upwards.  (In `codegen.py` all of these
    are strings in one namespace — see MANIFEST.md for what that costs.) *)

Definition pe (f : pname) (d : bool) : label := (1 + 6 * f + (if d then 3 else 0))%nat.
Definition pt (f : pname) (d : bool) : label := S (pe f d).
Definition pbt (f : pname) (d : bool) : label := S (S (pe f d)).
Definition L0 (Γ : penv) : label := (1 + 6 * length Γ)%nat.

Definition dirb (d : bool) (st : pstmt) : pstmt := if d then invert_p st else st.

(** ** The compiler *)

Fixpoint compile_p (st : pstmt) (b : reg) (n : label) : lprog * label :=
  match st with
  | PBase s => (ops (compile_at b s), n)
  | PSeq a c =>
      let '(p1, n1) := compile_p a b n in
      let '(p2, n2) := compile_p c b n1 in
      (p1 ++ p2, n2)
  | PIf e1 a c e2 =>
      let '(pa, na) := compile_p a (S b) (n + 5)%nat in
      let '(pc, nc) := compile_p c (S b) na in
      (if_code fin_label b n e1 e2 pa pc, nc)
  | PLoop e1 a c e2 =>
      let '(pa, na) := compile_p a (S b) (n + 4)%nat in
      let '(pc, nc) := compile_p c (S b) na in
      (loop_code fin_label b n e1 e2 pa pc, nc)
  | PCall f   => ([(None, CBra (pe f false))], n)
  | PUncall f => ([(None, CBra (pe f true))], n)
  end.

(** `gen_proc` *)
Definition proc_code (e t bt : label) (B : lprog) : lprog :=
  (Some t, CBra bt)
  :: (Some e, COp (ISubi 1%nat 1))
  :: (None, COp (IExch 2%nat 1%nat))
  :: (None, CSwapbr 2%nat)
  :: (None, COp (INeg 2%nat))
  :: (None, COp (IExch 2%nat 1%nat))
  :: (None, COp (IAddi 1%nat 1))
  :: B ++ [(Some bt, CBra t)].

(** Every procedure, forwards ([d = false]) or inverted ([d = true]). *)
Fixpoint emit_list (d : bool) (f : nat) (bs : list pstmt) (n : label) : lprog * label :=
  match bs with
  | [] => ([], n)
  | body :: bs' =>
      let '(B, n1) := compile_p (dirb d body) scratch n in
      let '(rest, n2) := emit_list d (S f) bs' n1 in
      (proc_code (pe f d) (pt f d) (pbt f d) B ++ rest, n2)
  end.

Definition procs_code (Γ : penv) : lprog :=
  let '(Fw, n1) := emit_list false 0 Γ (L0 Γ) in
  let '(Iv, _) := emit_list true 0 Γ n1 in
  Fw ++ Iv.

(** `start: START; ADDI r1 k; BRA main; finish: FINISH`.  START is a no-op
    and FINISH halts; here the first is a NOP line and the second the last
    line of the program (the fuel executor halts past it). *)
Definition start_code (main : pname) (k : Z) : lprog :=
  [ (None, COp nop); (None, COp (IAddi 1%nat k)); (None, CBra (pe main false))
  ; (Some fin_label, COp nop) ].

Definition whole (Γ : penv) (main : pname) (k : Z) : lprog :=
  procs_code Γ ++ start_code main k.

(** The procedure table `pisa_interp.py` derives from the `_top`/`_bot`
    naming convention. *)
Definition ptab (np : nat) : ptable :=
  flat_map (fun f => [(pe f false, pt f false, pbt f false); (pe f true, pt f true, pbt f true)])
           (seq 0 np).

(** ** Label arithmetic *)

Definition is_pl (i : nat) (d : bool) (l : label) : Prop :=
  l = pe i d \/ l = pt i d \/ l = pbt i d.

Lemma is_pl_bound : forall i d l, is_pl i d l -> (1 <= l < 1 + 6 * S i)%nat.
Proof. unfold is_pl, pt, pbt, pe; intros i [|] l H; lia. Qed.

Lemma pe_inj : forall i d f d', is_pl i d (pe f d') -> i = f /\ d = d'.
Proof. unfold is_pl, pt, pbt, pe; intros i [|] f [|] H; lia. Qed.

Lemma pt_inj : forall i d f d', is_pl i d (pt f d') -> i = f /\ d = d'.
Proof. unfold is_pl, pt, pbt, pe; intros i [|] f [|] H; lia. Qed.

Lemma pbt_inj : forall i d f d', is_pl i d (pbt f d') -> i = f /\ d = d'.
Proof. unfold is_pl, pt, pbt, pe; intros i [|] f [|] H; lia. Qed.

Lemma pl_lt_L0 : forall Γ f d, (f < length Γ)%nat ->
  (pe f d < L0 Γ)%nat /\ (pt f d < L0 Γ)%nat /\ (pbt f d < L0 Γ)%nat.
Proof. unfold L0, pt, pbt, pe; intros Γ f [|] H; lia. Qed.

Lemma in_ptab : forall np e t b, In (e, t, b) (ptab np) ->
  exists f d, (f < np)%nat /\ e = pe f d /\ t = pt f d /\ b = pbt f d.
Proof.
  intros np e t b H; unfold ptab in H; apply in_flat_map in H as [f [Hf H]].
  apply in_seq in Hf. destruct H as [H | [H | []]]; injection H as <- <- <-;
    [exists f, false | exists f, true]; repeat split; lia.
Qed.

Lemma ptab_in : forall np f d, (f < np)%nat -> In (pe f d, pt f d, pbt f d) (ptab np).
Proof.
  intros np f d H; unfold ptab; apply in_flat_map. exists f; split; [apply in_seq; lia |].
  destruct d; simpl; auto.
Qed.

(** ** Generic program lemmas *)

Lemma find_label_nth : forall l (p : lprog) a,
  find_label l p = Some a -> exists x, nth_error p a = Some (Some l, x).
Proof.
  intros l p; induction p as [| [[l'|] x] t IH]; intros a H; simpl in H; [discriminate | |].
  - destruct (Nat.eqb_spec l l') as [-> | Hne].
    + injection H as <-. now exists x.
    + destruct (find_label l t) as [a' |] eqn:E; [| discriminate].
      injection H as <-. simpl. now apply IH.
  - destruct (find_label l t) as [a' |] eqn:E; [| discriminate].
    injection H as <-. simpl. now apply IH.
Qed.

(** A branch on an unlabeled line is never paired: nothing can point back
    to it. *)
Lemma paired_unlabeled : forall (p : lprog) a x,
  nth_error p a = Some (None, x) -> paired p a = false.
Proof.
  intros p a x Ha. unfold paired; rewrite Ha.
  destruct (bra_target x) as [l |]; [| reflexivity].
  destruct (find_label l p) as [t |]; [| reflexivity].
  destruct (nth_error p t) as [[lo' b'] |]; [| reflexivity].
  unfold points_back.
  assert (G : forall l', match find_label l' p with Some a' => Nat.eqb a' a | None => false end
                         = false).
  { intro l'. destruct (find_label l' p) as [a' |] eqn:E; [| reflexivity].
    apply Nat.eqb_neq. intros ->. destruct (find_label_nth _ _ _ E) as [y Hy].
    congruence. }
  destruct (bra_target b'); [apply G |].
  destruct (cond_target b'); [apply G | reflexivity].
Qed.

(** *** A label on the first line *)

Definition relab (lo : option label) (p : lprog) : lprog :=
  match lo with None => p | Some l => label_first l p end.

Definition ops_o (lo : option label) (c : code) : lprog :=
  match c with [] => [] | i :: t => (lo, COp i) :: ops t end.

Lemma length_ops_o : forall lo c, length (ops_o lo c) = length c.
Proof. intros lo [| i t]; simpl; [reflexivity | now rewrite length_ops]. Qed.

Lemma snd_ops_o : forall lo c, map snd (ops_o lo c) = map COp c.
Proof. intros lo [| i t]; simpl; [reflexivity | now rewrite snd_ops]. Qed.

Lemma labels_ops_o : forall lo c l, In l (labels (ops_o lo c)) -> lo = Some l.
Proof.
  intros [l0|] [| i t] l H; cbn [ops_o labels] in H; try rewrite labels_ops in H.
  - destruct H.
  - destruct H as [-> | []]. reflexivity.
  - destruct H.
  - destruct H.
Qed.

Lemma relab_app : forall lo p q, p <> [] -> relab lo (p ++ q) = relab lo p ++ q.
Proof. intros [l|] [| [lo' x] t] q H; try (exfalso; now apply H); reflexivity. Qed.

Lemma relab_ops_app : forall lo c q, c <> [] -> relab lo (ops c ++ q) = ops_o lo c ++ q.
Proof. intros [l|] [| i t] q H; try (exfalso; now apply H); reflexivity. Qed.

Lemma relab_ops : forall lo c, c <> [] -> relab lo (ops c) = ops_o lo c.
Proof. intros lo c H; rewrite <- (app_nil_r (ops c)), relab_ops_app, app_nil_r; auto. Qed.

Lemma ops_o_none : forall c, ops_o None c = ops c.
Proof. intros [| i t]; reflexivity. Qed.

(** [if_code] and [loop_code] with a label on their first line. *)
Definition if_code_o (lo : option label) (fin : label) (b : reg) (n : label) (e1 e2 : expr)
                     (pa pb : lprog) : lprog :=
  ops_o lo (flag_block e1 b)
  ++ (Some (S n), CBeq b 0%nat n)
  :: (None, COp (IXori b 1))
  :: pa
  ++ (None, COp (IXori b 1))
  :: ops_l (n + 2)%nat (flag_block e2 b)
  ++ (Some (n + 3)%nat, CBra (n + 4)%nat)
  :: (Some n, CBra (S n))
  :: pb
  ++ (None, CBra (n + 2)%nat)
  :: (Some (n + 4)%nat, CBra (n + 3)%nat)
  :: (None, CBne b 0%nat fin)
  :: [].

Definition loop_code_o (lo : option label) (fin : label) (b : reg) (n : label) (e1 e2 : expr)
                       (pa pb : lprog) : lprog :=
  ops_o lo (flag_block e1 b)
  ++ (None, COp (IXori b 1))
  :: (None, CBne b 0%nat fin)
  :: label_first (n + 3)%nat pa
  ++ ops_l n (flag_block e2 b)
  ++ (None, CBeq b 0%nat (S n))
  :: (None, COp (IXori b 1))
  :: (None, CBra (n + 2)%nat)
  :: (Some (S n), COp nop)
  :: pb
  ++ ops (flag_block e1 b)
  ++ (None, CBne b 0%nat fin)
  :: (None, CBra (n + 3)%nat)
  :: (Some (n + 2)%nat, COp nop)
  :: [].

Lemma relab_if_code : forall lo fin b n e1 e2 pa pb,
  relab lo (if_code fin b n e1 e2 pa pb) = if_code_o lo fin b n e1 e2 pa pb.
Proof.
  intros; unfold if_code, if_code_o. apply relab_ops_app, flag_block_nonempty.
Qed.

Lemma relab_loop_code : forall lo fin b n e1 e2 pa pb,
  relab lo (loop_code fin b n e1 e2 pa pb) = loop_code_o lo fin b n e1 e2 pa pb.
Proof.
  intros; unfold loop_code, loop_code_o. apply relab_ops_app, flag_block_nonempty.
Qed.

Lemma length_label_first : forall l p, length (label_first l p) = max 1 (length p).
Proof. intros l [| [lo x] t]; simpl; lia. Qed.

(** [in_labels] of CompileIf.v, plus a labeled first line. *)
Ltac in_labels_o H :=
  in_labels H;
  repeat match goal with
  | H' : In _ (labels (ops_o _ _)) |- _ => apply labels_ops_o in H'
  | H' : In _ (labels (label_first _ _)) |- _ =>
      apply labels_label_first in H'; destruct H' as [H' | H']
  | H' : _ \/ _ |- _ => destruct H'
  end.

(** ** Shape facts about the compiler *)

Lemma compile_p_labels : forall st b n p n',
  compile_p st b n = (p, n') ->
  (n <= n')%nat /\ (forall l, In l (labels p) -> (n <= l < n')%nat).
Proof.
  induction st as [s | a IHa c IHc | e1 a IHa c IHc e2 | e1 a IHa c IHc e2 | f | f];
    intros b n p n' Hc; simpl in Hc.
  - injection Hc as <- <-. split; [lia |].
    intros l H; rewrite labels_ops in H; destruct H.
  - destruct (compile_p a b n) as [p1 n1] eqn:E1.
    destruct (compile_p c b n1) as [p2 n2] eqn:E2.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ E1) as [Hle1 Hin1].
    destruct (IHc _ _ _ _ E2) as [Hle2 Hin2].
    split; [lia |].
    intros l H; rewrite labels_app, in_app_iff in H.
    destruct H as [H | H]; [apply Hin1 in H | apply Hin2 in H]; lia.
  - destruct (compile_p a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_p c (S b) na) as [pb nb] eqn:Eb.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ Ea) as [Hlea Hina].
    destruct (IHc _ _ _ _ Eb) as [Hleb Hinb].
    split; [lia |].
    intros l H; apply labels_if_code in H.
    repeat match goal with
    | H : _ \/ _ |- _ => destruct H
    | H : In _ (labels pa) |- _ => apply Hina in H
    | H : In _ (labels pb) |- _ => apply Hinb in H
    end; subst; lia.
  - destruct (compile_p a (S b) (n + 4)%nat) as [pa na] eqn:Ea.
    destruct (compile_p c (S b) na) as [pb nb] eqn:Eb.
    injection Hc as <- <-.
    destruct (IHa _ _ _ _ Ea) as [Hlea Hina].
    destruct (IHc _ _ _ _ Eb) as [Hleb Hinb].
    split; [lia |].
    intros l H; apply labels_loop_code in H.
    repeat match goal with
    | H : _ \/ _ |- _ => destruct H
    | H : In _ (labels pa) |- _ => apply Hina in H
    | H : In _ (labels pb) |- _ => apply Hinb in H
    end; subst; lia.
  - injection Hc as <- <-. split; [lia | intros l []].
  - injection Hc as <- <-. split; [lia | intros l []].
Qed.

Lemma compile_at_nil : forall b s σ σ', compile_at b s = [] -> exec s σ σ' -> σ' = σ.
Proof.
  intros b s σ σ' Hc H; induction H; simpl in Hc.
  - reflexivity.
  - unfold gen_assign_at in Hc. apply (f_equal (@length instr)) in Hc.
    rewrite !length_app in Hc; simpl in Hc; lia.
  - discriminate.
  - apply app_eq_nil in Hc as [H1 H2]. rewrite IHexec2, IHexec1 by assumption. reflexivity.
Qed.

Lemma compile_p_nil : forall Γ st b n n' σ σ',
  compile_p st b n = ([], n') -> exec_p Γ st σ σ' -> σ' = σ.
Proof.
  intros Γ; induction st as [s | a IHa c IHc | e1 a IHa c IHc e2 | e1 a IHa c IHc e2 | f | f];
    intros b n n' σ σ' Hc H; simpl in Hc.
  - injection Hc as Hc _. apply map_eq_nil in Hc.
    inversion H; subst. eapply compile_at_nil; eassumption.
  - destruct (compile_p a b n) as [p1 n1] eqn:E1.
    destruct (compile_p c b n1) as [p2 n2] eqn:E2.
    injection Hc as Hc _. apply app_eq_nil in Hc as [-> ->].
    inversion H; subst.
    erewrite (IHc _ _ _ _ _ E2) by eassumption. eapply IHa; eassumption.
  - destruct (compile_p a (S b) (n + 5)%nat) as [pa na].
    destruct (compile_p c (S b) na) as [pb nb].
    injection Hc as Hc _. unfold if_code in Hc.
    apply (f_equal (@length line)) in Hc. rewrite !length_app in Hc; simpl in Hc; lia.
  - destruct (compile_p a (S b) (n + 4)%nat) as [pa na].
    destruct (compile_p c (S b) na) as [pb nb].
    injection Hc as Hc _. unfold loop_code in Hc.
    apply (f_equal (@length line)) in Hc. rewrite !length_app in Hc; simpl in Hc; lia.
  - discriminate.
  - discriminate.
Qed.

(** ** The [If] layout on the procedure machine

    Section [IfLayout] of CompileIf.v, redone for [psteps] (at any call
    stack [k]) and for a layout whose first line may carry a label [lo]
    (`_gen_from` puts `from_do` there when the [If] is the first statement
    of a loop body).  The branches of the layout are not calls or returns:
    their labels and targets are layout labels, which [Hsafe] says are not
    procedure labels.  The bodies are hypotheses (runs in the same program),
    so calls inside them are the business of the main induction. *)

Section IfLayoutP.

Variable T : ptable.
Variable pre post : lprog.
Variable lo : option label.
Variable b : reg.
Variable n : label.
Variable e1 e2 : expr.
Variable pa pb : lprog.
Variable fin : label.
Variable k : list (nat * Z).

Hypothesis Hb : b <> 0%nat.
Hypothesis Hpre  : forall l, In l (labels pre)  -> ~ (n <= l < n + 5)%nat.
Hypothesis Hpost : forall l, In l (labels post) -> ~ (n <= l < n + 5)%nat.
Hypothesis Hpa   : forall l, In l (labels pa)   -> ~ (n <= l < n + 5)%nat.
Hypothesis Hpb   : forall l, In l (labels pb)   -> ~ (n <= l < n + 5)%nat.
Hypothesis Hlo   : forall l, lo = Some l -> ~ (n <= l < n + 5)%nat.

Let lfalse  := n.
Let ltest   := S n.
Let lassert := (n + 2)%nat.
Let ltrue   := (n + 3)%nat.
Let lend    := (n + 4)%nat.
Let T1 := flag_block e1 b.
Let T2 := flag_block e2 b.

Let P := pre ++ if_code_o lo fin b n e1 e2 pa pb ++ post.

Hypothesis Hsafe : forall l, (n <= l < n + 5)%nat ->
  is_proc T P l = false /\ bot_of T P (Some l) = None.

Let n0 := length pre.
Let t1 := length T1.
Let la := length pa.
Let t2 := length T2.
Let lb := length pb.
Let pB := (n0 + t1)%nat.
Let pA := S (S pB).
Let pX := (pA + la)%nat.
Let pD := S pX.
Let pE := (pD + t2)%nat.
Let pF := S pE.
Let pC := S pF.
Let pG := (pC + lb)%nat.
Let pH := S pG.
Let pN := S pH.

Let preA := pre ++ ops_o lo T1
            ++ [(Some ltest, CBeq b 0%nat lfalse); (None, COp (IXori b 1))].
Let postA := (None, COp (IXori b 1))
             :: ops_l lassert T2
             ++ (Some ltrue, CBra lend)
             :: (Some lfalse, CBra ltest)
             :: pb
             ++ (None, CBra lassert)
             :: (Some lend, CBra ltrue)
             :: (None, CBne b 0%nat fin)
             :: post.
Let preB := pre ++ ops_o lo T1
            ++ (Some ltest, CBeq b 0%nat lfalse)
            :: (None, COp (IXori b 1))
            :: pa
            ++ (None, COp (IXori b 1))
            :: ops_l lassert T2
            ++ [(Some ltrue, CBra lend); (Some lfalse, CBra ltest)].
Let postB := (None, CBra lassert) :: (Some lend, CBra ltrue)
             :: (None, CBne b 0%nat fin) :: post.

Ltac len_solve :=
  unfold pN, pH, pG, pC, pF, pE, pD, pX, pA, pB, lb, t2, la, t1, n0, T1, T2 in *;
  repeat first [ rewrite length_app | rewrite length_ops | rewrite length_ops_l
               | rewrite length_ops_o | progress cbn [length] ];
  lia.

Ltac notin_labels :=
  let H := fresh in intro H; in_labels_o H;
  repeat match goal with
  | H : In _ (labels pre) |- _ => apply Hpre in H
  | H : In _ (labels post) |- _ => apply Hpost in H
  | H : In _ (labels pa) |- _ => apply Hpa in H
  | H : In _ (labels pb) |- _ => apply Hpb in H
  | H : lo = Some _ |- _ => apply Hlo in H
  end;
  unfold lfalse, ltest, lassert, ltrue, lend in *; lia.

Lemma p_len_if_code : length (if_code_o lo fin b n e1 e2 pa pb) = (t1 + la + t2 + lb + 8)%nat.
Proof. unfold if_code_o; len_solve. Qed.

Lemma p_preA_eq : preA ++ pa ++ postA = P.
Proof. unfold P, preA, postA, if_code_o; norm_app; reflexivity. Qed.

Lemma p_preB_eq : preB ++ pb ++ postB = P.
Proof. unfold P, preB, postB, if_code_o; norm_app; reflexivity. Qed.

Lemma p_len_preA : length preA = pA.
Proof. unfold preA; len_solve. Qed.

Lemma p_len_preB : length preB = pC.
Proof. unfold preB; len_solve. Qed.

Lemma p_end_pos : (length pre + length (if_code_o lo fin b n e1 e2 pa pb))%nat = S pN.
Proof. rewrite p_len_if_code; len_solve. Qed.

Lemma pi_nth_B : nth_error P pB = Some (Some ltest, CBeq b 0%nat lfalse).
Proof.
  replace pB with (length (pre ++ ops_o lo T1)) by len_solve.
  eapply nth_error_at. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_nth_X1 : nth_error P (S pB) = Some (None, COp (IXori b 1)).
Proof.
  replace (S pB) with (length (pre ++ ops_o lo T1 ++ [(Some ltest, CBeq b 0%nat lfalse)]))
    by len_solve.
  eapply nth_error_at. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_nth_X2 : nth_error P pX = Some (None, COp (IXori b 1)).
Proof.
  replace pX with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa))
    by len_solve.
  eapply nth_error_at. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_nth_E : nth_error P pE = Some (Some ltrue, CBra lend).
Proof.
  replace pE with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2))
    by len_solve.
  eapply nth_error_at. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_nth_F : nth_error P pF = Some (Some lfalse, CBra ltest).
Proof.
  replace pF with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2 ++ [(Some ltrue, CBra lend)]))
    by len_solve.
  eapply nth_error_at. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_nth_G : nth_error P pG = Some (None, CBra lassert).
Proof.
  replace pG with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2
      ++ (Some ltrue, CBra lend) :: (Some lfalse, CBra ltest) :: pb))
    by len_solve.
  eapply nth_error_at. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_nth_H : nth_error P pH = Some (Some lend, CBra ltrue).
Proof.
  replace pH with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2
      ++ (Some ltrue, CBra lend) :: (Some lfalse, CBra ltest) :: pb
      ++ [(None, CBra lassert)]))
    by len_solve.
  eapply nth_error_at. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_nth_N : nth_error P pN = Some (None, CBne b 0%nat fin).
Proof.
  replace pN with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2
      ++ (Some ltrue, CBra lend) :: (Some lfalse, CBra ltest) :: pb
      ++ [(None, CBra lassert); (Some lend, CBra ltrue)]))
    by len_solve.
  eapply nth_error_at. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_find_ltest : find_label ltest P = Some pB.
Proof.
  replace pB with (length (pre ++ ops_o lo T1)) by len_solve.
  eapply find_label_at; [unfold P, if_code_o; norm_app; reflexivity | notin_labels].
Qed.

Lemma pi_find_lassert : find_label lassert P = Some pD.
Proof.
  replace pD with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))]))
    by len_solve.
  eapply find_label_ops_l_at with (c := T2);
    [unfold P, if_code_o; norm_app; reflexivity | apply flag_block_nonempty | notin_labels].
Qed.

Lemma pi_find_ltrue : find_label ltrue P = Some pE.
Proof.
  replace pE with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2))
    by len_solve.
  eapply find_label_at; [unfold P, if_code_o; norm_app; reflexivity | notin_labels].
Qed.

Lemma pi_find_lfalse : find_label lfalse P = Some pF.
Proof.
  replace pF with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2 ++ [(Some ltrue, CBra lend)]))
    by len_solve.
  eapply find_label_at; [unfold P, if_code_o; norm_app; reflexivity | notin_labels].
Qed.

Lemma pi_find_lend : find_label lend P = Some pH.
Proof.
  replace pH with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ (None, COp (IXori b 1)) :: ops_l lassert T2
      ++ (Some ltrue, CBra lend) :: (Some lfalse, CBra ltest) :: pb
      ++ [(None, CBra lassert)]))
    by len_solve.
  eapply find_label_at; [unfold P, if_code_o; norm_app; reflexivity | notin_labels].
Qed.

Lemma pi_paired_E : paired P pE = true.
Proof.
  eapply paired_bra_bra; [apply pi_nth_E | apply pi_find_lend | apply pi_nth_H | apply pi_find_ltrue].
Qed.

Lemma pi_paired_H : paired P pH = true.
Proof.
  eapply paired_bra_bra; [apply pi_nth_H | apply pi_find_ltrue | apply pi_nth_E | apply pi_find_lend].
Qed.

Lemma pi_paired_F : paired P pF = true.
Proof.
  eapply paired_bra_beq; [apply pi_nth_F | apply pi_find_ltest | apply pi_nth_B | apply pi_find_lfalse].
Qed.

Ltac safe_lbl := unfold lfalse, ltest, lassert, ltrue, lend; lia.

Lemma pi_join_steps : forall s, psteps T P (mkC pE 0 s, k) (mkC pN 0 s, k).
Proof.
  intro s.
  eapply psteps_step.
  { eapply pstep_bra; [apply pi_nth_E | apply Hsafe; safe_lbl | apply Hsafe; safe_lbl |].
    eapply cstep_bra_paired_jump; [apply pi_nth_E | apply pi_find_lend | apply pi_paired_E | | ].
    - unfold pH, pG, pC, pF; lia.
    - lia. }
  rewrite Z.add_0_l, Nat2Z.id.
  apply psteps_one.
  eapply pstep_bra; [apply pi_nth_H | apply Hsafe; safe_lbl | apply Hsafe; safe_lbl |].
  eapply cstep_bra_paired_cancel; [apply pi_nth_H | apply pi_find_ltrue | apply pi_paired_H | lia].
Qed.

Lemma pi_check_pass : forall R M, R b = R 0%nat ->
  psteps T P (mkC pN 0 (mkState R M), k) (mkC (S pN) 0 (mkState R M), k).
Proof.
  intros R M H; apply psteps_one.
  eapply pstep_lift; [apply pi_nth_N | reflexivity |].
  eapply cstep_bne_direct_not_taken; [apply pi_nth_N | exact H].
Qed.

Lemma pi_assert_steps : forall R M σ' f,
  models (mkState R M) σ' -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  psteps T P (mkC pD 0 (mkState (rupd b f R) M), k)
             (mkC pE 0 (mkState (rupd b (Z.lxor f (truth (eval σ' e2))) R) M), k).
Proof.
  intros R M σ' f Hmod Hcl H0.
  assert (HT2 : run T2 (mkState (rupd b f R) M)
                = mkState (rupd b (Z.lxor f (truth (eval σ' e2))) R) M).
  { unfold T2. rewrite (flag_block_spec e2 b _ σ').
    - cbn [regs mem]. now rewrite rupd_same, rupd_shadow.
    - exact Hmod.
    - intros r Hr; cbn [regs]; rewrite rupd_other by lia; apply Hcl; lia.
    - cbn [regs]; rewrite rupd_other by lia; exact H0.
    - exact Hb. }
  rewrite <- HT2.
  replace pD with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))])) by len_solve.
  replace pE with (length (pre ++ ops_o lo T1
      ++ (Some ltest, CBeq b 0%nat lfalse) :: (None, COp (IXori b 1)) :: pa
      ++ [(None, COp (IXori b 1))]) + length T2)%nat by len_solve.
  eapply psteps_ops_l. unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_test_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  psteps T P (mkC n0 0 (mkState R M), k)
             (mkC pB 0 (mkState (rupd b (truth (eval σ e1)) R) M), k).
Proof.
  intros R M σ Hmod Hcl H0.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT1 : run T1 (mkState R M) = mkState (rupd b (truth (eval σ e1)) R) M).
  { unfold T1. rewrite (flag_block_spec e1 b _ σ).
    - cbn [regs mem]. rewrite Hb0. now rewrite Z.lxor_0_l.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  rewrite <- HT1.
  replace pB with (length pre + length T1)%nat by len_solve.
  eapply (psteps_ops_gen T T1 P pre (ops_o lo T1)); [| apply snd_ops_o].
  unfold P, if_code_o; norm_app; reflexivity.
Qed.

Lemma pi_then_path : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 ->
  psteps T P (mkC (length preA) 0 (mkState R M), k)
             (mkC (length preA + length pa) 0 (mkState R' M'), k) ->
  models (mkState R' M') σ' -> R' = R ->
  psteps T P (mkC (length pre) 0 (mkState R M), k)
             (mkC pN 0 (mkState (rupd b (Z.lxor 1 (truth (eval σ' e2))) R) M'), k).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 Hbody Hmod' HR; subst R'.
  rewrite p_len_preA in Hbody.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply psteps_trans; [apply (pi_test_steps R M σ Hmod Hcl H0) |].
  rewrite (truth_nz _ He1).
  eapply psteps_step.
  { eapply pstep_lift; [apply pi_nth_B | reflexivity |].
    eapply cstep_beq_direct_not_taken; [apply pi_nth_B |].
    cbn [regs]. rewrite rupd_same, rupd_other by auto. rewrite H0. discriminate. }
  eapply psteps_step; [eapply pstep_op; apply pi_nth_X1 |].
  cbn [step regs mem]. rewrite rupd_same, rupd_shadow.
  change (Z.lxor 1 1) with 0. rewrite rupd_zero by exact Hb0.
  eapply psteps_trans; [exact Hbody |].
  eapply psteps_step; [eapply pstep_op; apply pi_nth_X2 |].
  cbn [step regs mem]. rewrite Hb0. change (Z.lxor 0 1) with 1.
  eapply psteps_trans; [apply (pi_assert_steps R M' σ' 1 Hmod' Hcl H0) |].
  apply pi_join_steps.
Qed.

Lemma pi_else_path : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 ->
  psteps T P (mkC (length preB) 0 (mkState R M), k)
             (mkC (length preB + length pb) 0 (mkState R' M'), k) ->
  models (mkState R' M') σ' -> R' = R ->
  psteps T P (mkC (length pre) 0 (mkState R M), k)
             (mkC pN 0 (mkState (rupd b (truth (eval σ' e2)) R) M'), k).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 Hbody Hmod' HR; subst R'.
  rewrite p_len_preB in Hbody.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply psteps_trans; [apply (pi_test_steps R M σ Hmod Hcl H0) |].
  rewrite He1, truth_0, rupd_zero by exact Hb0.
  eapply psteps_step.
  { eapply pstep_lift; [apply pi_nth_B | reflexivity |].
    eapply cstep_beq_direct_taken; [apply pi_nth_B | apply pi_find_lfalse |].
    cbn [regs]. congruence. }
  eapply psteps_step.
  { eapply pstep_bra; [apply pi_nth_F | apply Hsafe; safe_lbl | apply Hsafe; safe_lbl |].
    eapply cstep_bra_paired_jump; [apply pi_nth_F | apply pi_find_ltest | apply pi_paired_F | |].
    - unfold pF, pE, pD; lia.
    - lia. }
  rewrite Z.add_0_l, Nat2Z.id.
  eapply psteps_step.
  { eapply pstep_lift; [apply pi_nth_B | reflexivity |].
    eapply cstep_beq_cancel; [apply pi_nth_B | apply pi_find_lfalse | | |].
    - cbn [regs]. congruence.
    - unfold pF, pE, pD; lia.
    - lia. }
  eapply psteps_trans; [exact Hbody |].
  eapply psteps_step.
  { eapply pstep_bra; [apply pi_nth_G | apply Hsafe; safe_lbl | reflexivity |].
    eapply cstep_bra_direct; [apply pi_nth_G | apply pi_find_lassert |].
    eapply paired_unlabeled; apply pi_nth_G. }
  assert (HA := pi_assert_steps R M' σ' 0 Hmod' Hcl H0).
  rewrite rupd_zero in HA by exact Hb0. rewrite Z.lxor_0_l in HA.
  eapply psteps_trans; [exact HA |].
  apply pi_join_steps.
Qed.

Lemma pi_if_true_run : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 -> eval σ' e2 <> 0 ->
  psteps T P (mkC (length preA) 0 (mkState R M), k)
             (mkC (length preA + length pa) 0 (mkState R' M'), k) ->
  models (mkState R' M') σ' -> R' = R ->
  psteps T P (mkC (length pre) 0 (mkState R M), k)
    (mkC (length pre + length (if_code_o lo fin b n e1 e2 pa pb)) 0 (mkState R M'), k).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 He2 Hbody Hmod' HR.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply psteps_trans; [eapply pi_then_path; eassumption |].
  rewrite (truth_nz _ He2). change (Z.lxor 1 1) with 0. rewrite rupd_zero by exact Hb0.
  rewrite p_end_pos. apply pi_check_pass. congruence.
Qed.

Lemma pi_if_false_run : forall R M R' M' σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 -> eval σ' e2 = 0 ->
  psteps T P (mkC (length preB) 0 (mkState R M), k)
             (mkC (length preB + length pb) 0 (mkState R' M'), k) ->
  models (mkState R' M') σ' -> R' = R ->
  psteps T P (mkC (length pre) 0 (mkState R M), k)
    (mkC (length pre + length (if_code_o lo fin b n e1 e2 pa pb)) 0 (mkState R M'), k).
Proof.
  intros R M R' M' σ σ' Hmod Hcl H0 He1 He2 Hbody Hmod' HR.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply psteps_trans; [eapply pi_else_path; eassumption |].
  rewrite He2, truth_0, rupd_zero by exact Hb0.
  rewrite p_end_pos. apply pi_check_pass. congruence.
Qed.

End IfLayoutP.

(** ** The loop layout on the procedure machine

    Section [LoopLayout] of CompileLoop.v redone for [psteps], with a
    possibly labeled first line.  S1 and S2 enter as runs in the same
    program: S1's run starts at the line labeled `from_do` — the main
    induction produces it for the code *as labeled*, so the relabeling
    argument of CompileLoop.v ([steps_relabel]) is not needed.  The back
    edge `BRA do` and `BRA exit` are unlabeled, hence unpaired
    ([paired_unlabeled]), whatever `from_do`'s line is (a data line, or a
    `BRA f` when S1 starts with a call). *)

Section LoopLayoutP.

Variable T : ptable.
Variable pre post : lprog.
Variable lo : option label.
Variable b : reg.
Variable n na nc : label.
Variable e1 e2 : expr.
Variable pa pb : lprog.
Variable fin : label.
Variable k : list (nat * Z).

Hypothesis Hb    : b <> 0%nat.
Hypothesis Hn    : (n + 4 <= na <= nc)%nat.
Hypothesis Hpre  : forall l, In l (labels pre)  -> ~ (n <= l < nc)%nat.
Hypothesis Hpost : forall l, In l (labels post) -> ~ (n <= l < nc)%nat.
Hypothesis Hpa   : forall l, In l (labels pa)   -> (n + 4 <= l < na)%nat.
Hypothesis Hpb   : forall l, In l (labels pb)   -> (na <= l < nc)%nat.
Hypothesis Hlo   : forall l, lo = Some l -> ~ (n <= l < nc)%nat.

Let ltest := n.
Let lloop := S n.
Let lexit := (n + 2)%nat.
Let ldo   := (n + 3)%nat.
Let T1 := flag_block e1 b.
Let T2 := flag_block e2 b.
Let LA := label_first ldo pa.

Let P := pre ++ loop_code_o lo fin b n e1 e2 pa pb ++ post.

Hypothesis Hsafe : forall l, (n <= l < n + 4)%nat ->
  is_proc T P l = false /\ bot_of T P (Some l) = None.

Let n0 := length pre.
Let t1 := length T1.
Let t2 := length T2.
Let la := length LA.
Let lb := length pb.
Let pK  := (n0 + t1)%nat.
Let pE1 := S pK.
Let pDo := S pE1.
Let pT  := (pDo + la)%nat.
Let pQ  := (pT + t2)%nat.
Let pC2 := S pQ.
Let pJ  := S pC2.
Let pL  := S pJ.
Let pS  := S pL.
Let pR  := (pS + lb)%nat.
Let pN3 := (pR + t1)%nat.
Let pBk := S pN3.
Let pX  := S pBk.

Let A0 := pre ++ ops_o lo T1 ++ [(None, COp (IXori b 1)); (None, CBne b 0%nat fin)].
Let A1 := A0 ++ LA ++ ops_l ltest T2.
Let A4 := A1 ++ [(None, CBeq b 0%nat lloop); (None, COp (IXori b 1)); (None, CBra lexit)].
Let A5 := A4 ++ [(Some lloop, COp nop)].
Let A6 := A5 ++ pb ++ ops T1.
Let A8 := A6 ++ [(None, CBne b 0%nat fin); (None, CBra ldo)].
Let Z9 := (Some lexit, COp nop) :: post.
Let postS := ops T1 ++ (None, CBne b 0%nat fin) :: (None, CBra ldo) :: Z9.
Let postD := ops_l ltest T2 ++ (None, CBeq b 0%nat lloop) :: (None, COp (IXori b 1))
             :: (None, CBra lexit) :: (Some lloop, COp nop) :: pb ++ postS.

Ltac len_solve :=
  unfold pX, pBk, pN3, pR, pS, pL, pJ, pC2, pQ, pT, pDo, pE1, pK, lb, la, t2, t1, n0,
         A8, A6, A5, A4, A1, A0, T1, T2 in *;
  repeat first [ rewrite length_app | rewrite length_ops | rewrite length_ops_l
               | rewrite length_ops_o | progress cbn [length] ];
  unfold lprog, line, label in *;
  lia.

Ltac lbl_solve :=
  unfold A8, A6, A5, A4, A1, A0, postD, postS, Z9, LA in *;
  let H := fresh in
  first [ intros ? H | intro H ]; in_labels_o H;
  repeat match goal with
  | H : In _ (labels (label_first _ _)) |- _ =>
      apply labels_label_first in H; destruct H as [H | H]
  | H : In _ (labels pre) |- _ => apply Hpre in H
  | H : In _ (labels post) |- _ => apply Hpost in H
  | H : In _ (labels pa) |- _ => apply Hpa in H
  | H : In _ (labels pb) |- _ => apply Hpb in H
  | H : lo = Some _ |- _ => apply Hlo in H
  end;
  unfold ltest, lloop, lexit, ldo in *; lia.

Ltac safe_lbl := unfold ltest, lloop, lexit, ldo; lia.

Lemma pl_P_D : P = A0 ++ LA ++ postD.
Proof. unfold P, A0, LA, postD, postS, Z9, loop_code_o; norm_app; reflexivity. Qed.

Lemma pl_P_S : P = A5 ++ pb ++ postS.
Proof. unfold P, A5, A4, A1, A0, LA, postS, Z9, loop_code_o; norm_app; reflexivity. Qed.

Lemma pl_len_loop_code : (length pre + length (loop_code_o lo fin b n e1 e2 pa pb))%nat = S pX.
Proof. unfold loop_code_o; fold T1 T2 ldo LA; len_solve. Qed.

Lemma pl_pDo_eq : (length pre + S (S (length (flag_block e1 b))))%nat = pDo.
Proof. len_solve. Qed.

Lemma pl_len_A0 : length A0 = pDo.
Proof. len_solve. Qed.

Lemma pl_pT_eq : (length A0 + length (label_first ldo pa))%nat = pT.
Proof. fold LA; len_solve. Qed.

Lemma pl_len_A5 : length A5 = pS.
Proof. len_solve. Qed.

Lemma pl_nth_K : nth_error P pK = Some (None, COp (IXori b 1)).
Proof.
  replace pK with (length (pre ++ ops_o lo T1)) by len_solve.
  eapply nth_error_at. unfold P, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_nth_E1 : nth_error P pE1 = Some (None, CBne b 0%nat fin).
Proof.
  replace pE1 with (length (pre ++ ops_o lo T1 ++ [(None, COp (IXori b 1))])) by len_solve.
  eapply nth_error_at. unfold P, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_nth_Q : nth_error P pQ = Some (None, CBeq b 0%nat lloop).
Proof.
  replace pQ with (length A1) by len_solve.
  eapply nth_error_at. unfold P, A1, A0, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_nth_C2 : nth_error P pC2 = Some (None, COp (IXori b 1)).
Proof.
  replace pC2 with (length (A1 ++ [(None, CBeq b 0%nat lloop)])) by len_solve.
  eapply nth_error_at. unfold P, A1, A0, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_nth_J : nth_error P pJ = Some (None, CBra lexit).
Proof.
  replace pJ with (length (A1 ++ [(None, CBeq b 0%nat lloop); (None, COp (IXori b 1))]))
    by len_solve.
  eapply nth_error_at. unfold P, A1, A0, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_nth_L : nth_error P pL = Some (Some lloop, COp nop).
Proof.
  replace pL with (length A4) by len_solve.
  eapply nth_error_at. unfold P, A4, A1, A0, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_nth_N3 : nth_error P pN3 = Some (None, CBne b 0%nat fin).
Proof.
  replace pN3 with (length A6) by len_solve.
  eapply nth_error_at. unfold P, A6, A5, A4, A1, A0, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_nth_Bk : nth_error P pBk = Some (None, CBra ldo).
Proof.
  replace pBk with (length (A6 ++ [(None, CBne b 0%nat fin)])) by len_solve.
  eapply nth_error_at. unfold P, A6, A5, A4, A1, A0, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_nth_X : nth_error P pX = Some (Some lexit, COp nop).
Proof.
  replace pX with (length A8) by len_solve.
  eapply nth_error_at. unfold P, A8, A6, A5, A4, A1, A0, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_find_ldo : find_label ldo P = Some pDo.
Proof.
  replace pDo with (length A0) by len_solve.
  rewrite pl_P_D. unfold LA.
  destruct pa as [| [lo' x] t] eqn:Epa; cbn [label_first].
  - eapply find_label_at; [reflexivity | lbl_solve].
  - eapply find_label_at; [reflexivity | lbl_solve].
Qed.

Lemma pl_find_lloop : find_label lloop P = Some pL.
Proof.
  replace pL with (length A4) by len_solve.
  eapply find_label_at;
    [unfold P, A4, A1, A0, loop_code_o; norm_app; reflexivity | lbl_solve].
Qed.

Lemma pl_find_lexit : find_label lexit P = Some pX.
Proof.
  replace pX with (length A8) by len_solve.
  eapply find_label_at;
    [unfold P, A8, A6, A5, A4, A1, A0, loop_code_o; norm_app; reflexivity | lbl_solve].
Qed.

(** *** Entry *)

Lemma pl_entry_check : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  psteps T P (mkC n0 0 (mkState R M), k)
             (mkC pE1 0 (mkState (rupd b (Z.lxor (truth (eval σ e1)) 1) R) M), k).
Proof.
  intros R M σ Hmod Hcl H0.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT1 : run T1 (mkState R M) = mkState (rupd b (truth (eval σ e1)) R) M).
  { unfold T1. rewrite (flag_block_spec e1 b _ σ).
    - cbn [regs mem]. now rewrite Hb0, Z.lxor_0_l.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  eapply psteps_trans.
  { eapply (psteps_ops_gen T T1 P pre (ops_o lo T1)); [| apply snd_ops_o].
    unfold P, loop_code_o; norm_app; reflexivity. }
  rewrite HT1.
  apply psteps_one.
  rewrite (pstep_op T P _ 0 _ k None (IXori b 1)) by (apply pl_nth_K).
  cbn [step regs mem]. now rewrite rupd_same, rupd_shadow.
Qed.

Lemma pl_entry_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 ->
  psteps T P (mkC n0 0 (mkState R M), k) (mkC pDo 0 (mkState R M), k).
Proof.
  intros R M σ Hmod Hcl H0 He1.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply psteps_trans; [apply (pl_entry_check R M σ Hmod Hcl H0) |].
  rewrite (truth_nz _ He1). change (Z.lxor 1 1) with 0.
  rewrite rupd_zero by exact Hb0.
  apply psteps_one.
  eapply pstep_lift; [apply pl_nth_E1 | reflexivity |].
  eapply cstep_bne_direct_not_taken; [apply pl_nth_E1 | cbn [regs]; congruence].
Qed.

(** *** The exit test holds *)

Lemma pl_exit_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e2 <> 0 ->
  psteps T P (mkC pT 0 (mkState R M), k) (mkC (S pX) 0 (mkState R M), k).
Proof.
  intros R M σ Hmod Hcl H0 He2.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT2 : run T2 (mkState R M) = mkState (rupd b 1 R) M).
  { unfold T2. rewrite (flag_block_spec e2 b _ σ).
    - cbn [regs mem]. now rewrite Hb0, Z.lxor_0_l, (truth_nz _ He2).
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  replace pT with (length (A0 ++ LA)) by len_solve.
  eapply psteps_trans.
  { eapply (psteps_ops_l T ltest T2 P (A0 ++ LA)).
    unfold P, A0, LA, loop_code_o; norm_app; reflexivity. }
  rewrite HT2.
  replace (length (A0 ++ LA) + length T2)%nat with pQ by len_solve.
  eapply psteps_step.
  { eapply pstep_lift; [apply pl_nth_Q | reflexivity |].
    eapply cstep_beq_direct_not_taken; [apply pl_nth_Q |].
    cbn [regs]. rewrite rupd_same, rupd_other by auto. rewrite H0. discriminate. }
  eapply psteps_step; [eapply pstep_op; apply pl_nth_C2 |].
  cbn [step regs mem]. rewrite rupd_same, rupd_shadow.
  change (Z.lxor 1 1) with 0. rewrite rupd_zero by exact Hb0.
  eapply psteps_step.
  { eapply pstep_bra; [apply pl_nth_J | apply Hsafe; safe_lbl | reflexivity |].
    eapply cstep_bra_direct; [apply pl_nth_J | apply pl_find_lexit |].
    eapply paired_unlabeled; apply pl_nth_J. }
  apply psteps_one.
  rewrite (pstep_op T P pX 0 _ k (Some lexit) nop) by apply pl_nth_X.
  now rewrite step_nop.
Qed.

(** *** The exit test fails: the flag stays 0, and S2 runs with it *)

Lemma pl_iter_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e2 = 0 ->
  psteps T P (mkC pT 0 (mkState R M), k) (mkC pS 0 (mkState R M), k).
Proof.
  intros R M σ Hmod Hcl H0 He2.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT2 : run T2 (mkState R M) = mkState R M).
  { unfold T2. rewrite (flag_block_spec e2 b _ σ).
    - cbn [regs mem]. rewrite Hb0, He2, truth_0. change (Z.lxor 0 0) with 0.
      now rewrite rupd_zero.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  replace pT with (length (A0 ++ LA)) by len_solve.
  eapply psteps_trans.
  { eapply (psteps_ops_l T ltest T2 P (A0 ++ LA)).
    unfold P, A0, LA, loop_code_o; norm_app; reflexivity. }
  rewrite HT2.
  replace (length (A0 ++ LA) + length T2)%nat with pQ by len_solve.
  eapply psteps_step.
  { eapply pstep_lift; [apply pl_nth_Q | reflexivity |].
    eapply cstep_beq_direct_taken; [apply pl_nth_Q | apply pl_find_lloop |].
    cbn [regs]. congruence. }
  apply psteps_one.
  rewrite (pstep_op T P pL 0 _ k (Some lloop) nop) by apply pl_nth_L.
  now rewrite step_nop.
Qed.

(** *** Re-entry *)

Lemma pl_reentry_check : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  psteps T P (mkC pR 0 (mkState R M), k)
    (mkC pN3 0 (mkState (rupd b (truth (eval σ e1)) R) M), k).
Proof.
  intros R M σ Hmod Hcl H0.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  assert (HT1 : run T1 (mkState R M) = mkState (rupd b (truth (eval σ e1)) R) M).
  { unfold T1. rewrite (flag_block_spec e1 b _ σ).
    - cbn [regs mem]. now rewrite Hb0, Z.lxor_0_l.
    - exact Hmod.
    - intros r Hr; apply Hcl; lia.
    - exact H0.
    - exact Hb. }
  replace pR with (length (A5 ++ pb)) by len_solve.
  replace pN3 with (length (A5 ++ pb) + length T1)%nat by len_solve.
  rewrite <- HT1.
  eapply (psteps_ops T T1 P (A5 ++ pb)).
  unfold P, A5, A4, A1, A0, LA, loop_code_o; norm_app; reflexivity.
Qed.

Lemma pl_back_steps : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 = 0 ->
  psteps T P (mkC pR 0 (mkState R M), k) (mkC pDo 0 (mkState R M), k).
Proof.
  intros R M σ Hmod Hcl H0 He1.
  assert (Hb0 : R b = 0) by (apply Hcl; lia).
  eapply psteps_trans; [apply (pl_reentry_check R M σ Hmod Hcl H0) |].
  rewrite He1, truth_0.
  rewrite rupd_zero by exact Hb0.
  eapply psteps_step.
  { eapply pstep_lift; [apply pl_nth_N3 | reflexivity |].
    eapply cstep_bne_direct_not_taken; [apply pl_nth_N3 | cbn [regs]; congruence]. }
  apply psteps_one.
  eapply pstep_bra; [apply pl_nth_Bk | apply Hsafe; safe_lbl | reflexivity |].
  eapply cstep_bra_direct; [apply pl_nth_Bk | apply pl_find_ldo |].
  eapply paired_unlabeled; apply pl_nth_Bk.
Qed.

(** *** The loop, in the three shapes of its derivation *)

Lemma pl_loop_enter : forall R M σ,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  eval σ e1 <> 0 ->
  psteps T P (mkC (length pre) 0 (mkState R M), k)
             (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M), k).
Proof. intros; rewrite pl_pDo_eq; eapply pl_entry_steps; eassumption. Qed.

(** A run of S1 as the main induction provides it: from `from_do`'s line,
    the code of S1 as labeled, in [P]. *)
Definition s1_run (ms : state) (σ' : store) : Prop :=
  exists ms', psteps T P (mkC (length A0) 0 ms, k)
                         (mkC (length A0 + length (label_first ldo pa)) 0 ms', k)
              /\ models ms' σ' /\ regs ms' = regs ms.

Definition s2_run (ms : state) (σ' : store) : Prop :=
  exists ms', psteps T P (mkC (length A5) 0 ms, k) (mkC (length A5 + length pb) 0 ms', k)
              /\ models ms' σ' /\ regs ms' = regs ms.

Lemma pl_loop_last : forall R M σ σ',
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  s1_run (mkState R M) σ' -> eval σ' e2 <> 0 ->
  exists ms',
    psteps T P (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M), k)
               (mkC (length pre + length (loop_code_o lo fin b n e1 e2 pa pb)) 0 ms', k)
    /\ models ms' σ' /\ regs ms' = R.
Proof.
  intros R M σ σ' Hmod Hcl H0 [[R1 M1] [Hst [Hm Hr]]] He2; cbn [regs] in Hr; subst R1.
  exists (mkState R M1). split; [| split; [exact Hm | reflexivity]].
  rewrite pl_pDo_eq, pl_len_loop_code.
  rewrite pl_pT_eq, pl_len_A0 in Hst.
  eapply psteps_trans; [exact Hst |].
  eapply pl_exit_steps; eassumption.
Qed.

Lemma pl_loop_round : forall R M σ σ1 σ2,
  models (mkState R M) σ -> clean_above b (mkState R M) -> R 0%nat = 0 ->
  s1_run (mkState R M) σ1 -> eval σ1 e2 = 0 ->
  (forall M1, models (mkState R M1) σ1 -> s2_run (mkState R M1) σ2) ->
  eval σ2 e1 = 0 ->
  exists M2,
    psteps T P (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M), k)
               (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M2), k)
    /\ models (mkState R M2) σ2.
Proof.
  intros R M σ σ1 σ2 Hmod Hcl H0 [[R1 M1] [Hst1 [Hm1 Hr1]]] He2 IHc He1;
    cbn [regs] in Hr1; subst R1.
  destruct (IHc M1 Hm1) as [[R2 M2] [Hst2 [Hm2 Hr2]]]; cbn [regs] in Hr2; subst R2.
  exists M2. split; [| exact Hm2].
  rewrite pl_pDo_eq.
  rewrite pl_pT_eq, pl_len_A0 in Hst1.
  rewrite pl_len_A5 in Hst2. replace (pS + length pb)%nat with pR in Hst2 by len_solve.
  eapply psteps_trans; [exact Hst1 |].
  eapply psteps_trans; [eapply pl_iter_steps; eassumption |].
  eapply psteps_trans; [exact Hst2 |].
  eapply pl_back_steps; [exact Hm2 | exact Hcl | exact H0 | exact He1].
Qed.

End LoopLayoutP.

(** ** The procedure prologue is the identity

    With [br = 0] and the stack cell [M[r1 - 1] = 0], the six instructions
    `SUBI r1 1; EXCH r2 r1; SWAPBR r2; NEG r2; EXCH r2 r1; ADDI r1 1` leave
    the registers, the memory and [br] exactly as they were: [r2] goes into
    the cell and comes back, [SWAPBR] exchanges [r2 = 0] (the cell's old
    content) with [br = 0].  The interpreter runs them twice per call —
    once as code on entry, once as data on return — and both are this
    computation. *)

Lemma prologue_id : forall R M, M (R 1%nat - 1) = 0 ->
  let s2 := step (IExch 2%nat 1%nat) (step (ISubi 1%nat 1) (mkState R M)) in
  regs s2 2%nat = 0 /\
  step (IAddi 1%nat 1) (step (IExch 2%nat 1%nat)
     (step (INeg 2%nat) (mkState (rupd 2%nat 0 (regs s2)) (mem s2)))) = mkState R M.
Proof.
  intros R M HM s2. subst s2. cbn [step regs mem].
  unfold rupd, mupd. cbn [Nat.eqb]. rewrite !Z.eqb_refl.
  replace (R 1%nat - 1 + 1) with (R 1%nat) by lia.
  rewrite HM. split; [reflexivity |].
  f_equal.
  - apply functional_extensionality; intro r.
    destruct (Nat.eqb_spec r 1); [subst; reflexivity |].
    destruct (Nat.eqb_spec r 2); [subst; reflexivity | reflexivity].
  - apply functional_extensionality; intro a.
    destruct (Z.eqb_spec a (R 1%nat - 1)); [subst; now rewrite HM | reflexivity].
Qed.

Definition prologue_lines (e : label) : lprog :=
  [ (Some e, COp (ISubi 1%nat 1)); (None, COp (IExch 2%nat 1%nat)); (None, CSwapbr 2%nat)
  ; (None, COp (INeg 2%nat)); (None, COp (IExch 2%nat 1%nat)); (None, COp (IAddi 1%nat 1)) ].

Lemma proc_code_eq : forall e t bt B,
  proc_code e t bt B = (Some t, CBra bt) :: prologue_lines e ++ B ++ [(Some bt, CBra t)].
Proof. reflexivity. Qed.

Fixpoint data_list (L : lprog) (br : Z) (s : state) : option (Z * state) :=
  match L with
  | [] => Some (br, s)
  | (_, x) :: L' =>
      match data_step x br s with
      | Some (br', s') => data_list L' br' s'
      | None => None
      end
  end.

Lemma data_run_app : forall L pre rest br s,
  data_run (pre ++ L ++ rest) (length pre) (length L) br s = data_list L br s.
Proof.
  induction L as [| [lo x] L IH]; intros pre rest br s; [reflexivity |].
  cbn [length data_run data_list].
  rewrite (nth_error_at pre (lo, x) (L ++ rest)) by reflexivity.
  destruct (data_step x br s) as [[br' s'] |]; [| reflexivity].
  pose proof (IH (pre ++ [(lo, x)]) rest br' s') as H.
  rewrite <- app_assoc, length_app, Nat.add_comm in H. exact H.
Qed.

Lemma prologue_list : forall e R M, M (R 1%nat - 1) = 0 ->
  data_list (prologue_lines e) 0 (mkState R M) = Some (0, mkState R M).
Proof.
  intros e R M HM. pose proof (prologue_id R M HM) as [H2 Hid]. cbv zeta in H2, Hid.
  cbn [data_list data_step prologue_lines]. now rewrite Hid, H2.
Qed.

(** Lines that are data instructions or [SWAPBR] run as [data_run] says. *)
Lemma data_run_psteps : forall T P cnt pc br s br' s' k,
  data_run P pc cnt br s = Some (br', s') ->
  psteps T P (mkC pc br s, k) (mkC (pc + cnt) br' s', k).
Proof.
  intros T P; induction cnt as [| cnt IH]; intros pc br s br' s' k H; cbn [data_run] in H.
  - injection H as <- <-. rewrite Nat.add_0_r. constructor.
  - destruct (nth_error P pc) as [[lo x] |] eqn:Hn; [| discriminate].
    destruct x; cbn [data_step] in H; try discriminate.
    + eapply psteps_step; [eapply pstep_op; exact Hn |].
      replace (pc + S cnt)%nat with (S pc + cnt)%nat by lia. now apply IH.
    + eapply psteps_step; [eapply pstep_swapbr; exact Hn |].
      replace (pc + S cnt)%nat with (S pc + cnt)%nat by lia. now apply IH.
Qed.

(** ** Procedures laid out in the program *)

(** Procedure [f] in direction [d] (forwards, or the inverted companion
    `f_inv`) sits in [P] as `gen_proc` emits it, its body compiled at base
    r3 with labels [[nb, nb')] that nothing around it uses. *)
Definition proc_in (Γ : penv) (P : lprog) (f : pname) (d : bool) (body : pstmt) : Prop :=
  exists pre post B nb nb',
    P = pre ++ proc_code (pe f d) (pt f d) (pbt f d) B ++ post /\
    compile_p (dirb d body) scratch nb = (B, nb') /\ (L0 Γ <= nb)%nat /\
    (forall l, In l (labels pre) -> ~ (nb <= l < nb')%nat) /\
    (forall l, In l (labels post) -> ~ (nb <= l < nb')%nat) /\
    ~ In (pe f d) (labels pre) /\ ~ In (pt f d) (labels pre) /\ ~ In (pbt f d) (labels pre).

Definition procs_in (Γ : penv) (P : lprog) : Prop :=
  forall f d body, nth_error Γ f = Some body -> proc_in Γ P f d body.

(** ** Main theorem *)

Section MainSpec.

Variable Γ : penv.
Variable P : lprog.
Variable slot : var.                  (** the stack cell is [M[r1 - 1]], variable [slot] *)

Hypothesis HP     : procs_in Γ P.
Hypothesis Hwfenv : env_wf Γ.
Hypothesis Hnomod : env_nomod Γ slot.

Let T := ptab (length Γ).
Let sp := (Z.of_nat slot + 1)%Z.

Lemma safe_high : forall l, (L0 Γ <= l)%nat ->
  is_proc T P l = false /\ bot_of T P (Some l) = None.
Proof.
  intros l Hl; split.
  - apply is_proc_false. intros e t b Hin E.
    apply in_ptab in Hin as [f [d [Hf [He _]]]]. subst.
    pose proof (pl_lt_L0 Γ f d Hf). lia.
  - cbn [bot_of]. apply bot_lookup_none. intros e t b Hin E.
    apply in_ptab in Hin as [f [d [Hf [_ [_ Hb]]]]]. subst.
    pose proof (pl_lt_L0 Γ f d Hf). lia.
Qed.

Lemma proc_facts : forall f d body, nth_error Γ f = Some body ->
  exists pre post B nb nb',
    P = pre ++ proc_code (pe f d) (pt f d) (pbt f d) B ++ post /\
    compile_p (dirb d body) scratch nb = (B, nb') /\ (L0 Γ <= nb)%nat /\
    (forall l, In l (labels pre) -> ~ (nb <= l < nb')%nat) /\
    (forall l, In l (labels post) -> ~ (nb <= l < nb')%nat) /\
    find_label (pe f d) P = Some (S (length pre)) /\
    find_label (pt f d) P = Some (length pre) /\
    find_label (pbt f d) P = Some (length pre + 7 + length B)%nat /\
    is_proc T P (pe f d) = true /\ bot_of T P (Some (pbt f d)) = Some (pe f d).
Proof.
  intros f d body Hf.
  assert (Hlt : (f < length Γ)%nat) by (apply nth_error_Some; congruence).
  destruct (pl_lt_L0 Γ f d Hlt) as [Lpe [Lpt Lpbt]].
  destruct (HP f d body Hf)
    as [pre [post [B [nb [nb' [HPq [HB [Hnb [Hpre [Hpost [Npe [Npt Npb]]]]]]]]]]]].
  destruct (compile_p_labels _ _ _ _ _ HB) as [Hle HinB].
  assert (Ft : find_label (pt f d) P = Some (length pre)).
  { eapply find_label_at; [exact HPq | exact Npt]. }
  assert (Fe : find_label (pe f d) P = Some (S (length pre))).
  { replace (S (length pre)) with (length (pre ++ [(Some (pt f d), CBra (pbt f d))]))
      by (rewrite length_app; simpl; lia).
    eapply find_label_at; [rewrite HPq; unfold proc_code; norm_app; reflexivity |].
    rewrite labels_app; cbn [labels]. rewrite in_app_iff. intros [H | [H | []]];
      [contradiction | unfold pt in H; lia]. }
  assert (Fb : find_label (pbt f d) P = Some (length pre + 7 + length B)%nat).
  { replace (length pre + 7 + length B)%nat
      with (length (pre ++ (Some (pt f d), CBra (pbt f d)) :: prologue_lines (pe f d) ++ B))
      by (rewrite !length_app; simpl; lia).
    eapply find_label_at; [rewrite HPq, proc_code_eq; norm_app; reflexivity |].
    rewrite labels_app; cbn [labels prologue_lines]. rewrite labels_app, !in_app_iff.
    intros [H | [H | [H | H]]];
      [contradiction | unfold pt, pbt in H; lia | unfold pbt in H; lia |].
    apply HinB in H. lia. }
  assert (Hok : entry_ok P (pe f d, pt f d, pbt f d) = true)
    by (unfold entry_ok, defined; now rewrite Fe, Ft, Fb).
  exists pre, post, B, nb, nb'. repeat split; try assumption.
  - eapply is_proc_true; [apply ptab_in; exact Hlt | exact Hok].
  - cbn [bot_of]. eapply bot_lookup_some; [apply ptab_in; exact Hlt |].
    intros e t Hin. apply in_ptab in Hin as [f' [d' [_ [-> [-> Hb]]]]].
    assert (Hpl : is_pl f' d' (pbt f d)) by (unfold is_pl; auto).
    apply pbt_inj in Hpl as [-> ->]. split; [reflexivity | exact Hok].
Qed.

(** A call: CALL, the prologue (identity), the body's run, RETURN. *)
Lemma call_steps : forall f d pc lo ms ms' k pre post B,
  P = pre ++ proc_code (pe f d) (pt f d) (pbt f d) B ++ post ->
  find_label (pe f d) P = Some (S (length pre)) ->
  find_label (pt f d) P = Some (length pre) ->
  is_proc T P (pe f d) = true -> bot_of T P (Some (pbt f d)) = Some (pe f d) ->
  nth_error P pc = Some (lo, CBra (pe f d)) -> bot_of T P lo = None ->
  mem ms (regs ms 1%nat - 1) = 0 -> mem ms' (regs ms' 1%nat - 1) = 0 ->
  psteps T P (mkC (length pre + 7) 0 ms, (S pc, 0) :: k)
             (mkC (length pre + 7 + length B) 0 ms', (S pc, 0) :: k) ->
  psteps T P (mkC pc 0 ms, k) (mkC (S pc) 0 ms', k).
Proof.
  intros f d pc lo [R M] [R' M'] k pre post B HPq Fe Ft Hip Hbot Hn Hlo HM HM' Hbody.
  cbn [regs mem] in HM, HM'.
  assert (Hsplit : P = (pre ++ [(Some (pt f d), CBra (pbt f d))]) ++ prologue_lines (pe f d)
                       ++ (B ++ [(Some (pbt f d), CBra (pt f d))] ++ post))
    by (rewrite HPq, proc_code_eq; norm_app; reflexivity).
  assert (Hlen : length (pre ++ [(Some (pt f d), CBra (pbt f d))]) = S (length pre))
    by (rewrite length_app; simpl; lia).
  assert (Hdr : forall R0 M0, M0 (R0 1%nat - 1) = 0 ->
            data_run P (S (length pre)) 6 0 (mkState R0 M0) = Some (0, mkState R0 M0)).
  { intros R0 M0 H. rewrite Hsplit, <- Hlen.
    change 6%nat with (length (prologue_lines (pe f d))).
    rewrite data_run_app. now apply prologue_list. }
  (* CALL *)
  eapply psteps_step; [eapply pstep_call; [exact Hn | exact Fe | exact Hlo | exact Hip] |].
  (* the prologue *)
  eapply psteps_trans; [eapply data_run_psteps; apply Hdr; exact HM |].
  replace (S (length pre) + 6)%nat with (length pre + 7)%nat by lia.
  (* the body *)
  eapply psteps_trans; [exact Hbody |].
  (* RETURN at `f_bot: BRA f_top` *)
  apply psteps_one.
  eapply pstep_return; [| exact Ft | exact Hbot | exact Fe | apply Hdr; exact HM'].
  replace (length pre + 7 + length B)%nat
    with (length ((pre ++ (Some (pt f d), CBra (pbt f d)) :: prologue_lines (pe f d)) ++ B))
    by (rewrite !length_app; simpl; lia).
  eapply nth_error_at. rewrite HPq, proc_code_eq; norm_app; reflexivity.
Qed.

Ltac lbl_main :=
  let H := fresh in
  intros ? H; in_labels_o H;
  repeat match goal with
  | H : In ?l (labels ?q), IH : forall l, In l (labels ?q) -> _ |- _ => apply IH in H
  | H : ?lo = Some _, IH : forall l, ?lo = Some l -> _ |- _ => apply IH in H; destruct H
  end; lia.

Ltac side_p HPe :=
  first
    [ assumption
    | (unfold scratch in *; lia)
    | (let l := fresh "l" in let Hl := fresh "Hl" in intros l Hl;
       first
         [ (rewrite <- HPe; apply safe_high; unfold scratch in *; lia)
         | (match goal with
            | IH : forall l, In l (labels ?q) -> _, H : In l (labels ?q) |- _ => apply IH in H
            end; unfold scratch in *; lia)
         | (match goal with
            | IH : forall l, ?lo = Some l -> _, H : ?lo = Some l |- _ => destruct (IH l H)
            end; unfold scratch in *; lia) ]) ].

Theorem compile_p_spec : forall st σ σ', exec_p Γ st σ σ' ->
  forall b n p n' ms pre post lo k,
  wf_p st -> pmods st slot = false ->
  compile_p st b n = (p, n') -> (scratch <= b)%nat -> (L0 Γ <= n)%nat ->
  P = pre ++ relab lo p ++ post ->
  (forall l, In l (labels pre)  -> ~ (n <= l < n')%nat) ->
  (forall l, In l (labels post) -> ~ (n <= l < n')%nat) ->
  (forall l, lo = Some l -> ~ (n <= l < n')%nat /\ (L0 Γ <= l)%nat) ->
  models ms σ -> σ slot = 0 -> clean_above b ms -> regs ms 0%nat = 0 -> regs ms 1%nat = sp ->
  (has_call st = true -> clean_above scratch ms) ->
  exists ms',
    psteps T P (mkC (length pre) 0 ms, k) (mkC (length pre + length (relab lo p)) 0 ms', k)
    /\ models ms' σ' /\ regs ms' = regs ms.
Proof.
  intros st σ σ' Hex.
  induction Hex as [s σ σ' Hs | a c σ m σ' Ha IHa Hc IHc
                   | e1 a c e2 σ σ' He1 Ha IHa He2 | e1 a c e2 σ σ' He1 Hc IHc He2
                   | e1 a c e2 σ σ' He1 Hlp IHlp
                   | f body σ σ' Hf Hbd IHbd | f body σ σ' Hf Hbd IHbd
                   | e1 a c e2 σ σ' Ha IHa He2
                   | e1 a c e2 σ σ1 σ2 σ' Ha IHa He2 Hc IHc He1 Hlp IHlp]
    using exec_p_mut with
    (P0 := fun e1 a c e2 σ σ' (_ : lp_p Γ e1 a c e2 σ σ') =>
      forall b n pa na pc nc R M pre post lo k,
      wf_p a -> wf_p c -> pmods a slot = false -> pmods c slot = false ->
      compile_p a (S b) (n + 4)%nat = (pa, na) -> compile_p c (S b) na = (pc, nc) ->
      (scratch <= b)%nat -> (L0 Γ <= n)%nat ->
      P = pre ++ loop_code_o lo fin_label b n e1 e2 pa pc ++ post ->
      (forall l, In l (labels pre)  -> ~ (n <= l < nc)%nat) ->
      (forall l, In l (labels post) -> ~ (n <= l < nc)%nat) ->
      (forall l, lo = Some l -> ~ (n <= l < nc)%nat /\ (L0 Γ <= l)%nat) ->
      models (mkState R M) σ -> σ slot = 0 -> clean_above b (mkState R M) -> R 0%nat = 0 ->
      R 1%nat = sp ->
      (has_call a || has_call c = true -> clean_above scratch (mkState R M)) ->
      exists ms',
        psteps T P (mkC (length pre + S (S (length (flag_block e1 b)))) 0 (mkState R M), k)
                   (mkC (length pre + length (loop_code_o lo fin_label b n e1 e2 pa pc)) 0 ms', k)
        /\ models ms' σ' /\ regs ms' = R).
  - (* PBase *)
    intros b n p n' ms pre post lo k Hwf Hmd Hcomp Hb Hn HPe Hpre Hpost Hlo Hmodel Hslot Hcl
           H0 H1 Hcall; simpl in Hcomp.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    injection Hcomp as <- <-.
    destruct (compile_at_spec b s σ σ' ms Hs Hwf Hmodel Hcl H0 ltac:(lia)) as [Hm Hr].
    destruct (compile_at b s) as [| i c] eqn:Ec.
    + assert (σ' = σ) by (eapply compile_at_nil; eassumption). subst σ'.
      destruct lo as [l |].
      * exists ms. split; [| split; [exact Hmodel | reflexivity]].
        apply psteps_one. cbn [relab label_first ops map length] in HPe |- *.
        rewrite (pstep_op T P (length pre) 0 ms k (Some l) nop), step_nop
          by (rewrite HPe; eapply nth_error_at; reflexivity).
        do 3 f_equal. lia.
      * exists ms. split; [| split; [exact Hmodel | reflexivity]].
        cbn [relab ops map length]. rewrite Nat.add_0_r. constructor.
    + exists (run (i :: c) ms). split; [| split; assumption].
      rewrite relab_ops in HPe |- * by discriminate.
      rewrite length_ops_o.
      eapply psteps_ops_gen; [exact HPe | apply snd_ops_o].
  - (* PSeq *)
    intros b n p n' ms pre post lo k Hwf Hmd Hcomp Hb Hn HPe Hpre Hpost Hlo Hmodel Hslot Hcl
           H0 H1 Hcall; simpl in Hcomp.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    destruct Hwf as [Hwfa Hwfc]. cbn [pmods] in Hmd. apply orb_false_elim in Hmd as [Hma Hmc].
    cbn [has_call] in Hcall.
    destruct (compile_p a b n) as [p1 n1] eqn:E1.
    destruct (compile_p c b n1) as [p2 n2] eqn:E2.
    injection Hcomp as <- <-.
    destruct (compile_p_labels _ _ _ _ _ E1) as [Hle1 Hin1].
    destruct (compile_p_labels _ _ _ _ _ E2) as [Hle2 Hin2].
    assert (Hsm : m slot = 0)
      by (rewrite (exec_p_frame Γ a σ m slot Ha Hnomod Hma); exact Hslot).
    destruct p1 as [| x1 p1'].
    + (* S1 emits no code: it is a no-op, and the label (if any) goes to S2 *)
      assert (m = σ) by (eapply compile_p_nil; eassumption). subst m.
      cbn [app] in HPe |- *.
      apply (IHc b n1 p2 n2 ms pre post lo k Hwfc Hmc E2 Hb ltac:(unfold scratch in *; lia) HPe); try assumption.
      * intros l Hl; apply Hpre in Hl; lia.
      * intros l Hl; apply Hpost in Hl; lia.
      * intros l E; destruct (Hlo l E); split; [lia | assumption].
      * intro H; apply Hcall; now rewrite H, orb_true_r.
    + rewrite relab_app in HPe |- * by discriminate.
      set (p1 := x1 :: p1') in *.
      destruct (IHa b n p1 n1 ms pre (p2 ++ post) lo k Hwfa Hma E1 Hb Hn)
        as [ms1 [Hst1 [Hm1 Hr1]]].
      { rewrite HPe. now rewrite <- app_assoc. }
      { intros l Hl; apply Hpre in Hl; lia. }
      { intros l Hl; rewrite labels_app, in_app_iff in Hl.
        destruct Hl as [Hl | Hl]; [apply Hin2 in Hl | apply Hpost in Hl]; lia. }
      { intros l E; destruct (Hlo l E); split; [lia | assumption]. }
      all: try assumption.
      { intro H; apply Hcall; now rewrite H. }
      assert (Hcl1 : clean_above b ms1) by (intros r Hr; rewrite Hr1; now apply Hcl).
      assert (H01 : regs ms1 0%nat = 0) by (rewrite Hr1; exact H0).
      assert (H11 : regs ms1 1%nat = sp) by (rewrite Hr1; exact H1).
      destruct (IHc b n1 p2 n2 ms1 (pre ++ relab lo p1) post None k Hwfc Hmc E2 Hb
                    ltac:(unfold scratch in *; lia))
        as [ms2 [Hst2 [Hm2 Hr2]]].
      { rewrite HPe; cbn [relab]; now rewrite <- !app_assoc. }
      { intros l Hl; rewrite labels_app, in_app_iff in Hl.
        destruct Hl as [Hl | Hl]; [apply Hpre in Hl; lia |].
        destruct lo as [l0 |]; cbn [relab] in Hl.
        - apply labels_label_first in Hl as [-> | Hl].
          + destruct (Hlo l0 eq_refl); lia.
          + apply Hin1 in Hl; lia.
        - apply Hin1 in Hl; lia. }
      { intros l Hl; apply Hpost in Hl; lia. }
      { intros l E; discriminate. }
      all: try assumption.
      { intro H; unfold clean_above; rewrite Hr1; apply Hcall; now rewrite H, orb_true_r. }
      exists ms2. split; [| split; [exact Hm2 | now rewrite Hr2, Hr1]].
      rewrite length_app in Hst2. cbn [relab] in Hst2.
      rewrite length_app.
      eapply psteps_trans; [exact Hst1 |].
      replace (length pre + (length (relab lo p1) + length p2))%nat
        with (length pre + length (relab lo p1) + length p2)%nat by lia.
      exact Hst2.
  - (* PIf, then path *)
    intros b n p n' ms pre post lo k Hwf Hmd Hcomp Hb Hn HPe Hpre Hpost Hlo Hmodel Hslot Hcl
           H0 H1 Hcall; simpl in Hcomp.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    destruct Hwf as [Hwfa Hwfc]. cbn [pmods] in Hmd. apply orb_false_elim in Hmd as [Hma Hmc].
    cbn [has_call] in Hcall.
    destruct (compile_p a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_p c (S b) na) as [pb nb] eqn:Eb.
    injection Hcomp as <- <-.
    destruct (compile_p_labels _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_p_labels _ _ _ _ _ Eb) as [Hleb Hinb].
    rewrite relab_if_code in HPe |- *.
    destruct ms as [R M].
    assert (HclS : clean_above (S b) (mkState R M)) by (intros r Hr; apply Hcl; lia).
    destruct (IHa (S b) (n + 5)%nat pa na (mkState R M)
                  (pre ++ ops_o lo (flag_block e1 b)
                       ++ [(Some (S n), CBeq b 0%nat n); (None, COp (IXori b 1))])
                  ((None, COp (IXori b 1))
                   :: ops_l (n + 2)%nat (flag_block e2 b)
                   ++ (Some (n + 3)%nat, CBra (n + 4)%nat)
                   :: (Some n, CBra (S n))
                   :: pb
                   ++ (None, CBra (n + 2)%nat)
                   :: (Some (n + 4)%nat, CBra (n + 3)%nat)
                   :: (None, CBne b 0%nat fin_label)
                   :: post) None k
                  Hwfa Hma Ea ltac:(unfold scratch in *; lia) ltac:(unfold scratch in *; lia))
      as [[R' M'] [Hst [Hm' Hr']]].
    { rewrite HPe; unfold if_code_o; cbn [relab]; norm_app; reflexivity. }
    { lbl_main. }
    { lbl_main. }
    { intros l E; discriminate. }
    all: try assumption.
    { intro H; apply Hcall; now rewrite H. }
    cbn [regs] in Hr'. cbn [relab] in Hst.
    exists (mkState R M'). split; [| split; [exact Hm' | reflexivity]].
    rewrite HPe. rewrite HPe in Hst.
    apply (pi_if_true_run T pre post lo b n e1 e2 pa pb fin_label k)
      with (σ := σ) (σ' := σ') (R' := R').
    all: side_p HPe.
  - (* PIf, else path *)
    intros b n p n' ms pre post lo k Hwf Hmd Hcomp Hb Hn HPe Hpre Hpost Hlo Hmodel Hslot Hcl
           H0 H1 Hcall; simpl in Hcomp.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    destruct Hwf as [Hwfa Hwfc]. cbn [pmods] in Hmd. apply orb_false_elim in Hmd as [Hma Hmc].
    cbn [has_call] in Hcall.
    destruct (compile_p a (S b) (n + 5)%nat) as [pa na] eqn:Ea.
    destruct (compile_p c (S b) na) as [pb nb] eqn:Eb.
    injection Hcomp as <- <-.
    destruct (compile_p_labels _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_p_labels _ _ _ _ _ Eb) as [Hleb Hinb].
    rewrite relab_if_code in HPe |- *.
    destruct ms as [R M].
    assert (HclS : clean_above (S b) (mkState R M)) by (intros r Hr; apply Hcl; lia).
    destruct (IHc (S b) na pb nb (mkState R M)
                  (pre ++ ops_o lo (flag_block e1 b)
                       ++ (Some (S n), CBeq b 0%nat n)
                       :: (None, COp (IXori b 1))
                       :: pa
                       ++ (None, COp (IXori b 1))
                       :: ops_l (n + 2)%nat (flag_block e2 b)
                       ++ [(Some (n + 3)%nat, CBra (n + 4)%nat); (Some n, CBra (S n))])
                  ((None, CBra (n + 2)%nat) :: (Some (n + 4)%nat, CBra (n + 3)%nat)
                   :: (None, CBne b 0%nat fin_label) :: post) None k
                  Hwfc Hmc Eb ltac:(unfold scratch in *; lia) ltac:(unfold scratch in *; lia))
      as [[R' M'] [Hst [Hm' Hr']]].
    { rewrite HPe; unfold if_code_o; cbn [relab]; norm_app; reflexivity. }
    { lbl_main. }
    { lbl_main. }
    { intros l E; discriminate. }
    all: try assumption.
    { intro H; apply Hcall; now rewrite H, orb_true_r. }
    cbn [regs] in Hr'. cbn [relab] in Hst.
    exists (mkState R M'). split; [| split; [exact Hm' | reflexivity]].
    rewrite HPe. rewrite HPe in Hst.
    apply (pi_if_false_run T pre post lo b n e1 e2 pa pb fin_label k)
      with (σ := σ) (σ' := σ') (R' := R').
    all: side_p HPe.
  - (* PLoop: the entry, then the rounds *)
    intros b n p n' ms pre post lo k Hwf Hmd Hcomp Hb Hn HPe Hpre Hpost Hlo Hmodel Hslot Hcl
           H0 H1 Hcall; simpl in Hcomp.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    destruct Hwf as [Hwfa Hwfc]. cbn [pmods] in Hmd.
    apply orb_false_elim in Hmd as [Hma Hmc]. cbn [has_call] in Hcall.
    destruct (compile_p a (S b) (n + 4)%nat) as [pa na] eqn:Ea.
    destruct (compile_p c (S b) na) as [pc nc] eqn:Ec.
    injection Hcomp as <- <-.
    destruct (compile_p_labels _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_p_labels _ _ _ _ _ Ec) as [Hlec Hinc].
    rewrite relab_loop_code in HPe |- *.
    destruct ms as [R M].
    destruct (IHlp b n pa na pc nc R M pre post lo k Hwfa Hwfc Hma Hmc Ea Ec Hb Hn HPe
                   Hpre Hpost Hlo Hmodel Hslot Hcl H0 H1 Hcall)
      as [ms' [Hst [Hm Hr]]].
    exists ms'. split; [| split; [exact Hm | exact Hr]].
    eapply psteps_trans; [| exact Hst].
    rewrite HPe.
    eapply (pl_loop_enter T pre post lo b n na nc e1 e2 pa pc fin_label k); try eassumption.
    all: side_p HPe.
  - (* PCall *)
    intros b n p n' ms pre post lo k Hwf Hmd Hcomp Hb Hn HPe Hpre Hpost Hlo Hmodel Hslot Hcl
           H0 H1 Hcall; simpl in Hcomp.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    injection Hcomp as <- <-.
    specialize (Hcall eq_refl).
    destruct (proc_facts f false body Hf)
      as [pre' [post' [B [nb [nb' [HPq [HB [Hnb [Hpre' [Hpost' [Fe [Ft [Fb [Hip Hbot]]]]]]]]]]]]]].
    cbn [dirb] in HB.
    destruct (compile_p_labels _ _ _ _ _ HB) as [HleB HinB].
    assert (Hlt : (f < length Γ)%nat) by (apply nth_error_Some; congruence).
    destruct (pl_lt_L0 Γ f false Hlt) as [Lpe [Lpt Lpbt]].
    destruct (IHbd scratch nb B nb' ms
                   (pre' ++ (Some (pt f false), CBra (pbt f false)) :: prologue_lines (pe f false))
                   ((Some (pbt f false), CBra (pt f false)) :: post') None ((S (length pre), 0) :: k)
                   (Hwfenv f body Hf) (Hnomod f body Hf) HB ltac:(unfold scratch in *; lia) Hnb)
      as [ms' [Hst [Hm Hr]]].
    { rewrite HPq, proc_code_eq; cbn [relab]; norm_app; reflexivity. }
    { intros l Hl; rewrite labels_app, in_app_iff in Hl.
      destruct Hl as [Hl | Hl]; [now apply Hpre' |].
      cbn [labels prologue_lines] in Hl. unfold pt in Hl, Lpt. destruct Hl as [<- | [<- | []]]; lia. }
    { intros l Hl; cbn [labels] in Hl. destruct Hl as [<- | Hl]; [lia | now apply Hpost']. }
    { intros l E; discriminate. }
    all: try assumption.
    { intros _; exact Hcall. }
    exists ms'. split; [| split; assumption].
    assert (Hnt : nth_error P (length pre) = Some (lo, CBra (pe f false))).
    { rewrite HPe. eapply nth_error_at. destruct lo; reflexivity. }
    replace (length pre + length (relab lo [(None, CBra (pe f false))]))%nat
      with (S (length pre)) by (destruct lo; cbn; lia).
    eapply (call_steps f false); try eassumption.
    + destruct lo as [l |]; [| reflexivity]. apply safe_high. now destruct (Hlo l eq_refl).
    + rewrite H1. unfold sp. replace (Z.of_nat slot + 1 - 1) with (Z.of_nat slot) by lia.
      rewrite Hmodel. exact Hslot.
    + rewrite Hr, H1. unfold sp. replace (Z.of_nat slot + 1 - 1) with (Z.of_nat slot) by lia.
      rewrite Hm. rewrite (exec_p_frame Γ body σ σ' slot Hbd Hnomod (Hnomod f body Hf)).
      exact Hslot.
    + cbn [relab length] in Hst.
      replace (length (pre' ++ (Some (pt f false), CBra (pbt f false))
                         :: prologue_lines (pe f false)))
        with (length pre' + 7)%nat in Hst by (rewrite length_app; simpl; lia).
      exact Hst.
  - (* PUncall: a call of the inverted companion *)
    intros b n p n' ms pre post lo k Hwf Hmd Hcomp Hb Hn HPe Hpre Hpost Hlo Hmodel Hslot Hcl
           H0 H1 Hcall; simpl in Hcomp.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    injection Hcomp as <- <-.
    specialize (Hcall eq_refl).
    destruct (proc_facts f true body Hf)
      as [pre' [post' [B [nb [nb' [HPq [HB [Hnb [Hpre' [Hpost' [Fe [Ft [Fb [Hip Hbot]]]]]]]]]]]]]].
    cbn [dirb] in HB.
    destruct (compile_p_labels _ _ _ _ _ HB) as [HleB HinB].
    assert (Hlt : (f < length Γ)%nat) by (apply nth_error_Some; congruence).
    destruct (pl_lt_L0 Γ f true Hlt) as [Lpe [Lpt Lpbt]].
    destruct (IHbd scratch nb B nb' ms
                   (pre' ++ (Some (pt f true), CBra (pbt f true)) :: prologue_lines (pe f true))
                   ((Some (pbt f true), CBra (pt f true)) :: post') None ((S (length pre), 0) :: k)
                   (wf_p_invert _ (Hwfenv f body Hf))
                   ltac:(rewrite pmods_invert; exact (Hnomod f body Hf)) HB ltac:(unfold scratch in *; lia) Hnb)
      as [ms' [Hst [Hm Hr]]].
    { rewrite HPq, proc_code_eq; cbn [relab]; norm_app; reflexivity. }
    { intros l Hl; rewrite labels_app, in_app_iff in Hl.
      destruct Hl as [Hl | Hl]; [now apply Hpre' |].
      cbn [labels prologue_lines] in Hl. unfold pt in Hl, Lpt. destruct Hl as [<- | [<- | []]]; lia. }
    { intros l Hl; cbn [labels] in Hl. destruct Hl as [<- | Hl]; [lia | now apply Hpost']. }
    { intros l E; discriminate. }
    all: try assumption.
    { intros _; exact Hcall. }
    exists ms'. split; [| split; assumption].
    assert (Hnt : nth_error P (length pre) = Some (lo, CBra (pe f true))).
    { rewrite HPe. eapply nth_error_at. destruct lo; reflexivity. }
    replace (length pre + length (relab lo [(None, CBra (pe f true))]))%nat
      with (S (length pre)) by (destruct lo; cbn; lia).
    eapply (call_steps f true); try eassumption.
    + destruct lo as [l |]; [| reflexivity]. apply safe_high. now destruct (Hlo l eq_refl).
    + rewrite H1. unfold sp. replace (Z.of_nat slot + 1 - 1) with (Z.of_nat slot) by lia.
      rewrite Hmodel. exact Hslot.
    + rewrite Hr, H1. unfold sp. replace (Z.of_nat slot + 1 - 1) with (Z.of_nat slot) by lia.
      rewrite Hm.
      rewrite (exec_p_frame Γ (invert_p body) σ σ' slot Hbd Hnomod)
        by (rewrite pmods_invert; exact (Hnomod f body Hf)).
      exact Hslot.
    + cbn [relab length] in Hst.
      replace (length (pre' ++ (Some (pt f true), CBra (pbt f true))
                         :: prologue_lines (pe f true)))
        with (length pre' + 7)%nat in Hst by (rewrite length_app; simpl; lia).
      exact Hst.
  - (* LPP_One: S1, and the exit test holds *)
    intros b n pa na pc nc R M pre post lo k Hwfa Hwfc Hma Hmc Ea Ec Hb Hn HPe Hpre Hpost
           Hlo Hmodel Hslot Hcl H0 H1 Hcall.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    destruct (compile_p_labels _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_p_labels _ _ _ _ _ Ec) as [Hlec Hinc].
    rewrite HPe.
    eapply (pl_loop_last T pre post lo b n na nc e1 e2 pa pc fin_label k); try eassumption.
    all: try side_p HPe.
    (* S1's run, labeled `from_do` *)
      destruct (IHa (S b) (n + 4)%nat pa na (mkState R M)
                    (pre ++ ops_o lo (flag_block e1 b)
                         ++ [(None, COp (IXori b 1)); (None, CBne b 0%nat fin_label)])
                    (ops_l n (flag_block e2 b) ++ (None, CBeq b 0%nat (S n))
                     :: (None, COp (IXori b 1)) :: (None, CBra (n + 2)%nat)
                     :: (Some (S n), COp nop) :: pc
                     ++ ops (flag_block e1 b)
                     ++ (None, CBne b 0%nat fin_label) :: (None, CBra (n + 3)%nat)
                     :: (Some (n + 2)%nat, COp nop) :: post)
                    (Some (n + 3)%nat) k Hwfa Hma Ea ltac:(unfold scratch in *; lia) ltac:(unfold scratch in *; lia))
        as [ms' [Hst Hrest]].
      { rewrite HPe; unfold loop_code_o; cbn [relab]; norm_app; reflexivity. }
      { lbl_main. }
      { lbl_main. }
      { intros l E; injection E as <-; split; lia. }
      all: try assumption.
      { intros r Hr; apply Hcl; lia. }
      { intro H; apply Hcall; now rewrite H. }
      exists ms'. split; [| exact Hrest].
      rewrite <- HPe. cbn [relab] in Hst. exact Hst.
  - (* LPP_More: S1, the test fails, S2, the re-entry assertion fails, and again *)
    intros b n pa na pc nc R M pre post lo k Hwfa Hwfc Hma Hmc Ea Ec Hb Hn HPe Hpre Hpost
           Hlo Hmodel Hslot Hcl H0 H1 Hcall.
    pose proof Hb as Hb3; unfold scratch in Hb3.
    destruct (compile_p_labels _ _ _ _ _ Ea) as [Hlea Hina].
    destruct (compile_p_labels _ _ _ _ _ Ec) as [Hlec Hinc].
    assert (Hs1 : σ1 slot = 0)
      by (rewrite (exec_p_frame Γ a σ σ1 slot Ha Hnomod Hma); exact Hslot).
    assert (Hs2 : σ2 slot = 0)
      by (rewrite (exec_p_frame Γ c σ1 σ2 slot Hc Hnomod Hmc); exact Hs1).
    assert (HLs : forall l, (n <= l < n + 4)%nat ->
              is_proc T (pre ++ loop_code_o lo fin_label b n e1 e2 pa pc ++ post) l = false
              /\ bot_of T (pre ++ loop_code_o lo fin_label b n e1 e2 pa pc ++ post) (Some l) = None)
      by (intros l Hl; rewrite <- HPe; apply safe_high; lia).
    destruct (pl_loop_round T pre post lo b n na nc e1 e2 pa pc fin_label k
                ltac:(unfold scratch in *; lia) ltac:(unfold scratch in *; lia) Hpre
                ltac:(intros l Hl; apply Hina in Hl; lia)
                ltac:(intros l E; destruct (Hlo l E); lia) HLs
                R M σ σ1 σ2 Hmodel Hcl H0)
      as [M2 [Hst Hm2]].
    { (* S1's run *)
      destruct (IHa (S b) (n + 4)%nat pa na (mkState R M)
                    (pre ++ ops_o lo (flag_block e1 b)
                         ++ [(None, COp (IXori b 1)); (None, CBne b 0%nat fin_label)])
                    (ops_l n (flag_block e2 b) ++ (None, CBeq b 0%nat (S n))
                     :: (None, COp (IXori b 1)) :: (None, CBra (n + 2)%nat)
                     :: (Some (S n), COp nop) :: pc
                     ++ ops (flag_block e1 b)
                     ++ (None, CBne b 0%nat fin_label) :: (None, CBra (n + 3)%nat)
                     :: (Some (n + 2)%nat, COp nop) :: post)
                    (Some (n + 3)%nat) k Hwfa Hma Ea ltac:(unfold scratch in *; lia) ltac:(unfold scratch in *; lia))
        as [ms' [Hst Hrest]].
      { rewrite HPe; unfold loop_code_o; cbn [relab]; norm_app; reflexivity. }
      { lbl_main. }
      { lbl_main. }
      { intros l E; injection E as <-; split; lia. }
      all: try assumption.
      { intros r Hr; apply Hcl; lia. }
      { intro H; apply Hcall; now rewrite H. }
      exists ms'. split; [| exact Hrest].
      rewrite <- HPe. cbn [relab] in Hst. exact Hst. }
    { exact He2. }
    { (* S2's run, with the flag at 0: the registers are those of [do],
         so a call in S2 finds r3, r4, ... clean exactly when one in S1 does *)
      intros M1 Hm1.
      destruct (IHc (S b) na pc nc (mkState R M1)
                    ((((pre ++ ops_o lo (flag_block e1 b)
                              ++ [(None, COp (IXori b 1)); (None, CBne b 0%nat fin_label)])
                         ++ label_first (n + 3)%nat pa ++ ops_l n (flag_block e2 b))
                         ++ [(None, CBeq b 0%nat (S n)); (None, COp (IXori b 1));
                             (None, CBra (n + 2)%nat)])
                         ++ [(Some (S n), COp nop)])
                    (ops (flag_block e1 b)
                     ++ (None, CBne b 0%nat fin_label) :: (None, CBra (n + 3)%nat)
                     :: (Some (n + 2)%nat, COp nop) :: post)
                    None k Hwfc Hmc Ec ltac:(unfold scratch in *; lia) ltac:(unfold scratch in *; lia))
        as [ms' [Hst Hrest]].
      { rewrite HPe; unfold loop_code_o; cbn [relab]; norm_app; reflexivity. }
      { lbl_main. }
      { lbl_main. }
      { intros l E; discriminate. }
      all: try assumption.
      { intros r Hr; apply Hcl; lia. }
      { intros H r Hr. exact (Hcall ltac:(now rewrite H, orb_true_r) r Hr). }
      exists ms'. split; [| exact Hrest].
      rewrite <- HPe. cbn [relab] in Hst. exact Hst. }
    { exact He1. }
    destruct (IHlp b n pa na pc nc R M2 pre post lo k Hwfa Hwfc Hma Hmc Ea Ec Hb Hn HPe
                   Hpre Hpost Hlo Hm2 Hs2 Hcl H0 H1 Hcall)
      as [ms' [Hst' [Hm' Hr']]].
    exists ms'. split; [| split; assumption].
    rewrite HPe in Hst' |- *.
    eapply psteps_trans; [exact Hst | exact Hst'].
Qed.

End MainSpec.

(** ** The whole program

    [whole Γ main k] lays out every procedure and every inverted companion
    ([emit_list]), then `start: START; ADDI r1 k; BRA main; finish:`.
    First, that this layout satisfies [procs_in]. *)

Lemma labels_proc_code : forall e t bt B l,
  In l (labels (proc_code e t bt B)) -> l = t \/ l = e \/ In l (labels B) \/ l = bt.
Proof.
  intros e t bt B l H. rewrite proc_code_eq in H. cbn [labels prologue_lines app] in H.
  rewrite labels_app in H. cbn [labels] in H.
  destruct H as [-> | [-> | H]]; [auto | auto |].
  apply in_app_iff in H as [H | [-> | []]]; auto.
Qed.

Lemma emit_list_labels : forall d bs f n C n',
  emit_list d f bs n = (C, n') ->
  (n <= n')%nat /\
  (forall l, In l (labels C) ->
     (n <= l < n')%nat \/ exists i, (f <= i < f + length bs)%nat /\ is_pl i d l).
Proof.
  intros d; induction bs as [| body bs IH]; intros f n C n' He; cbn [emit_list] in He.
  - injection He as <- <-. split; [lia | intros l []].
  - destruct (compile_p (dirb d body) scratch n) as [B n1] eqn:EB.
    destruct (emit_list d (S f) bs n1) as [rest n2] eqn:Er.
    injection He as <- <-.
    destruct (compile_p_labels _ _ _ _ _ EB) as [Hle1 Hin1].
    destruct (IH _ _ _ _ Er) as [Hle2 Hin2].
    split; [lia |].
    intros l Hl.
    assert (Hl' : In l (labels (proc_code (pe f d) (pt f d) (pbt f d) B)) \/ In l (labels rest))
      by (rewrite <- in_app_iff, <- labels_app; exact Hl).
    clear Hl. destruct Hl' as [Hl | Hl].
    + apply labels_proc_code in Hl as [-> | [-> | [Hl | ->]]].
      * right; exists f; split; [cbn [length]; lia | unfold is_pl; auto].
      * right; exists f; split; [cbn [length]; lia | unfold is_pl; auto].
      * left; apply Hin1 in Hl; lia.
      * right; exists f; split; [cbn [length]; lia | unfold is_pl; auto].
    + destruct (Hin2 l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
      right; exists i; split; [cbn [length]; lia | exact Hpl].
Qed.

Lemma emit_list_in : forall d bs f n C n' j body,
  emit_list d f bs n = (C, n') -> nth_error bs j = Some body ->
  exists C1 C2 B nb nb',
    C = C1 ++ proc_code (pe (f + j)%nat d) (pt (f + j)%nat d) (pbt (f + j)%nat d) B ++ C2 /\
    compile_p (dirb d body) scratch nb = (B, nb') /\ (n <= nb)%nat /\ (nb' <= n')%nat /\
    (forall l, In l (labels C1) ->
       (n <= l < nb)%nat \/ exists i, (f <= i < f + j)%nat /\ is_pl i d l) /\
    (forall l, In l (labels C2) ->
       (nb' <= l < n')%nat \/ exists i, (f + j < i < f + length bs)%nat /\ is_pl i d l).
Proof.
  intros d; induction bs as [| b0 bs IH]; intros f n C n' j body He Hj;
    [destruct j; discriminate |].
  cbn [emit_list] in He.
  destruct (compile_p (dirb d b0) scratch n) as [B0 n1] eqn:EB.
  destruct (emit_list d (S f) bs n1) as [rest n2] eqn:Er.
  injection He as <- <-.
  destruct (compile_p_labels _ _ _ _ _ EB) as [Hle1 Hin1].
  destruct (emit_list_labels _ _ _ _ _ _ Er) as [Hle2 Hin2].
  destruct j as [| j].
  - cbn in Hj. injection Hj as <-.
    exists [], rest, B0, n, n1. rewrite Nat.add_0_r.
    repeat split; try assumption; try lia.
    + intros l [].
    + intros l Hl. destruct (Hin2 l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
      right; exists i; split; [cbn [length]; lia | exact Hpl].
  - cbn in Hj.
    destruct (IH (S f) n1 rest n2 j body Er Hj)
      as [C1 [C2 [B [nb [nb' [HC [HB [Hn [Hn' [HC1 HC2]]]]]]]]]].
    exists (proc_code (pe f d) (pt f d) (pbt f d) B0 ++ C1), C2, B, nb, nb'.
    replace (f + S j)%nat with (S f + j)%nat by lia.
    repeat split; try assumption; try lia.
    + rewrite HC, <- (app_assoc (proc_code (pe f d) (pt f d) (pbt f d) B0) C1). reflexivity.
    + intros l Hl.
      assert (Hl' : In l (labels (proc_code (pe f d) (pt f d) (pbt f d) B0)) \/ In l (labels C1))
        by (rewrite <- in_app_iff, <- labels_app; exact Hl).
      clear Hl. destruct Hl' as [Hl | Hl].
      * apply labels_proc_code in Hl as [-> | [-> | [Hl | ->]]].
        -- right; exists f; split; [lia | unfold is_pl; auto].
        -- right; exists f; split; [lia | unfold is_pl; auto].
        -- left; apply Hin1 in Hl; lia.
        -- right; exists f; split; [lia | unfold is_pl; auto].
      * destruct (HC1 l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
        right; exists i; split; [lia | exact Hpl].
    + intros l Hl. destruct (HC2 l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
      right; exists i; split; [cbn [length]; lia | exact Hpl].
Qed.

(** Labels of the program that are not the labels of procedure [(f, d)]'s
    three lines. *)
Lemma not_pl : forall Γ f d l,
  (f < length Γ)%nat ->
  ((L0 Γ <= l)%nat \/ l = fin_label \/
   exists i d', (i < length Γ)%nat /\ is_pl i d' l /\ (i <> f \/ d' <> d)) ->
  l <> pe f d /\ l <> pt f d /\ l <> pbt f d.
Proof.
  intros Γ f d l Hf H. destruct (pl_lt_L0 Γ f d Hf) as [H1 [H2 H3]].
  destruct H as [H | [-> | [i [d' [Hi [Hpl Hne]]]]]].
  - repeat split; intro; subst; lia.
  - unfold fin_label, pt, pbt, pe; repeat split; lia.
  - repeat split; intro E; subst l;
      [apply pe_inj in Hpl | apply pt_inj in Hpl | apply pbt_inj in Hpl];
      destruct Hpl; destruct Hne; congruence.
Qed.

Lemma procs_in_whole : forall Γ main k, procs_in Γ (whole Γ main k).
Proof.
  intros Γ main k f d body Hf.
  assert (Hlt : (f < length Γ)%nat) by (apply nth_error_Some; congruence).
  unfold whole, procs_code.
  destruct (emit_list false 0 Γ (L0 Γ)) as [Fw n1] eqn:EF.
  destruct (emit_list true 0 Γ n1) as [Iv n2] eqn:EI.
  destruct (emit_list_labels _ _ _ _ _ _ EF) as [HleF HlabF].
  destruct (emit_list_labels _ _ _ _ _ _ EI) as [HleI HlabI].
  assert (Hbnd : forall i d' l, (i < length Γ)%nat -> is_pl i d' l -> (l < L0 Γ)%nat).
  { intros i d' l Hi Hpl. apply is_pl_bound in Hpl. unfold L0; lia. }
  assert (Hstart : forall l, In l (labels (start_code main k)) -> l = fin_label).
  { intros l Hl; cbn in Hl. destruct Hl as [<- | []]; reflexivity. }
  destruct d.
  - destruct (emit_list_in true Γ 0 n1 Iv n2 f body EI Hf)
      as [C1 [C2 [B [nb [nb' [HC [HB [Hn [Hn' [HC1 HC2]]]]]]]]]].
    exists (Fw ++ C1), (C2 ++ start_code main k), B, nb, nb'.
    assert (Hpre : forall l, In l (labels (Fw ++ C1)) ->
              ((L0 Γ <= l < nb)%nat \/
               exists i d', (i < length Γ)%nat /\ is_pl i d' l /\ (i <> f \/ d' <> true))).
    { intros l Hl. rewrite labels_app, in_app_iff in Hl. destruct Hl as [Hl | Hl].
      - destruct (HlabF l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
        right; exists i, false; repeat split; [lia | exact Hpl | right; discriminate].
      - destruct (HC1 l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
        right; exists i, true; repeat split; [lia | exact Hpl | left; lia]. }
    assert (Hpost : forall l, In l (labels (C2 ++ start_code main k)) ->
              ((nb' <= l)%nat \/ l = fin_label \/ exists i, (i < length Γ)%nat /\ is_pl i true l)).
    { intros l Hl. rewrite labels_app, in_app_iff in Hl. destruct Hl as [Hl | Hl].
      - destruct (HC2 l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
        right; right; exists i; split; [cbn in Hi; lia | exact Hpl].
      - right; left; now apply Hstart. }
    repeat split.
    + rewrite HC. cbn [Nat.add]. now rewrite <- !app_assoc.
    + exact HB.
    + unfold L0 in *; lia.
    + intros l Hl. destruct (Hpre l Hl) as [Hr | [i [d' [Hi [Hpl _]]]]]; [lia |].
      apply (Hbnd i d' l Hi) in Hpl. unfold L0 in *; lia.
    + intros l Hl. destruct (Hpost l Hl) as [Hr | [-> | [i [Hi Hpl]]]]; [lia | unfold fin_label, L0 in *; lia |].
      apply (Hbnd i true l Hi) in Hpl. unfold L0 in *; lia.
    + intro Hl. destruct (Hpre _ Hl) as [Hr | Hr].
      * destruct (pl_lt_L0 Γ f true Hlt); lia.
      * destruct (not_pl Γ f true (pe f true) Hlt (or_intror (or_intror Hr))); tauto.
    + intro Hl. destruct (Hpre _ Hl) as [Hr | Hr].
      * destruct (pl_lt_L0 Γ f true Hlt) as [_ [? _]]; lia.
      * destruct (not_pl Γ f true (pt f true) Hlt (or_intror (or_intror Hr))); tauto.
    + intro Hl. destruct (Hpre _ Hl) as [Hr | Hr].
      * destruct (pl_lt_L0 Γ f true Hlt) as [_ [_ ?]]; lia.
      * destruct (not_pl Γ f true (pbt f true) Hlt (or_intror (or_intror Hr))); tauto.
  - destruct (emit_list_in false Γ 0 (L0 Γ) Fw n1 f body EF Hf)
      as [C1 [C2 [B [nb [nb' [HC [HB [Hn [Hn' [HC1 HC2]]]]]]]]]].
    exists C1, (C2 ++ Iv ++ start_code main k), B, nb, nb'.
    assert (Hpre : forall l, In l (labels C1) ->
              ((L0 Γ <= l < nb)%nat \/
               exists i d', (i < length Γ)%nat /\ is_pl i d' l /\ (i <> f \/ d' <> false))).
    { intros l Hl. destruct (HC1 l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
      right; exists i, false; repeat split; [lia | exact Hpl | left; lia]. }
    assert (Hpost : forall l, In l (labels (C2 ++ Iv ++ start_code main k)) ->
              ((nb' <= l)%nat \/ l = fin_label \/
               exists i d', (i < length Γ)%nat /\ is_pl i d' l)).
    { intros l Hl. rewrite !labels_app, !in_app_iff in Hl. destruct Hl as [Hl | [Hl | Hl]].
      - destruct (HC2 l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
        right; right; exists i, false; split; [cbn in Hi; lia | exact Hpl].
      - destruct (HlabI l Hl) as [Hr | [i [Hi Hpl]]]; [left; lia |].
        right; right; exists i, true; split; [cbn in Hi; lia | exact Hpl].
      - right; left; now apply Hstart. }
    repeat split.
    + rewrite HC. cbn [Nat.add]. now rewrite <- !app_assoc.
    + exact HB.
    + lia.
    + intros l Hl. destruct (Hpre l Hl) as [Hr | [i [d' [Hi [Hpl _]]]]]; [lia |].
      apply (Hbnd i d' l Hi) in Hpl. lia.
    + intros l Hl. destruct (Hpost l Hl) as [Hr | [-> | [i [d' [Hi Hpl]]]]];
        [lia | unfold fin_label, L0 in *; lia |].
      apply (Hbnd i d' l Hi) in Hpl. lia.
    + intro Hl. destruct (Hpre _ Hl) as [Hr | Hr].
      * destruct (pl_lt_L0 Γ f false Hlt); lia.
      * destruct (not_pl Γ f false (pe f false) Hlt (or_intror (or_intror Hr))); tauto.
    + intro Hl. destruct (Hpre _ Hl) as [Hr | Hr].
      * destruct (pl_lt_L0 Γ f false Hlt) as [_ [? _]]; lia.
      * destruct (not_pl Γ f false (pt f false) Hlt (or_intror (or_intror Hr))); tauto.
    + intro Hl. destruct (Hpre _ Hl) as [Hr | Hr].
      * destruct (pl_lt_L0 Γ f false Hlt) as [_ [_ ?]]; lia.
      * destruct (not_pl Γ f false (pbt f false) Hlt (or_intror (or_intror Hr))); tauto.
Qed.

(** *** The whole-program theorem

    Started at `start` with an empty call stack, [r1 = 0] and every scratch
    register clean, the fuel executor runs `START; ADDI r1 k; BRA main`,
    the whole call of [main], and halts past `finish` with [br = 0], an
    empty call stack, memory representing the final store, and the start
    registers except [r1 = k].  [k - 1] is the stack cell: a variable no
    procedure assigns ([env_nomod]) that starts at 0.  `codegen.py` puts
    it at [nvars + max(2 * depth, 4) - 1], past every declared variable. *)

Theorem compile_p_program : forall Γ main σ σ' ms slot,
  exec_p Γ (PCall main) σ σ' -> env_wf Γ -> env_nomod Γ slot ->
  models ms σ -> σ slot = 0 ->
  clean_above scratch ms -> regs ms 0%nat = 0 -> regs ms 1%nat = 0 ->
  exists ms' fuel,
    pexec_fuel fuel (ptab (length Γ)) (whole Γ main (Z.of_nat slot + 1))
               (mkC (length (procs_code Γ)) 0 ms, [])
    = Some (mkC (length (whole Γ main (Z.of_nat slot + 1))) 0 ms', [])
    /\ models ms' σ' /\ regs ms' = rupd 1%nat (Z.of_nat slot + 1) (regs ms).
Proof.
  intros Γ main σ σ' [R M] slot Hex Hwf Hnm Hmod Hslot Hcl H0 H1.
  cbn [regs mem] in *.
  set (k := (Z.of_nat slot + 1)%Z).
  set (P := whole Γ main k).
  set (Q := procs_code Γ).
  assert (HP : procs_in Γ P) by apply procs_in_whole.
  set (R1 := rupd 1%nat k R).
  assert (HPe : P = (Q ++ [(None, COp nop); (None, COp (IAddi 1%nat k))])
                    ++ relab None [(None, CBra (pe main false))] ++ [(Some fin_label, COp nop)])
    by (unfold P, whole, start_code; fold Q; cbn [relab]; norm_app; reflexivity).
  destruct (compile_p_spec Γ P slot HP Hwf Hnm (PCall main) σ σ' Hex scratch (L0 Γ)
              [(None, CBra (pe main false))] (L0 Γ) (mkState R1 M)
              (Q ++ [(None, COp nop); (None, COp (IAddi 1%nat k))]) [(Some fin_label, COp nop)]
              None [] I eq_refl eq_refl (le_n _) (le_n _) HPe)
    as [ms' [Hst [Hm Hr]]].
  { intros l _; lia. }
  { intros l _; lia. }
  { intros l E; discriminate. }
  { exact Hmod. }
  { exact Hslot. }
  { intros r Hr; cbn [regs]; unfold R1; rewrite rupd_other by (unfold scratch in Hr; lia).
    now apply Hcl. }
  { cbn [regs]; unfold R1; now rewrite rupd_other by lia. }
  { cbn [regs]; unfold R1; now rewrite rupd_same. }
  { intros _ r Hr; cbn [regs]; unfold R1; rewrite rupd_other by (unfold scratch in Hr; lia).
    now apply Hcl. }
  assert (HlenP : length P = (length Q + 4)%nat)
    by (unfold P, whole, start_code; fold Q; rewrite length_app; reflexivity).
  assert (Hrun : psteps (ptab (length Γ)) P (mkC (length Q) 0 (mkState R M), [])
                        (mkC (length P) 0 ms', [])).
  { (* START *)
    eapply psteps_step.
    { apply (pstep_op _ P (length Q) 0 _ [] None nop).
      eapply nth_error_at. unfold P, whole, start_code; fold Q; reflexivity. }
    rewrite step_nop.
    (* ADDI r1 k *)
    eapply psteps_step.
    { apply (pstep_op _ P (S (length Q)) 0 _ [] None (IAddi 1%nat k)).
      replace (S (length Q)) with (length (Q ++ [(None, COp nop)]))
        by (rewrite length_app; simpl; lia).
      eapply nth_error_at. unfold P, whole, start_code; fold Q; norm_app; reflexivity. }
    cbn [step regs mem]. rewrite H1, Z.add_0_l. fold R1.
    (* BRA main: the call of main *)
    eapply psteps_trans.
    { replace (S (S (length Q))) with (length (Q ++ [(None, COp nop); (None, COp (IAddi 1%nat k))]))
        by (rewrite length_app; simpl; lia).
      exact Hst. }
    (* finish *)
    apply psteps_one.
    rewrite HlenP.
    replace (length (Q ++ [(None, COp nop); (None, COp (IAddi 1%nat k))])
             + length (relab None [(None, CBra (pe main false))]))%nat
      with (length (Q ++ [(None, COp nop); (None, COp (IAddi 1%nat k));
                          (None, CBra (pe main false))]))
      by (rewrite !length_app; cbn [length relab]; lia).
    rewrite (pstep_op _ P _ 0 ms' [] (Some fin_label) nop), step_nop.
    - do 3 f_equal. rewrite !length_app. cbn [length]. lia.
    - eapply nth_error_at. unfold P, whole, start_code; fold Q; norm_app; reflexivity. }
  destruct (psteps_exec_fuel _ _ _ _ Hrun) as [fuel Hf].
  { cbn [fst cpc]. apply nth_error_None. lia. }
  exists ms', fuel. split; [exact Hf |]. split; [exact Hm |].
  rewrite Hr. reflexivity.
Qed.

(** ** Axiom footprint *)

Print Assumptions compile_p_spec.
Print Assumptions compile_p_program.
Print Assumptions procs_in_whole.
Print Assumptions prologue_id.
Print Assumptions exec_p_rev.
Print Assumptions exec_p_det.
Print Assumptions exec_p_frame.
Print Assumptions run_p_sound.

