(** * TestProc.v — executable checks for procedures, including calls in S2

    Variables [x0 x1 x2 c y0 y1 d] at addresses [0 .. 6]; the stack cell is
    variable 7, so [r1 = k = 8] (`codegen.py` would use
    [k = nvars + max(2 * depth, 4)]; the cross-check passes its own [k]).
    For each program, [run_src] runs the source with the fuel interpreter
    [run_p] (sound for [exec_p], [run_p_exec]) and [run_mach] runs
    [whole Γ main 8] on the machine of PISAProc.v from `start`; the two
    stores must agree, and the machine must end past `finish` with
    [br = 0], an empty call stack, r2..r8 = 0 and r1 = 8.  The same
    programs are run through `codegen.py` / `pisa_interp.py` by
    tools/rocq_proc_crosscheck.py. *)

From Stdlib Require Import ZArith List Lia Bool.
Require Import PISA Src Compile PISACtl CompileIf CompileLoop PISAProc SrcProc CompileProc.
Import ListNotations.
Open Scope Z_scope.

Definition zero_store : store := fun _ => 0.

Definition obs_store (σ : store) : list Z := map σ (seq 0 7).

Definition run_src (Γ : penv) (main : pname) : option (list Z) :=
  option_map obs_store (run_p 400 Γ (PCall main) zero_store).

Definition start_state (Γ : penv) : pstate := (mkC (length (procs_code Γ)) 0 zero_state, []).

Definition mach_fuel : nat := Nat.pow 2 17.

Definition mach_result (Γ : penv) (main : pname) : option pstate :=
  pexec_fuel mach_fuel (ptab (length Γ)) (whole Γ main 8) (start_state Γ).

(** [(halted past finish, br, stack depth, store, r2..r8, r1)] *)
Definition run_mach (Γ : penv) (main : pname) :=
  match mach_result Γ main with
  | Some (c, k) =>
      Some (Nat.eqb (cpc c) (length (whole Γ main 8)), cbr c, length k,
            map (fun a => mem (cst c) (Z.of_nat a)) (seq 0 7),
            map (regs (cst c)) (seq 2 7), regs (cst c) 1%nat)
  | None => None
  end.

Arguments PCall f%_nat_scope.
Arguments PUncall f%_nat_scope.
Arguments Var x%_nat_scope.
Arguments Assign x%_nat_scope o e.
Arguments run_src Γ main%_nat_scope.
Arguments run_mach Γ main%_nat_scope.

Definition inc (x : var) (k : Z) : pstmt := PBase (Assign x AAdd (Cst k)).
Definition dec (x : var) (k : Z) : pstmt := PBase (Assign x ASub (Cst k)).
Arguments inc x%_nat_scope k%_Z_scope.
Arguments dec x%_nat_scope k%_Z_scope.
(** [x1 <=> x2 ; x0 <=> x1] *)
Definition rotp : pstmt := PBase (Seq (Swap 1%nat 2%nat) (Swap 0%nat 1%nat)).

(** A clean end: past `finish`, [br = 0], empty stack, r2..r8 = 0, r1 = 8,
    and the store the source computes. *)
Definition agrees (Γ : penv) (main : pname) : bool :=
  match run_src Γ main, run_mach Γ main with
  | Some s, Some (fin, br, depth, m, rs, r1) =>
      fin && (br =? 0) && Nat.eqb depth 0 && forallb (fun v => v =? 0) rs && (r1 =? 8)
      && forallb (fun p => fst p =? snd p) (combine s m) && Nat.eqb (length s) (length m)
  | _, _ => false
  end.
Arguments agrees Γ main%_nat_scope.
Arguments whole Γ main%_nat_scope k%_Z_scope.

(** *** 1. [call]: [f: c += 1; x0 += 2], [main: call f; call f] *)
Definition g_call : penv := [ PSeq (inc 3 1) (inc 0 2); PSeq (PCall 0) (PCall 0) ].
Example ex_call_src : run_src g_call 1 = Some [4; 0; 0; 2; 0; 0; 0].
Proof. vm_compute. reflexivity. Qed.
Example ex_call : agrees g_call 1 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 2. [uncall]: [f: c += 3; x1 ^= c], [main: call f; call f; uncall f] *)
Definition g_uncall : penv :=
  [ PSeq (inc 3 3) (PBase (Assign 1 AXor (Var 3)))
  ; PSeq (PCall 0) (PSeq (PCall 0) (PUncall 0)) ].
Example ex_uncall_src : run_src g_uncall 1 = Some [0; 3; 0; 3; 0; 0; 0].
Proof. vm_compute. reflexivity. Qed.
Example ex_uncall : agrees g_uncall 1 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 3. Nested calls, and the uncall of a procedure that calls:
    [f: c += 1], [g: call f; d += 1; call f], [main: call g; call g; uncall g]
    (`g_inv` uncalls [f], so `f_inv` runs too). *)
Definition g_nested : penv :=
  [ inc 3 1
  ; PSeq (PCall 0) (PSeq (inc 6 1) (PCall 0))
  ; PSeq (PCall 1) (PSeq (PCall 1) (PUncall 1)) ].
Example ex_nested_src : run_src g_nested 2 = Some [0; 0; 0; 2; 0; 0; 1].
Proof. vm_compute. reflexivity. Qed.
Example ex_nested : agrees g_nested 2 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 4. A call as the first statement of a loop body (so `from_do`
    labels the `BRA f`): [x0 += 1; from x0 do call f loop rot until x2; call f] *)
Definition g_loop : penv :=
  [ inc 3 1
  ; PSeq (inc 0 1) (PSeq (PLoop (Var 0) (PCall 0) rotp (Var 2)) (PCall 0)) ].
Example ex_loop_src : run_src g_loop 1 = Some [0; 0; 1; 4; 0; 0; 0].
Proof. vm_compute. reflexivity. Qed.
Example ex_loop : agrees g_loop 1 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 5. Calls in both branches of an [if], and an [uncall] in one:
    [x2 += 1; if x2 then call f else skip fi c; if x0 then call f else uncall f fi x0] *)
Definition g_if : penv :=
  [ inc 3 10
  ; PSeq (inc 2 1)
         (PSeq (PIf (Var 2) (PCall 0) (PBase Skip) (Var 3))
               (PIf (Var 0) (PCall 0) (PUncall 0) (Var 0))) ].
Example ex_if_src : run_src g_if 1 = Some [0; 0; 1; 0; 0; 0; 0].
Proof. vm_compute. reflexivity. Qed.
Example ex_if : agrees g_if 1 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 6. Recursion, forwards and backwards:
    [f: if x0 then x0 -= 1; c += 1; call f; x0 += 1 else skip fi x0],
    [main: x0 += 3; call f; call f; uncall f] — [f] adds [x0] to [c];
    `f_inv` recurses through `uncall f`. *)
Definition g_rec : penv :=
  [ PIf (Var 0) (PSeq (dec 0 1) (PSeq (inc 3 1) (PSeq (PCall 0) (inc 0 1))))
        (PBase Skip) (Var 0)
  ; PSeq (inc 0 3) (PSeq (PCall 0) (PSeq (PCall 0) (PUncall 0))) ].
Example ex_rec_src : run_src g_rec 1 = Some [3; 0; 0; 3; 0; 0; 0].
Proof. vm_compute. reflexivity. Qed.
Example ex_rec : agrees g_rec 1 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 7. A call inside an [if] inside a loop body:
    [x0 += 1; from x0 do (if x2 then call f else call f fi x2) loop rot until x2] *)
Definition g_loop_if : penv :=
  [ inc 6 1
  ; PSeq (inc 0 1)
         (PSeq (PLoop (Var 0) (PIf (Var 2) (PCall 0) (PCall 0) (Var 2)) rotp (Var 2))
               (PCall 0)) ].
Example ex_loop_if_src : run_src g_loop_if 1 = Some [0; 0; 1; 0; 0; 0; 4].
Proof. vm_compute. reflexivity. Qed.
Example ex_loop_if : agrees g_loop_if 1 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 8. A procedure whose Janus name is another one's companion:
    [f: x0 += 1; x0 += 1], [f_inv: x1 += 100],
    [main: call f; call f; call f_inv; call f_inv; uncall f].
    Here labels are numbers and `f_inv` (procedure 1) and the inverted
    companion of [f] ([pe 0 true]) are different labels, so the verified
    layout is correct ([x0 = 2], [x1 = 200], as in Janus).  `codegen.py`
    names the companion by the string `f + "_inv"`, which is the user's
    procedure: both calls of `f_inv` and the `uncall f` reach the same
    code (see MANIFEST.md and tools/rocq_proc_crosscheck.py). *)
Definition g_finv : penv :=
  [ PSeq (inc 0 1) (inc 0 1)
  ; inc 1 100
  ; PSeq (PCall 0) (PSeq (PCall 0) (PSeq (PCall 1) (PSeq (PCall 1) (PUncall 0)))) ].
Example ex_finv_src : run_src g_finv 2 = Some [2; 200; 0; 0; 0; 0; 0].
Proof. vm_compute. reflexivity. Qed.
Example ex_finv : agrees g_finv 2 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 9. Calls in the [loop] part (S2) of a loop:
    [f: c += 5]; [main: x0 += 1; from x0 do skip loop call f; rot until x2;
    call f].  Janus: S2 runs twice, then the final call — [c = 15].  This
    used to be the counterexample of the S2 restriction: `_gen_from` entered
    S2 with the loop flag r3 = 1 ([loop: XORI rt 1]), and [f]'s body,
    compiled at base r3 like every procedure body, computed [5 + r3] into
    r3, so the verified layout ended with [c = 17] (and `codegen.py`, whose
    straight-line code differs, with [c = 5, d = 10]).  Now `loop:` is a NOP
    and S2 runs with the flag at 0: the layout gives [c = 15], and the
    program is in the scope of [compile_p_spec] ([s2_call_works] below). *)
Definition g_s2 : penv :=
  [ inc 3 5
  ; PSeq (inc 0 1)
         (PSeq (PLoop (Var 0) (PBase Skip) (PSeq (PCall 0) rotp) (Var 2)) (PCall 0)) ].
Example ex_s2_src : run_src g_s2 1 = Some [0; 0; 1; 15; 0; 0; 0].
Proof. vm_compute. reflexivity. Qed.
Example ex_s2 : agrees g_s2 1 = true.
Proof. vm_compute. reflexivity. Qed.

(** *** 10. Recursion through S2, forwards and backwards:
    [h: d += 1; if y0 then y0 -= 1; y1 += 1;
          from y1 do skip loop y1 -= 1; call h; x1 += 1 until x1;
          x1 -= 1; y0 += 1
        else skip fi y0]
    — at each level the loop runs S2 once (the entry sees [y1 = 1], the
    re-entry [y1 = 0], the second exit test [x1 = 1]), and S2 calls [h] one
    level down.  [main: y0 += 2; call h; uncall h; call h]: [h] runs at
    levels 2, 1, 0, so [d = 3] after each call; the [uncall] runs `h_inv`,
    whose loop (the inverse of [h]'s) has `uncall h` in its S2. *)
Definition g_rec_s2 : penv :=
  [ PSeq (inc 6 1)
         (PIf (Var 4)
              (PSeq (dec 4 1) (PSeq (inc 5 1)
               (PSeq (PLoop (Var 5) (PBase Skip)
                            (PSeq (dec 5 1) (PSeq (PCall 0) (inc 1 1)))
                            (Var 1))
                     (PSeq (dec 1 1) (inc 4 1)))))
              (PBase Skip) (Var 4))
  ; PSeq (inc 4 2) (PSeq (PCall 0) (PSeq (PUncall 0) (PCall 0))) ].
Example ex_rec_s2_src : run_src g_rec_s2 1 = Some [0; 0; 0; 0; 2; 0; 3].
Proof. vm_compute. reflexivity. Qed.
Example ex_rec_s2 : agrees g_rec_s2 1 = true.
Proof. vm_compute. reflexivity. Qed.

(** Every program above is within the theorem's scope ([wf_p] on every
    body, decided by [wf_pb]). *)
Fixpoint wf_sb (s : stmt) : bool :=
  match s with
  | Skip | Assign _ _ _ => true
  | Swap x y => negb (Nat.eqb x y)
  | Seq a b => wf_sb a && wf_sb b
  end.

Fixpoint wf_pb (st : pstmt) : bool :=
  match st with
  | PBase s => wf_sb s
  | PSeq a c | PIf _ a c _ | PLoop _ a c _ => wf_pb a && wf_pb c
  | PCall _ | PUncall _ => true
  end.

Lemma wf_sb_sound : forall s, wf_sb s = true -> wf_stmt s.
Proof.
  induction s; cbn; intro H; auto.
  - apply negb_true_iff, Nat.eqb_neq in H. exact H.
  - apply andb_true_iff in H as [H1 H2]. auto.
Qed.

Lemma wf_pb_sound : forall st, wf_pb st = true -> wf_p st.
Proof.
  induction st; cbn; intro H; auto;
    repeat match goal with H : (_ && _)%bool = true |- _ => apply andb_true_iff in H as [? ?] end;
    auto using wf_sb_sound.
Qed.

Lemma env_wf_of : forall Γ, forallb wf_pb Γ = true -> env_wf Γ.
Proof.
  intros Γ H f body Hf. apply wf_pb_sound.
  rewrite forallb_forall in H. apply H. eapply nth_error_In; exact Hf.
Qed.

Lemma env_nomod_of : forall Γ x, forallb (fun st => negb (pmods st x)) Γ = true -> env_nomod Γ x.
Proof.
  intros Γ x H f body Hf. rewrite forallb_forall in H.
  apply negb_true_iff, H. eapply nth_error_In; exact Hf.
Qed.

Example ex_envs_wf :
  forallb (forallb wf_pb)
    [g_call; g_uncall; g_nested; g_loop; g_if; g_rec; g_loop_if; g_finv; g_s2; g_rec_s2]
  = true.
Proof. reflexivity. Qed.

(** ** A call in S2, through the main theorem

    [g_s2] (a call in S2, formerly excluded by [wf_p]) is now an instance
    of [compile_p_program]: the source has an execution, every execution
    ends with [c = 15], and the verified layout, run from `start` by the
    fuel executor, halts past `finish` with memory representing that store
    and every register but [r1 = 8] clean. *)

Theorem s2_call_works :
  (exists σ', exec_p g_s2 (PCall 1) zero_store σ') /\
  (forall σ', exec_p g_s2 (PCall 1) zero_store σ' ->
     σ' 3%nat = 15 /\
     exists ms' fuel,
       pexec_fuel fuel (ptab (length g_s2)) (whole g_s2 1 8) (start_state g_s2)
       = Some (mkC (length (whole g_s2 1 8)) 0 ms', [])
       /\ models ms' σ' /\ regs ms' = rupd 1%nat 8 (regs zero_state)).
Proof.
  destruct (run_p 400 g_s2 (PCall 1) zero_store) as [σ0 |] eqn:E;
    [| vm_compute in E; discriminate].
  assert (Hex : exec_p g_s2 (PCall 1) zero_store σ0) by (eapply run_p_exec; exact E).
  assert (H3 : σ0 3%nat = 15).
  { pose proof ex_s2_src as H. unfold run_src in H. rewrite E in H.
    unfold option_map, obs_store in H. cbn [map seq] in H.
    injection H as _ _ _ H3 _ _ _. exact H3. }
  split; [now exists σ0 |].
  intros σ' H. pose proof (exec_p_det _ _ _ _ Hex σ' H) as Hdet. subst σ'.
  split; [exact H3 |].
  destruct (compile_p_program g_s2 1%nat zero_store σ0 zero_state 7%nat Hex)
    as [ms' [fuel [Hf [Hm Hr]]]].
  - apply env_wf_of. reflexivity.
  - apply env_nomod_of. reflexivity.
  - intro x; reflexivity.
  - reflexivity.
  - intros r _; reflexivity.
  - reflexivity.
  - reflexivity.
  - exists ms', fuel. split; [exact Hf | split; [exact Hm | exact Hr]].
Qed.

Print Assumptions s2_call_works.
