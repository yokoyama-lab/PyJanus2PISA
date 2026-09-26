open BinInt
open Datatypes
open PISA
open Src

(** val op_instr : binop -> reg -> reg -> instr **)

let op_instr o rd rs =
  match o with
  | OAdd -> IAdd (rd, rs)
  | OSub -> ISub (rd, rs)
  | _ -> IXor (rd, rs)

(** val aop_instr : aop -> reg -> reg -> instr **)

let aop_instr o rd rs =
  match o with
  | AAdd -> IAdd (rd, rs)
  | ASub -> ISub (rd, rs)
  | AXor -> IXor (rd, rs)

(** val gen_var : var -> reg -> code **)

let gen_var x rt =
  (IAddi ((Stdlib.Int.succ rt), (Z.of_nat x))) :: ((IExch ((Stdlib.Int.succ
    (Stdlib.Int.succ rt)), (Stdlib.Int.succ rt))) :: ((IXor (rt,
    (Stdlib.Int.succ (Stdlib.Int.succ rt)))) :: ((IExch ((Stdlib.Int.succ
    (Stdlib.Int.succ rt)), (Stdlib.Int.succ rt))) :: ((ISubi
    ((Stdlib.Int.succ rt), (Z.of_nat x))) :: []))))

(** val is_flag_expr : expr -> bool **)

let is_flag_expr = function
| Cst k -> (||) (Z.eqb k 0) (Z.eqb k 1)
| Var _ -> false
| Bin (o, _, _) -> flag_op o

(** val is_nz_test : binop -> expr -> bool **)

let is_nz_test o e2 =
  match o with
  | ONe -> (match e2 with
            | Cst k -> Z.eqb k 0
            | _ -> false)
  | _ -> false

(** val nz_code : (reg -> code) -> (reg -> code) -> reg -> code **)

let nz_code g u r =
  app (g (Stdlib.Int.succ r)) ((ISltx (r, (Stdlib.Int.succ r), 0)) :: ((ISltx
    (r, 0, (Stdlib.Int.succ r))) :: (u (Stdlib.Int.succ r))))

(** val flag_code : bool -> (reg -> code) -> (reg -> code) -> reg -> code **)

let flag_code p g u r =
  if p then g r else nz_code g u r

(** val unflag_code :
    bool -> (reg -> code) -> (reg -> code) -> reg -> code **)

let unflag_code p g u r =
  if p then u r else nz_code g u r

(** val cmp_fwd : binop -> reg -> reg -> reg -> reg -> code **)

let cmp_fwd o rd r1 r2 t =
  match o with
  | OEq ->
    (ISltx (rd, r1, r2)) :: ((ISltx (t, r2, r1)) :: ((IOrx (rd,
      t)) :: ((IXori (rd, 1)) :: [])))
  | ONe ->
    (ISltx (rd, r1, r2)) :: ((ISltx (t, r2, r1)) :: ((IOrx (rd, t)) :: []))
  | OLt -> (ISltx (rd, r1, r2)) :: []
  | OGt -> (ISltx (rd, r2, r1)) :: []
  | OLe -> (ISltx (rd, r2, r1)) :: ((IXori (rd, 1)) :: [])
  | OGe -> (ISltx (rd, r1, r2)) :: ((IXori (rd, 1)) :: [])
  | _ -> []

(** val gen_expr : expr -> reg -> code **)

let rec gen_expr e rt =
  match e with
  | Cst n -> (IAddi (rt, n)) :: []
  | Var x -> gen_var x rt
  | Bin (o, e1, e2) ->
    if arith_op o
    then app (gen_expr e1 rt)
           (app (gen_expr e2 (Stdlib.Int.succ rt))
             (app ((op_instr o rt (Stdlib.Int.succ rt)) :: [])
               (ungen_expr e2 (Stdlib.Int.succ rt))))
    else if is_nz_test o e2
         then nz_code (gen_expr e1) (ungen_expr e1) rt
         else (match o with
               | OAnd ->
                 app
                   (flag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1)
                     (Stdlib.Int.succ rt))
                   (app
                     (flag_code (is_flag_expr e2) (gen_expr e2)
                       (ungen_expr e2) (Stdlib.Int.succ (Stdlib.Int.succ rt)))
                     ((IAndx (rt, (Stdlib.Int.succ rt), (Stdlib.Int.succ
                     (Stdlib.Int.succ rt)))) :: ((IXor ((Stdlib.Int.succ
                     (Stdlib.Int.succ rt)), (Stdlib.Int.succ (Stdlib.Int.succ
                     rt)))) :: [])))
               | OOr ->
                 app
                   (flag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1)
                     (Stdlib.Int.succ rt))
                   (app
                     (flag_code (is_flag_expr e2) (gen_expr e2)
                       (ungen_expr e2) (Stdlib.Int.succ (Stdlib.Int.succ rt)))
                     ((IOrx (rt, (Stdlib.Int.succ rt))) :: ((IOrx (rt,
                     (Stdlib.Int.succ (Stdlib.Int.succ rt)))) :: [])))
               | _ ->
                 app (gen_expr e1 (Stdlib.Int.succ rt))
                   (app (gen_expr e2 (Stdlib.Int.succ (Stdlib.Int.succ rt)))
                     (app
                       (cmp_fwd o rt (Stdlib.Int.succ rt) (Stdlib.Int.succ
                         (Stdlib.Int.succ rt)) (Stdlib.Int.succ
                         (Stdlib.Int.succ (Stdlib.Int.succ rt))))
                       ((IXor ((Stdlib.Int.succ rt), (Stdlib.Int.succ
                       rt))) :: ((IXor ((Stdlib.Int.succ (Stdlib.Int.succ
                       rt)), (Stdlib.Int.succ (Stdlib.Int.succ rt)))) :: [])))))

(** val ungen_expr : expr -> reg -> code **)

and ungen_expr e rt =
  match e with
  | Cst n -> (ISubi (rt, n)) :: []
  | Var x -> invert_code (gen_var x rt)
  | Bin (o, e1, e2) ->
    if arith_op o
    then app (gen_expr e2 (Stdlib.Int.succ rt))
           (app ((invert_instr (op_instr o rt (Stdlib.Int.succ rt))) :: [])
             (app (ungen_expr e2 (Stdlib.Int.succ rt)) (ungen_expr e1 rt)))
    else if is_nz_test o e2
         then nz_code (gen_expr e1) (ungen_expr e1) rt
         else (match o with
               | OAnd ->
                 app
                   (flag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1)
                     (Stdlib.Int.succ rt))
                   (app
                     (flag_code (is_flag_expr e2) (gen_expr e2)
                       (ungen_expr e2) (Stdlib.Int.succ (Stdlib.Int.succ rt)))
                     (app ((IXor ((Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ rt))), (Stdlib.Int.succ
                       rt))) :: ((IAndx ((Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ rt)))),
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       rt))), (Stdlib.Int.succ (Stdlib.Int.succ
                       rt)))) :: ((IXor (rt, (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       rt)))))) :: ((IXor ((Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ rt)))),
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ rt)))))) :: []))))
                       (app
                         (unflag_code (is_flag_expr e2) (gen_expr e2)
                           (ungen_expr e2) (Stdlib.Int.succ (Stdlib.Int.succ
                           rt)))
                         (unflag_code (is_flag_expr e1) (gen_expr e1)
                           (ungen_expr e1) (Stdlib.Int.succ rt)))))
               | OOr ->
                 app
                   (flag_code (is_flag_expr e1) (gen_expr e1) (ungen_expr e1)
                     (Stdlib.Int.succ rt))
                   (app
                     (flag_code (is_flag_expr e2) (gen_expr e2)
                       (ungen_expr e2) (Stdlib.Int.succ (Stdlib.Int.succ rt)))
                     (app ((IXor ((Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ rt))), (Stdlib.Int.succ
                       rt))) :: ((IOrx ((Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       rt))))), (Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ rt))))) :: ((IXor ((Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       rt)))), (Stdlib.Int.succ (Stdlib.Int.succ
                       rt)))) :: ((IOrx ((Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       rt))))), (Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ rt)))))) :: ((IXor
                       (rt, (Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       rt))))))) :: ((IXor ((Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       rt))))), (Stdlib.Int.succ (Stdlib.Int.succ
                       (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                       rt))))))) :: []))))))
                       (app
                         (unflag_code (is_flag_expr e2) (gen_expr e2)
                           (ungen_expr e2) (Stdlib.Int.succ (Stdlib.Int.succ
                           rt)))
                         (unflag_code (is_flag_expr e1) (gen_expr e1)
                           (ungen_expr e1) (Stdlib.Int.succ rt)))))
               | _ ->
                 app (gen_expr e1 (Stdlib.Int.succ rt))
                   (app (gen_expr e2 (Stdlib.Int.succ (Stdlib.Int.succ rt)))
                     (app
                       (cmp_fwd o (Stdlib.Int.succ (Stdlib.Int.succ
                         (Stdlib.Int.succ rt))) (Stdlib.Int.succ rt)
                         (Stdlib.Int.succ (Stdlib.Int.succ rt))
                         (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                         (Stdlib.Int.succ rt)))))
                       (app ((IXor (rt, (Stdlib.Int.succ (Stdlib.Int.succ
                         (Stdlib.Int.succ rt))))) :: ((IXor ((Stdlib.Int.succ
                         (Stdlib.Int.succ (Stdlib.Int.succ rt))),
                         (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
                         rt))))) :: []))
                         (app
                           (ungen_expr e2 (Stdlib.Int.succ (Stdlib.Int.succ
                             rt)))
                           (ungen_expr e1 (Stdlib.Int.succ rt)))))))

(** val gen_assign : var -> aop -> expr -> code **)

let gen_assign x o e =
  app (gen_expr e (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0))))
    (app ((IAddi ((Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
      (Stdlib.Int.succ 0)))), (Z.of_nat x))) :: ((IExch ((Stdlib.Int.succ
      (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
      0))))), (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
      (Stdlib.Int.succ
      0)))))) :: ((aop_instr o (Stdlib.Int.succ (Stdlib.Int.succ
                    (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0)))))
                    (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0)))) :: ((IExch
      ((Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
      (Stdlib.Int.succ 0))))), (Stdlib.Int.succ (Stdlib.Int.succ
      (Stdlib.Int.succ (Stdlib.Int.succ 0)))))) :: ((ISubi ((Stdlib.Int.succ
      (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0)))),
      (Z.of_nat x))) :: [])))))
      (ungen_expr e (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0)))))

(** val gen_swap : var -> var -> code **)

let gen_swap x y =
  (IAddi ((Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0))),
    (Z.of_nat x))) :: ((IAddi ((Stdlib.Int.succ (Stdlib.Int.succ
    (Stdlib.Int.succ (Stdlib.Int.succ 0)))), (Z.of_nat y))) :: ((IExch
    ((Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
    (Stdlib.Int.succ 0))))), (Stdlib.Int.succ (Stdlib.Int.succ
    (Stdlib.Int.succ 0))))) :: ((IExch ((Stdlib.Int.succ (Stdlib.Int.succ
    (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
    0)))))), (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
    (Stdlib.Int.succ 0)))))) :: ((IExch ((Stdlib.Int.succ (Stdlib.Int.succ
    (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
    0)))))), (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
    0))))) :: ((IExch ((Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ
    (Stdlib.Int.succ (Stdlib.Int.succ 0))))), (Stdlib.Int.succ
    (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0)))))) :: ((ISubi
    ((Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0))),
    (Z.of_nat x))) :: ((ISubi ((Stdlib.Int.succ (Stdlib.Int.succ
    (Stdlib.Int.succ (Stdlib.Int.succ 0)))), (Z.of_nat y))) :: [])))))))

(** val compile : stmt -> code **)

let rec compile = function
| Skip -> []
| Assign (x, o, e) -> gen_assign x o e
| Swap (x, y) -> gen_swap x y
| Seq (s1, s2) -> app (compile s1) (compile s2)
