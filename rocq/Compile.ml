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

(** val cst_code : int -> reg -> code **)

let cst_code n rt =
  if Z.eqb n 0
  then []
  else if Z.ltb 0 n
       then (IAddi (rt, n)) :: []
       else (ISubi (rt, (Z.opp n))) :: []

(** val imm_code : binop -> int -> reg -> code **)

let imm_code o k rt =
  match o with
  | OSub -> cst_code (Z.opp k) rt
  | OXor -> if Z.eqb k 0 then [] else (IXori (rt, k)) :: []
  | _ -> cst_code k rt

(** val const_value : expr -> int option **)

let rec const_value = function
| Cst n -> Some n
| Var _ -> None
| Bin (o, e1, e2) ->
  if arith_op o
  then (match const_value e1 with
        | Some a ->
          (match const_value e2 with
           | Some b -> Some (denote o a b)
           | None -> None)
        | None -> None)
  else None

(** val fold_csts : binop -> expr -> expr -> int option **)

let fold_csts o e1 e2 =
  match e1 with
  | Cst a -> (match e2 with
              | Cst b -> Some (denote o a b)
              | _ -> None)
  | _ -> None

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

(** val is_logic : binop -> bool **)

let is_logic = function
| OAnd -> true
| OOr -> true
| _ -> false

(** val nz_code : (reg -> code) -> reg -> code **)

let nz_code g r =
  app (g (Stdlib.Int.succ r))
    (app ((ISltx (r, (Stdlib.Int.succ r), 0)) :: ((ISltx (r, 0,
      (Stdlib.Int.succ r))) :: [])) (invert_code (g (Stdlib.Int.succ r))))

(** val flag_gen : expr -> (reg -> code) -> reg -> code **)

let flag_gen e g =
  if is_flag_expr e
  then g
  else (match e with
        | Cst k -> cst_code (b2z (negb (Z.eqb k 0)))
        | _ -> nz_code g)

(** val comb_code : binop -> reg -> reg -> reg -> code **)

let comb_code o re rl rr =
  match o with
  | OEq ->
    (ISltx (re, rl, rr)) :: ((ISltx (re, rr, rl)) :: ((IXori (re, 1)) :: []))
  | ONe -> (ISltx (re, rl, rr)) :: ((ISltx (re, rr, rl)) :: [])
  | OLt -> (ISltx (re, rl, rr)) :: []
  | OGt -> (ISltx (re, rr, rl)) :: []
  | OLe -> (ISltx (re, rr, rl)) :: ((IXori (re, 1)) :: [])
  | OGe -> (ISltx (re, rl, rr)) :: ((IXori (re, 1)) :: [])
  | OAnd -> (IAndx (re, rl, rr)) :: []
  | OOr -> (IOrx (re, rl, rr)) :: []
  | _ -> []

(** val comb_block :
    binop -> (reg -> code) -> (reg -> code) -> reg -> code **)

let comb_block o g1 g2 rt =
  app (g1 (Stdlib.Int.succ rt))
    (app (g2 (Stdlib.Int.succ (Stdlib.Int.succ rt)))
      (app
        (comb_code o rt (Stdlib.Int.succ rt) (Stdlib.Int.succ
          (Stdlib.Int.succ rt)))
        (app (invert_code (g2 (Stdlib.Int.succ (Stdlib.Int.succ rt))))
          (invert_code (g1 (Stdlib.Int.succ rt))))))

(** val gen_expr : expr -> reg -> code **)

let rec gen_expr e rt =
  match e with
  | Cst n -> cst_code n rt
  | Var x -> gen_var x rt
  | Bin (o, e1, e2) ->
    (match fold_csts o e1 e2 with
     | Some k -> cst_code k rt
     | None ->
       if is_nz_test o e2
       then nz_code (gen_expr e1) rt
       else if arith_op o
            then app (gen_expr e1 rt)
                   (match const_value e2 with
                    | Some k -> imm_code o k rt
                    | None ->
                      app (gen_expr e2 (Stdlib.Int.succ rt))
                        (app ((op_instr o rt (Stdlib.Int.succ rt)) :: [])
                          (invert_code (gen_expr e2 (Stdlib.Int.succ rt)))))
            else if is_logic o
                 then comb_block o (flag_gen e1 (gen_expr e1))
                        (flag_gen e2 (gen_expr e2)) rt
                 else comb_block o (gen_expr e1) (gen_expr e2) rt)

(** val ungen_expr : expr -> reg -> code **)

let ungen_expr e rt =
  invert_code (gen_expr e rt)

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
