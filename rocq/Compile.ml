open BinInt
open Datatypes
open PISA
open Src

(** val op_instr : binop -> reg -> reg -> instr **)

let op_instr o rd rs =
  match o with
  | OAdd -> IAdd (rd, rs)
  | OSub -> ISub (rd, rs)
  | OXor -> IXor (rd, rs)

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

(** val gen_expr : expr -> reg -> code **)

let rec gen_expr e rt =
  match e with
  | Cst n -> (IAddi (rt, n)) :: []
  | Var x -> gen_var x rt
  | Bin (o, e1, e2) ->
    app (gen_expr e1 rt)
      (app (gen_expr e2 (Stdlib.Int.succ rt))
        (app ((op_instr o rt (Stdlib.Int.succ rt)) :: [])
          (invert_code (gen_expr e2 (Stdlib.Int.succ rt)))))

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
      (invert_code
        (gen_expr e (Stdlib.Int.succ (Stdlib.Int.succ (Stdlib.Int.succ 0))))))

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
