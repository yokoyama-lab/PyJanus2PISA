open BinInt
open Datatypes

type var = int

type store = var -> int

type binop =
| OAdd
| OSub
| OXor
| OEq
| ONe
| OLt
| OGt
| OLe
| OGe
| OAnd
| OOr

(** val b2z : bool -> int **)

let b2z = function
| true -> 1
| false -> 0

(** val denote : binop -> int -> int -> int **)

let denote o a b =
  match o with
  | OAdd -> Z.add a b
  | OSub -> Z.sub a b
  | OXor -> Z.coq_lxor a b
  | OEq -> b2z (Z.eqb a b)
  | ONe -> b2z (negb (Z.eqb a b))
  | OLt -> b2z (Z.ltb a b)
  | OGt -> b2z (Z.ltb b a)
  | OLe -> b2z (Z.leb a b)
  | OGe -> b2z (Z.leb b a)
  | OAnd -> b2z ((&&) (negb (Z.eqb a 0)) (negb (Z.eqb b 0)))
  | OOr -> b2z ((||) (negb (Z.eqb a 0)) (negb (Z.eqb b 0)))

(** val arith_op : binop -> bool **)

let arith_op = function
| OAdd -> true
| OSub -> true
| OXor -> true
| _ -> false

(** val flag_op : binop -> bool **)

let flag_op o =
  negb (arith_op o)

type expr =
| Cst of int
| Var of var
| Bin of binop * expr * expr

type aop =
| AAdd
| ASub
| AXor

type stmt =
| Skip
| Assign of var * aop * expr
| Swap of var * var
| Seq of stmt * stmt

(** val eval : store -> expr -> int **)

let rec eval s = function
| Cst n -> n
| Var x -> s x
| Bin (o, e1, e2) -> denote o (eval s e1) (eval s e2)
