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

val b2z : bool -> int

val denote : binop -> int -> int -> int

val arith_op : binop -> bool

val flag_op : binop -> bool

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

val eval : store -> expr -> int
