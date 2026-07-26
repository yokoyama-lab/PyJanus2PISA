open BinInt

type var = int

type store = var -> int

type binop =
| OAdd
| OSub
| OXor

(** val denote : binop -> int -> int -> int **)

let denote o a b =
  match o with
  | OAdd -> Z.add a b
  | OSub -> Z.sub a b
  | OXor -> Z.coq_lxor a b

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
