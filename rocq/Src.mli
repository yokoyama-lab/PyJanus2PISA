open BinInt

type var = int

type store = var -> int

type binop =
| OAdd
| OSub
| OXor

val denote : binop -> int -> int -> int

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
