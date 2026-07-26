open BinInt
open Datatypes
open PISA
open Src

val op_instr : binop -> reg -> reg -> instr

val aop_instr : aop -> reg -> reg -> instr

val gen_var : var -> reg -> code

val gen_expr : expr -> reg -> code

val gen_assign : var -> aop -> expr -> code

val gen_swap : var -> var -> code

val compile : stmt -> code
