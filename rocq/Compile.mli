open BinInt
open Datatypes
open PISA
open Src

val op_instr : binop -> reg -> reg -> instr

val aop_instr : aop -> reg -> reg -> instr

val gen_var : var -> reg -> code

val is_flag_expr : expr -> bool

val is_nz_test : binop -> expr -> bool

val nz_code : (reg -> code) -> (reg -> code) -> reg -> code

val flag_code : bool -> (reg -> code) -> (reg -> code) -> reg -> code

val unflag_code : bool -> (reg -> code) -> (reg -> code) -> reg -> code

val cmp_fwd : binop -> reg -> reg -> reg -> reg -> code

val gen_expr : expr -> reg -> code

val ungen_expr : expr -> reg -> code

val gen_assign : var -> aop -> expr -> code

val gen_swap : var -> var -> code

val compile : stmt -> code
