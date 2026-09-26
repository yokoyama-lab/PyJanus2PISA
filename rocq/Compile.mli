open BinInt
open Datatypes
open PISA
open Src

val op_instr : binop -> reg -> reg -> instr

val aop_instr : aop -> reg -> reg -> instr

val cst_code : int -> reg -> code

val imm_code : binop -> int -> reg -> code

val const_value : expr -> int option

val fold_csts : binop -> expr -> expr -> int option

val gen_var : var -> reg -> code

val is_flag_expr : expr -> bool

val is_nz_test : binop -> expr -> bool

val is_logic : binop -> bool

val nz_code : (reg -> code) -> reg -> code

val flag_gen : expr -> (reg -> code) -> reg -> code

val comb_code : binop -> reg -> reg -> reg -> code

val comb_block : binop -> (reg -> code) -> (reg -> code) -> reg -> code

val gen_expr : expr -> reg -> code

val ungen_expr : expr -> reg -> code

val gen_assign : var -> aop -> expr -> code

val gen_swap : var -> var -> code

val compile : stmt -> code
