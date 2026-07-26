open BinInt
open List
open ListDef

type reg = int

type addr = int

type state = { regs : (reg -> int); mem : (addr -> int) }

val rupd : reg -> int -> (reg -> int) -> reg -> int

val mupd : addr -> int -> (addr -> int) -> addr -> int

type instr =
| IAdd of reg * reg
| ISub of reg * reg
| IXor of reg * reg
| IAddi of reg * int
| ISubi of reg * int
| IXori of reg * int
| INeg of reg
| IExch of reg * reg

type code = instr list

val step : instr -> state -> state

val run : code -> state -> state

val invert_instr : instr -> instr

val invert_code : code -> code

val zero_state : state
