open NatDef
open PosDef

module Z :
 sig
  val double : int -> int

  val succ_double : int -> int

  val pred_double : int -> int

  val pos_sub : int -> int -> int

  val add : int -> int -> int

  val opp : int -> int

  val sub : int -> int -> int

  val eqb : int -> int -> bool

  val of_nat : int -> int

  val of_N : int -> int

  val coq_lxor : int -> int -> int
 end
