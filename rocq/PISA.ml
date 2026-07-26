open BinInt
open List
open ListDef

type reg = int

type addr = int

type state = { regs : (reg -> int); mem : (addr -> int) }

(** val rupd : reg -> int -> (reg -> int) -> reg -> int **)

let rupd r v f r' =
  if (=) r' r then v else f r'

(** val mupd : addr -> int -> (addr -> int) -> addr -> int **)

let mupd a v f a' =
  if Z.eqb a' a then v else f a'

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

(** val step : instr -> state -> state **)

let step i s =
  match i with
  | IAdd (rd, rs) ->
    { regs = (rupd rd (Z.add (s.regs rd) (s.regs rs)) s.regs); mem = s.mem }
  | ISub (rd, rs) ->
    { regs = (rupd rd (Z.sub (s.regs rd) (s.regs rs)) s.regs); mem = s.mem }
  | IXor (rd, rs) ->
    { regs = (rupd rd (Z.coq_lxor (s.regs rd) (s.regs rs)) s.regs); mem =
      s.mem }
  | IAddi (rd, c) ->
    { regs = (rupd rd (Z.add (s.regs rd) c) s.regs); mem = s.mem }
  | ISubi (rd, c) ->
    { regs = (rupd rd (Z.sub (s.regs rd) c) s.regs); mem = s.mem }
  | IXori (rd, c) ->
    { regs = (rupd rd (Z.coq_lxor (s.regs rd) c) s.regs); mem = s.mem }
  | INeg rd -> { regs = (rupd rd (Z.opp (s.regs rd)) s.regs); mem = s.mem }
  | IExch (rd, ra) ->
    let a = s.regs ra in
    { regs = (rupd rd (s.mem a) s.regs); mem = (mupd a (s.regs rd) s.mem) }

(** val run : code -> state -> state **)

let run c s =
  fold_left (fun st i -> step i st) c s

(** val invert_instr : instr -> instr **)

let invert_instr = function
| IAdd (rd, rs) -> ISub (rd, rs)
| ISub (rd, rs) -> IAdd (rd, rs)
| IAddi (rd, c) -> ISubi (rd, c)
| ISubi (rd, c) -> IAddi (rd, c)
| x -> x

(** val invert_code : code -> code **)

let invert_code c =
  rev (map invert_instr c)

(** val zero_state : state **)

let zero_state =
  { regs = (fun _ -> 0); mem = (fun _ -> 0) }
