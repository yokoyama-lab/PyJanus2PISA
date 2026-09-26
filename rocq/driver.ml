(* driver.ml — run the extracted verified compiler on a few programs.

   Prints, for each program, the compiled PISA code and the final variable
   values as computed by the *verified* machine model (PISA.run).  The output
   is consumed by ../tools/rocq_diff.py, which replays the same instructions on
   the Python interpreter and compiles the same source with codegen.py, so that
   three things can be compared:

     verified compiler + verified machine   (this program)
     verified compiler + pisa_interp.py     (checks the Python interpreter)
     codegen.py        + pisa_interp.py     (checks the Python compiler)

   Build: see Makefile.driver *)

let instr_to_string (i : PISA.instr) : string =
  match i with
  | PISA.IAdd  (rd, rs) -> Printf.sprintf "ADD %d %d"  rd rs
  | PISA.ISub  (rd, rs) -> Printf.sprintf "SUB %d %d"  rd rs
  | PISA.IXor  (rd, rs) -> Printf.sprintf "XOR %d %d"  rd rs
  | PISA.IAddi (rd, c)  -> Printf.sprintf "ADDI %d %d" rd c
  | PISA.ISubi (rd, c)  -> Printf.sprintf "SUBI %d %d" rd c
  | PISA.IXori (rd, c)  -> Printf.sprintf "XORI %d %d" rd c
  | PISA.INeg  rd       -> Printf.sprintf "NEG %d"     rd
  | PISA.IExch (rd, ra) -> Printf.sprintf "EXCH %d %d" rd ra
  | PISA.ISltx (rd, rs, rt) -> Printf.sprintf "SLTX %d %d %d" rd rs rt
  | PISA.IOrx  (rd, rs) -> Printf.sprintf "ORX %d %d"  rd rs
  | PISA.IAndx (rd1, rd2, rs) -> Printf.sprintf "ANDX %d %d %d" rd1 rd2 rs

(* Programs are given together with the Janus source text that
   tools/rocq_diff.py feeds to codegen.py, so the two sides cannot drift. *)
type case = {
  name    : string;
  janus   : string;          (* source for the Python compiler *)
  ast     : Src.stmt;        (* the same program as an AST *)
  nvars   : int;
}

(* Milestone 3: comparisons and && / || (x, y are variables 0 and 1;
   z0..z5 are 2..7, each receiving one operator). *)
let seqs (l : Src.stmt list) : Src.stmt =
  match Stdlib.List.rev l with
  | [] -> Src.Skip
  | last :: rest -> Stdlib.List.fold_left (fun acc s -> Src.Seq (s, acc)) last rest

let v i = Src.Var i
let c k = Src.Cst k
let bin o a b = Src.Bin (o, a, b)
let add x e = Src.Assign (x, Src.AAdd, e)
let decls = "int x\nint y\nint z0\nint z1\nint z2\nint z3\nint z4\nint z5\nprocedure main\n"

(* one program per (x, y) pair: z0..z5 += x = y, x != y, x < y, x > y, x <= y, x >= y *)
let cmp_prog (name : string) (x0 : int) (y0 : int) : case =
  let init k v0 = if v0 >= 0 then (Printf.sprintf "%s += %d" k v0, Src.Assign ((if k = "x" then 0 else 1), Src.AAdd, c v0))
                  else (Printf.sprintf "%s -= %d" k (- v0), Src.Assign ((if k = "x" then 0 else 1), Src.ASub, c (- v0))) in
  let (jx, ax) = init "x" x0 and (jy, ay) = init "y" y0 in
  let ops = [ ("=", Src.OEq); ("!=", Src.ONe); ("<", Src.OLt);
              (">", Src.OGt); ("<=", Src.OLe); (">=", Src.OGe) ] in
  let lines = Stdlib.List.mapi (fun i (s, _) -> Printf.sprintf "z%d += x %s y" i s) ops in
  let asts = Stdlib.List.mapi (fun i (_, o) -> add (i + 2) (bin o (v 0) (v 1))) ops in
  { name; janus = decls ^ "  " ^ Stdlib.String.concat "\n  " (jx :: jy :: lines);
    ast = seqs (ax :: ay :: asts); nvars = 8 }

let cmp_cases : case list = [
  cmp_prog "cmp_lt" 3 5;
  cmp_prog "cmp_eq" 4 4;
  cmp_prog "cmp_gt_neg" (-2) (-7);
  { name  = "logical";
    (* 1 && 2 = 1 (bitwise 0), 2 || 0 = 1 (bitwise 2), nested and non-Boolean operands *)
    janus = decls ^ "  x += 1\n  y += 2\n  z0 += x && y\n  z1 += y || 0\n"
            ^ "  z2 += (x - 1) && y\n  z3 += (x - 1) || (y - 2)\n"
            ^ "  z4 += (x < y) && (y != 0)\n  z5 += ((x = y) || (x <= 1)) + 5";
    ast   = seqs [ add 0 (c 1); add 1 (c 2);
                   add 2 (bin Src.OAnd (v 0) (v 1));
                   add 3 (bin Src.OOr (v 1) (c 0));
                   add 4 (bin Src.OAnd (bin Src.OSub (v 0) (c 1)) (v 1));
                   add 5 (bin Src.OOr (bin Src.OSub (v 0) (c 1)) (bin Src.OSub (v 1) (c 2)));
                   add 6 (bin Src.OAnd (bin Src.OLt (v 0) (v 1)) (bin Src.ONe (v 1) (c 0)));
                   add 7 (bin Src.OAdd (bin Src.OOr (bin Src.OEq (v 0) (v 1))
                                                    (bin Src.OLe (v 0) (c 1))) (c 5)) ];
    nvars = 8 };
  { name  = "cmp_nested";
    (* comparisons of comparisons, and a comparison as the right operand of + *)
    janus = decls ^ "  x -= 3\n  y += 3\n  z0 += (x < y) = (y > x)\n  z1 += 1 + (x >= y)\n"
            ^ "  z2 += (x != 0) ^ (y != 0)\n  z3 += ((x < 0) && (y > 0)) || (x = y)";
    ast   = seqs [ Src.Assign (0, Src.ASub, c 3); add 1 (c 3);
                   add 2 (bin Src.OEq (bin Src.OLt (v 0) (v 1)) (bin Src.OGt (v 1) (v 0)));
                   add 3 (bin Src.OAdd (c 1) (bin Src.OGe (v 0) (v 1)));
                   add 4 (bin Src.OXor (bin Src.ONe (v 0) (c 0)) (bin Src.ONe (v 1) (c 0)));
                   add 5 (bin Src.OOr (bin Src.OAnd (bin Src.OLt (v 0) (c 0)) (bin Src.OGt (v 1) (c 0)))
                                      (bin Src.OEq (v 0) (v 1))) ];
    nvars = 8 };
]

let cases : case list = [
  { name  = "assign_const";
    janus = "int x\nprocedure main\n  x += 3";
    ast   = Src.Assign (0, Src.AAdd, Src.Cst 3);
    nvars = 1 };

  { name  = "assign_sub";
    janus = "int x\nprocedure main\n  x += 7\n  x -= 2";
    ast   = Src.Seq (Src.Assign (0, Src.AAdd, Src.Cst 7),
                     Src.Assign (0, Src.ASub, Src.Cst 2));
    nvars = 1 };

  { name  = "assign_var";
    janus = "int x\nint y\nprocedure main\n  x += 3\n  y += x";
    ast   = Src.Seq (Src.Assign (0, Src.AAdd, Src.Cst 3),
                     Src.Assign (1, Src.AAdd, Src.Var 0));
    nvars = 2 };

  { name  = "assign_expr";
    janus = "int x\nint y\nprocedure main\n  x += 3\n  y += x + 2";
    ast   = Src.Seq (Src.Assign (0, Src.AAdd, Src.Cst 3),
                     Src.Assign (1, Src.AAdd,
                                 Src.Bin (Src.OAdd, Src.Var 0, Src.Cst 2)));
    nvars = 2 };

  { name  = "xor_assign";
    janus = "int x\nint y\nprocedure main\n  x += 12\n  y += 10\n  x ^= y";
    ast   = Src.Seq (Src.Assign (0, Src.AAdd, Src.Cst 12),
                     Src.Seq (Src.Assign (1, Src.AAdd, Src.Cst 10),
                              Src.Assign (0, Src.AXor, Src.Var 1)));
    nvars = 2 };

  { name  = "swap";
    janus = "int x\nint y\nprocedure main\n  x += 7\n  y += 2\n  x <=> y";
    ast   = Src.Seq (Src.Assign (0, Src.AAdd, Src.Cst 7),
                     Src.Seq (Src.Assign (1, Src.AAdd, Src.Cst 2),
                              Src.Swap (0, 1)));
    nvars = 2 };

  { name  = "crosscheck";   (* the program used by tools/pyjanus_crosscheck.py *)
    janus = "int x\nint y\nprocedure main\n  x += 3\n  y += x + 2\n  x <=> y";
    ast   = Src.Seq (Src.Assign (0, Src.AAdd, Src.Cst 3),
                     Src.Seq (Src.Assign (1, Src.AAdd,
                                          Src.Bin (Src.OAdd, Src.Var 0, Src.Cst 2)),
                              Src.Swap (0, 1)));
    nvars = 2 };

  { name  = "nested_expr";
    janus = "int x\nint y\nint z\nprocedure main\n  x += 4\n  y += 5\n  z += x + y - 2";
    ast   = Src.Seq (Src.Assign (0, Src.AAdd, Src.Cst 4),
                     Src.Seq (Src.Assign (1, Src.AAdd, Src.Cst 5),
                              Src.Assign (2, Src.AAdd,
                                Src.Bin (Src.OSub,
                                  Src.Bin (Src.OAdd, Src.Var 0, Src.Var 1),
                                  Src.Cst 2))));
    nvars = 3 };
] @ cmp_cases


let () =
  Stdlib.List.iter (fun c ->
    let code  = Compile.compile c.ast in
    let final = PISA.run code PISA.zero_state in
    Printf.printf "CASE %s\n" c.name;
    Printf.printf "SOURCE %s\n" (Stdlib.String.concat "\\n" (Stdlib.String.split_on_char '\n' c.janus));
    Printf.printf "NVARS %d\n" c.nvars;
    Stdlib.List.iter (fun i -> Printf.printf "I %s\n" (instr_to_string i)) code;
    for v = 0 to c.nvars - 1 do
      Printf.printf "VAR %d %d\n" v (final.PISA.mem v)
    done;
    (* cleanliness, as computed by the verified machine *)
    for r = 3 to 31 do
      Printf.printf "REG %d %d\n" r (final.PISA.regs r)
    done;
    Printf.printf "END\n")
    cases
