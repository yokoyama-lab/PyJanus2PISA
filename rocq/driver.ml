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

(* Programs are given together with the Janus source text that
   tools/rocq_diff.py feeds to codegen.py, so the two sides cannot drift. *)
type case = {
  name    : string;
  janus   : string;          (* source for the Python compiler *)
  ast     : Src.stmt;        (* the same program as an AST *)
  nvars   : int;
}

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
]

let () =
  List.iter (fun c ->
    let code  = Compile.compile c.ast in
    let final = PISA.run code PISA.zero_state in
    Printf.printf "CASE %s\n" c.name;
    Printf.printf "SOURCE %s\n" (String.concat "\\n" (String.split_on_char '\n' c.janus));
    Printf.printf "NVARS %d\n" c.nvars;
    List.iter (fun i -> Printf.printf "I %s\n" (instr_to_string i)) code;
    for v = 0 to c.nvars - 1 do
      Printf.printf "VAR %d %d\n" v (final.PISA.mem v)
    done;
    (* cleanliness, as computed by the verified machine *)
    for r = 3 to 8 do
      Printf.printf "REG %d %d\n" r (final.PISA.regs r)
    done;
    Printf.printf "END\n")
    cases
