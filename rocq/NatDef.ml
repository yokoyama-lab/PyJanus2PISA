open PosDef

module N =
 struct
  (** val succ_pos : int -> int **)

  let succ_pos n =
    (fun f0 fp n -> if n=0 then f0 () else fp n)
      (fun _ -> 1)
      (fun p -> Pos.succ p)
      n

  (** val coq_lxor : int -> int -> int **)

  let coq_lxor n m =
    (fun f0 fp n -> if n=0 then f0 () else fp n)
      (fun _ -> m)
      (fun p ->
      (fun f0 fp n -> if n=0 then f0 () else fp n)
        (fun _ -> n)
        (fun q -> Pos.coq_lxor p q)
        m)
      n
 end
