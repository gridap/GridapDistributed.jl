There are some things I would like to understand better:

1. What do we need at the geometrical level to run at least all the methods we have for the serial `Gridap`. I think we need some kind of extended triangulation that will permit the computation of face terms, error estimates, etc.
2. How do we plan to include the global numbering into the `FESpace` implementation in parallel?
3. What types of distributed arrays we need in parallel. @fverdugo has created two arrays, `ScatteredVector` and `GhostedVector`. I think that `ScatteredVector` should be extremely simple but with a strong constraint, only provides local access to entries. The `GhostedVector` is far more complicated because entries are shared among processors and one must be able to access the ghost entries. Here I think we should work on the abstract interface of this method and also think about concepts like ummutability and laziness. Do we want to conceptually define a pre-vector that conceptually represents the `GhostedVector` but in other stages. E.g., we could consider a `UncommGhostVector`, `UncommUnassembledGhostVector` that will require to trigger comms when accessed, etc.

More to come...
