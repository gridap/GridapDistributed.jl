# [Assembly](@id assembly)

`GridapDistributed.jl` provides three assembly strategies that control how local
contributions from different ranks are combined into the global system. These are implemented as Gridap `AssemblyStrategy` instances, and can be specified via the `strategy` argument of `SparseMatrixAssembler`.

```julia
assem = SparseMatrixAssembler(U, V, strategy)
op = AffineFEOperator(a, l, U, V, assem)
```

## `Assembled` (default)

Standard parallel assembly:

- Each rank integrates over its **owned** cells. Contributions to owned rows coming from other ranks are communicated and assembled.
- The result is a fully assembled system, where each processor holds the correct portions of the global matrix and vector for its owned rows.
- Extra rows corresponding to ghost DOFs are allocated as cache for in-place re-assembly (e.g. for nonlinear or transient problems), but may contain garbage at all times.

```julia
assem = SparseMatrixAssembler(U, V, Assembled())
```

Unless you know what you are doing, this is the assembly strategy you should use.

## `SubAssembled`

- Each rank integrates over its **owned** cells, and assembles the local contributions for both owned and ghost rows. No communication of ghost contributions is performed.
- The result is a sub-assembled system, where local matrices are incomplete (sub-assembled). As a consequence, matrix-vector products require assembly of the resulting vector (which makes it a bit more expensive than the `Assembled` case).
- The advantage: The row and column layout of the resulting system is closer to the DOF layout of the FESpace. This is ideal for Domain Decomposition methods (or other geometrical solvers).

```julia
assem = SparseMatrixAssembler(U, V, SubAssembled())
```

## `LocallyAssembled`

- Each rank integrates over its **owned and ghost** cells. Contributions to ghost rows are discarded, and no communication is performed. **This assumes that all local contributions to owned rows can be computed without communication**. This is in general **NOT** true.
- If the invariant holds, the result is a fully assembled system like in the `Assembled` case, but without the communication cost.
- No extra rows corresponding to ghost DOFs are required for in-place re-assembly.

```julia
assem = SparseMatrixAssembler(U, V, LocallyAssembled())
```

!!! warning
    `LocallyAssembled` produces incorrect results if the assumption above is violated.
    Use only when you are sure that owned-row contributions are purely local.

## Changing the sparse matrix format

The default format is `SparseMatrixCSC`. To use a different format (e.g. CSR for
better SpMV performance), pass a custom assembler:

```julia
using SparseMatricesCSR
assem = SparseMatrixAssembler(
  SparseMatrixCSR{0,Float64,Int},
  Vector{Float64},
  U, V, strategy
)
op = AffineFEOperator(a, l, U, V, assem)
```

## API

### Strategies

```@docs
Assembled
SubAssembled
LocallyAssembled
```

### Builder types

```@docs
GridapDistributed.DistributedArrayBuilder
GridapDistributed.DistributedCounter
GridapDistributed.DistributedAllocation
```
