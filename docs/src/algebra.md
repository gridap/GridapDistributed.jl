# [Algebra](@id algebra)

`GridapDistributed.jl` relies on [`PartitionedArrays.jl`](https://github.com/fverdugo/PartitionedArrays.jl) for all distributed linear algebra. Users interacting with the assembled system — for example, to inspect residuals or feed vectors into a custom solver — will work directly with PartitionedArrays types. This page gives a brief conceptual overview; the [PartitionedArrays documentation](https://fverdugo.github.io/PartitionedArrays.jl/stable/) is the authoritative reference.

## Distributed index ranges: `PRange`

A `PRange` is a distributed range of integers. It is the backbone of every distributed array: it records, for each local index on each rank, its global ID and which rank owns it.

Each rank's slice of a `PRange` is an `AbstractLocalIndices` object that exposes the local index layout:

| Function | Returns |
|---|---|
| `local_to_global(idx)` | Global ID for each local index (owned + ghost) |
| `own_to_global(idx)` | Global IDs of owned (non-ghost) indices |
| `ghost_to_global(idx)` | Global IDs of ghost indices |
| `own_to_local(idx)` | Local positions of owned indices |
| `ghost_to_local(idx)` | Local positions of ghost indices |
| `local_to_owner(idx)` | Owner rank for each local index |

Access the local indices on each rank via `partition`:

```julia
map(partition(prange)) do indices
  lid_to_gid = local_to_global(indices)
end
```

`PRanges` will be used as axes for distributed vectors and matrices, and to define the communication pattern for synchronizing ghost values.

## Distributed vectors: `PVector`

A `PVector` is a distributed dense vector. Each rank stores a local segment that includes both **owned** entries (for which this rank is authoritative) and **ghost** entries (copies of entries owned by neighboring ranks).

Two operations synchronize the distributed state:

- `consistent!(x)` — broadcasts owned values to ghost copies on neighboring ranks. Call
  this after a solve to ensure ghost entries reflect the current solution.
- `assemble!(x)` — accumulates ghost contributions into the owning rank's entry. Used
  during assembly to collect off-rank contributions.

Both return a task; wait for completion with `|> wait` or `fetch`.

A normal workflow would be:

```julia
  map(partition(x)) do x
    # Do something locally with the values of x
    # NO communication is allowed here!
  end
  consistent!(x) |> wait # Synchronize ghost values after local updates
```

## Distributed sparse matrices: `PSparseMatrix`

A `PSparseMatrix` is a distributed sparse matrix stored in compressed form. Each rank owns a block of rows. Non-zero entries in off-rank columns are stored locally and communicated as needed during matrix-vector products (`mul!`).

## GridapDistributed additions

`GridapDistributed.jl` adds a small set of utilities on top of PartitionedArrays to support the FE assembly pipeline.

### API

```@docs
GridapDistributed.permuted_variable_partition
```
