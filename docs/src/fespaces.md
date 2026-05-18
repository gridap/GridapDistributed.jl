# [FE Spaces and Cell Data](@id fespaces)

## Distributed FESpaces

`DistributedSingleFieldFESpace` is the parallel counterpart of Gridap's
`SingleFieldFESpace`. It is constructed with the same syntax:

```julia
reffe = ReferenceFE(lagrangian, Float64, 1)
V = TestFESpace(model, reffe, dirichlet_tags="boundary")
U = TrialFESpace(V, u_dirichlet)
```

where `model` is a `DistributedDiscreteModel`.

Internally, each rank holds a local `FESpace` on its local portion of the mesh
(owned + ghost cells). These are accessible via `local_views`:

```julia
map(local_views(V)) do local_V
  println("Local ndofs = ", num_free_dofs(local_V))
end
```

This API also extends to multi-field spaces.

## Global DOF numbering and ownership

Global DOF IDs are stored as a `PRange` (see [Algebra](@ref algebra)), which can be accessed via `get_free_dof_gids(V)`. Some general considerations:

- DOFs are **owned by the rank that owns the cell(s) they are attached to**. DOFs on the interface between two partitions are owned by the rank **with the highest rank ID**. This convention ensures that each DOF is owned by exactly one rank and appears as a ghost DOF on all neighboring ranks.
- The global DOF numbering is such that DOFs owned by rank 1 come first, followed by those owned by rank 2, and so on. This means that the global DOF IDs are contiguous within each rank's owned portion, which is convenient for assembly and solver interfaces.
- We preserve the local DOF numbering of the serial `FESpace` on each rank. That means that locally, owned DOFs are not necessarily (locally) numbered before ghost DOFs.

## Ghost DOF consistency

Within the library, we internally maintain the consistency of ghost DOF values as needed. When implementing your own low-level operations, it is important to do the same. In particular, before using an `FEFunction` for assembly it is important to ensure that ghost DOF values are consistent with their owned counterparts. See for instance the following example:

```julia
uh = zero(U)
r(v) = ∫( v*uh )dΩ

x = get_free_dof_values(uh)
map(partition(x)) do x
  # Do something locally with the values of x
  # After this, ghost values may be inconsistent!
end

b = assemble_vector(r, V) # May be WRONG!

consistent!(x) |> wait
b = assemble_vector(r, V) # OK
```

## API

### Abstract interface

```@docs
GridapDistributed.DistributedFESpace
```

### Single-field spaces

```@docs
GridapDistributed.DistributedSingleFieldFESpace
```

### Multi-field spaces

```@docs
GridapDistributed.DistributedMultiFieldFESpace
```

### Cell data

```@docs
GridapDistributed.DistributedCellField
GridapDistributed.DistributedMeasure
GridapDistributed.DistributedDomainContribution
GridapDistributed.DistributedCellPoint
```

### FE functions

```@docs
GridapDistributed.DistributedFEFunctionData
GridapDistributed.DistributedSingleFieldFEFunction
```

### Constant FE space

```@docs
Gridap.FESpaces.ConstantFESpace(::GridapDistributed.DistributedDiscreteModel; constraint_type, kwargs...)
```

### DOF numbering utilities

```@docs
GridapDistributed.generate_gids
GridapDistributed.generate_posneg_gids
GridapDistributed.generate_gids_by_color
GridapDistributed.split_gids_by_color
GridapDistributed.vcat_gids
```
