# [Adaptivity](@id adaptivity)

`GridapDistributed.jl` provides an interface for the adaptive mesh refinement (AMR) abstractions defined in `Gridap.jl`. Moreover, we also provide redistribution of meshes.
The framework here works out of the box for Gridap's built-in refinement routines, but also interfaces with [`GridapP4est.jl`](https://github.com/gridap/GridapP4est.jl) for scalable AMR on unstructured meshes.

## Adapted discrete models

`DistributedAdaptedDiscreteModel` represents a distributed mesh obtained by refining a
parent mesh. It wraps both the fine (child) mesh, the coarse (parent) mesh, and a
distributed array of `AdaptivityGlue` objects that record the parent-child cell
relationship.

### Uniform Cartesian refinement

```julia
using Gridap, GridapDistributed, PartitionedArrays

function main(distribute)
  ranks  = distribute(LinearIndices((4,)))
  coarse = CartesianDiscreteModel(ranks, (2,2), (0,1,0,1), (4,4))

  # Refine uniformly by a factor of 2 in every direction
  fine = refine(coarse, 2)   # returns a DistributedAdaptedDiscreteModel

  parent     = get_parent(fine)
  glue_array = get_adaptivity_glue(fine)

  # The fine mesh is used like any other DistributedDiscreteModel
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V     = TestFESpace(fine, reffe)
  uh    = interpolate(x -> sum(x), V)
end

with_debug(main)
```

`refine(model, r)` accepts an integer or a tuple refinement factor; `refine(model, 2)`
halves the cell size in every direction.

## Sub-communicators

AMR workflows sometimes involve different sets of ranks at different stages (e.g. a coarse mesh on 2 ranks that is refined onto 8). Use `generate_subparts` to carve out a sub-communicator of `n` ranks from the full set:

```julia
fine_ranks   = distribute(LinearIndices((8,)))
coarse_ranks = generate_subparts(fine_ranks, 2)  # ranks 1–2 only

if i_am_in(coarse_ranks)
  # only executed on ranks 1 and 2
end
```

Ranks outside the sub-communicator receive an inert placeholder and skip the guarded block.

## Redistribution

After refinement the load may become unbalanced. `redistribute` moves a distributed mesh from one partition layout to another and returns a `RedistributeGlue` that records the data movement:

```julia
# Move from a 2×2 to a 4×1 partition
new_desc       = DistributedCartesianDescriptor(ranks, (4,1), (0,1,0,1), (8,8))
new_model, glue = redistribute(old_model, new_desc)
```

The `RedistributeGlue` can then migrate free-DOF values from the old to the new partition:

```julia
redistribute_free_values(
  new_free_values, new_fespace,
  old_free_values, old_dir_values, old_fespace,
  new_model, glue
)
```

Redistribution for unstructured meshes is handled inside `GridapP4est.jl`.

## API

### Adapted models

```@docs
GridapDistributed.DistributedAdaptedDiscreteModel
```

### Redistribution

```@docs
redistribute
GridapDistributed.RedistributeGlue
```

### Sub-communicators

```@docs
GridapDistributed.generate_subparts
GridapDistributed.i_am_in
```
