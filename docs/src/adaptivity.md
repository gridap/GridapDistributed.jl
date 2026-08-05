# [Adaptivity](@id adaptivity)

`GridapDistributed.jl` provides an interface for the adaptive mesh refinement (AMR)
abstractions defined in `Gridap.jl`. The framework works out of the box for Gridap's
built-in refinement routines. For scalable AMR on unstructured meshes with full MPI
performance, see [`GridapP4est.jl`](https://github.com/gridap/GridapP4est.jl), which
integrates the p4est library with `GridapDistributed.jl`.

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

AMR workflows sometimes involve different sets of ranks at different stages (e.g. a coarse
mesh on 2 ranks that is refined onto 8). Use `generate_subparts` to carve out a
sub-communicator of `n` ranks from the full set:

```julia
fine_ranks   = distribute(LinearIndices((8,)))
coarse_ranks = generate_subparts(fine_ranks, 2)  # ranks 1–2 only

if i_am_in(coarse_ranks)
  # only executed on ranks 1 and 2
end
```

Ranks outside the sub-communicator receive an inert placeholder and skip the guarded block.

## API

### Adapted models

```@docs
GridapDistributed.DistributedAdaptedDiscreteModel
```

### Sub-communicators

```@docs
GridapDistributed.generate_subparts
GridapDistributed.i_am_in
```

### Refinement internals

```@docs
GridapDistributed.refine_local_models
GridapDistributed.refine_cell_gids
```

### Redistribution

```@docs
GridapDistributed.RedistributeGlue
GridapDistributed.redistribute
GridapDistributed.redistribute_cartesian
GridapDistributed.redistribute_array_by_cells
GridapDistributed.redistribution_local_indices
GridapDistributed.redistribute_indices
```
