# [Backends and Philosophy](@id backends)

`GridapDistributed.jl` is a parallel extension of [`Gridap.jl`](https://github.com/gridap/Gridap.jl) for distributed-memory environments. The central design goal is **minimal API change**: a simulation written for a single process can be made parallel with only a handful of modifications.

## A sequential Poisson problem

Consider the Poisson equation

```math
-\Delta u = f \quad \text{in } \Omega = (0,1)^2, \qquad u = 0 \text{ on } \partial\Omega,
```

solved with Gridap on a single process:

```julia
using Gridap

model = CartesianDiscreteModel((0,1,0,1), (8,8))

reffe = ReferenceFE(lagrangian, Float64, 1)
V  = TestFESpace(model, reffe, dirichlet_tags="boundary")
U  = TrialFESpace(V, x -> 0.0)

Ω  = Triangulation(model)
dΩ = Measure(Ω, 2)

a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
l(v)   = ∫( v*1 )dΩ

op = AffineFEOperator(a, l, U, V)
uh = solve(op)
```

## Converting to distributed

To run the same problem across multiple MPI ranks, two changes are needed:

1. Wrap the driver into a function that accepts a `distribute` function argument and the number of processors `parts`.
2. Distribute the model. In this case, we can use `CartesianDiscreteModel(ranks, parts, domain, cells)`.

Everything else will dispatch on the distributed model, and thus the rest of the driver can be left unchanged:

```julia
using Gridap, GridapDistributed, PartitionedArrays

function poisson(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  model = CartesianDiscreteModel(ranks, parts, (0,1,0,1), (8,8))

  reffe = ReferenceFE(lagrangian, Float64, 1)
  V  = TestFESpace(model, reffe, dirichlet_tags="boundary")
  U  = TrialFESpace(V, x -> 0.0)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω, 2)

  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(v)   = ∫( v*1 )dΩ

  op = AffineFEOperator(a, l, U, V)
  solve(op)
end
```

`distribute` is a function that maps a `LinearIndices` range onto a
set of MPI ranks and returns an array of rank IDs.

## The two backends

### `with_debug` — single-process simulation

Unfortunately, good MPI debugging tools are scarce. Moreover, they never integrate well within Julia's development workflow (that is using `Revise.jl` to edit code and re-run it interactively).

To facilitate development and debugging, `PartitionedArrays.jl` provides a `DebugArray` backend that **simulates multiple MPI ranks inside a single Julia process**. This allows you to print, inspect, and debug distributed objects within the REPL using your ordinary Julia workflow. To use it, wrap your driver function with `with_debug`:

```julia
with_debug() do distribute
  poisson(distribute, (2,2))
end
```

Use this during development and for running tests in CI without MPI.

!!!note
    The only thing `DebugArray` does is wrap serial arrays with `DebugArray`, which is a serial array that does not allow scalar indexing (e.g. `a[i]`). If you know what you are doing, you can directly use arrays and call `poisson(collect, (2,2))`.

### `with_mpi` — true MPI execution

Once the code is working with `with_debug`, you can switch to the real MPI backend by replacing `with_debug` with `with_mpi`. This will run the code across multiple processes using MPI.

Place the following at the bottom of `poisson.jl`:

```julia
with_mpi() do distribute
  poisson(distribute, (2,2))
end
```

```bash
mpiexecjl --project=. -n 4 julia poisson.jl
```
