# [Visualization](@id visualization)

`GridapDistributed.jl` supports parallel VTK output via `writevtk`, using the same
interface as sequential Gridap. Each MPI rank writes its local portion as a `.vtu` file;
the VTK parallel format (`.pvtu`) is written by rank 0 and references all pieces.
Open the `.pvtu` file in [ParaView](https://www.paraview.org/) to visualise the global
solution.

## Static output

```julia
writevtk(Ω, "solution", cellfields=["uh" => uh, "eh" => eh])
```

This generates `solution.pvtu` (the parallel descriptor) and one `solution_n.vtu` per
rank. Ghost cells are omitted by default; each cell appears in exactly one `.vtu` file.

## Time-dependent output

Use `createpvd` / `savepvd` to collect snapshots from a time loop into a single `.pvd`
collection that ParaView can animate:

```julia
pvd = createpvd(get_parts(model), "simulation")

for (t, uh) in time_steps
  pvd[t] = createvtk(Ω, "solution_$(t)", cellfields=["uh" => uh])
end

savepvd(pvd)
```

`get_parts(model)` provides the rank array needed to coordinate parallel file I/O.

## API

```@docs
GridapDistributed.DistributedPvd
GridapDistributed.DistributedVisualizationData
```
