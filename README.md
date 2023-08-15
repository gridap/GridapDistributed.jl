# GridapDistributed

[comment]: [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapDistributed.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapDistributed.jl/dev)
![CI](https://github.com/Gridap/GridapDistributed.jl/workflows/CI/badge.svg)
[![DOI](https://zenodo.org/badge/258832236.svg)](https://zenodo.org/badge/latestdoi/258832236)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02520/status.svg)](https://joss.theoj.org/papers/10.21105/joss.04157)

Parallel distributed-memory version of `Gridap.jl`.

## Purpose

`GridapDistributed.jl` provides a fully-parallel distributed memory extension of the [`Gridap.jl`](https://github.com/gridap/Gridap.jl) library. It allows users to approximate PDEs on parallel computers, from multi-core CPU desktop computers to HPC clusters and supercomputers. The sub-package is designed to be as non-intrusive as possible. As a result, sequential Julia scripts written in the high level API of `Gridap.jl` can be used almost verbatim up to minor adjustments in a parallel context using `GridapDistributed.jl`. 

At present, `GridapDistributed.jl` provides scalable parallel data structures for grid handling,  finite element spaces setup, and distributed linear system assembly. For the latter part, i.e., global distributed sparse matrices and vectors, `GridapDistributed.jl` relies on [`PartitionedArrays.jl`](https://github.com/fverdugo/PartitionedArrays.jl) as distributed linear algebra backend. 

## Documentation

`GridapDistributed.jl` and `Gridap.jl` share almost the same high-level API. We refer to the documentation of `Gridap.jl` for more details about the API. In the example below, we show the minor differences among the APIs of `Gridap.jl` and `GridapDistributed.jl`. We also refer to the following [tutorial](https://gridap.github.io/Tutorials/dev/pages/t016_poisson_distributed/) and the [`GridapDistributed.jl`](https://gridap.github.io/GridapDistributed.jl/dev) documentation for additional examples and rationale.

## Execution modes and how to execute the program in each mode

`GridapDistributed.jl` driver programs can be either run in debug execution mode (very useful for developing/debugging parallel programs, see `test/sequential/` folder for examples) or in message-passing (MPI) execution mode (when you want to deploy the code in the actual parallel computer and perform a fast simulation, see `test/mpi/` folder for examples). In any case, even if you do no have access to a parallel machine, you should be able to run in both modes in your local desktop/laptop. 

A `GridapDistributed.jl` driver program written in debug execution mode as, e.g., the one available at `test/sequential/PoissonTests.jl`, is executed from the terminal just as any other Julia script:

```bash
julia test/sequential/PoissonTests.jl
```

On the other hand, a driver program written in MPI execution mode, such as the one shown in the snippet in the next section, involves an invocation of the `mpiexecjl` script (see [below](https://github.com/gridap/GridapDistributed.jl/edit/master/README.md#mpi-parallel-julia-script-execution-instructions](https://github.com/gridap/GridapDistributed.jl#mpi-parallel-julia-script-execution-instructions))):

```
mpiexecjl -n 4 julia gridap_distributed_mpi_mode_example.jl
```

with the appropriate number of MPI tasks, `-n 4` in this particular example.

## Simple example (MPI-parallel execution mode)

The following Julia code snippet solves a 2D Poisson problem in parallel on the unit square. The example follows the MPI-parallel execution mode (note the `with_mpi()` function call) and thus it must be executed on 4 MPI tasks (note the mesh is partitioned into 4 parts) using the instructions [below](https://github.com/gridap/GridapDistributed.jl#mpi-parallel-julia-script-execution-instructions). If a user wants to use the debug execution mode, one just replaces `with_mpi()` by `with_debug()`. 
`GridapDistributed.jl` debug execution mode scripts are executed as any other julia sequential script.

```julia
using Gridap
using GridapDistributed
using PartitionedArrays
function main(ranks)
  domain = (0,1,0,1)
  mesh_partition = (2,2) 
  mesh_cells = (4,4)
  model = CartesianDiscreteModel(ranks,mesh_partition,domain,mesh_cells)
  order = 2
  u((x,y)) = (x+y)^order
  f(x) = -Δ(u,x)
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(v) = ∫( v*f )dΩ
  op = AffineFEOperator(a,l,U,V)
  uh = solve(op)
  writevtk(Ω,"results",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
end
with_mpi() do distribute 
  ranks = distribute_with_mpi(LinearIndices((4,)))
  main(ranks)
end
```
The domain is discretized using the parallel Cartesian-like mesh generator built-in in `GridapDistributed`. The only minimal difference with respect to the sequential `Gridap` script is a call to the `with_mpi` function of [`PartitionedArrays.jl`](https://github.com/fverdugo/PartitionedArrays.jl) right at the beginning of the program. With this function, the programmer sets up the `PartitionedArrays.jl` communication backend (i.e., MPI in the example), specifies, and provides a function to be run on each part (using Julia do-block syntax in the example). The function body is equivalent to a sequential `Gridap` script, except for the `CartesianDiscreteModel` call, which in `GridapDistributed` also requires the `ranks` and `mesh_partition` arguments to this function.

## Using parallel solvers

`GridapDistributed.jl` is _not_ a library of parallel linear solvers. The linear solver kernel within `GridapDistributed.jl`, defined with the backslash operator `\`, is just a sparse LU solver applied to the global system gathered on a master task (not scalable, but very useful for testing and debug purposes). 

We provide the full set of scalable linear and nonlinear solvers in the [PETSc](https://petsc.org/release/) library in [`GridapPETSc.jl`](https://github.com/gridap/GridapPETSc.jl). For an example which combines `GridapDistributed` with `GridapPETSc.jl`, we refer to the following [tutorial](https://gridap.github.io/Tutorials/dev/pages/t016_poisson_distributed/). Additional examples can be found in the `test/` folder of `GridapPETSc`. Other linear solver libraries on top of `GridapDistributed` can be developed in the future. 

## Partitioned meshes

`GridapDistributed.jl` provides a built-in parallel generator of Cartesian-like meshes of arbitrary-dimensional, topologically n-cube domains. 

Distributed unstructured meshes are generated using [`GridapGmsh.jl`](https://github.com/gridap/GridapGmsh.jl). We also refer to [`GridapP4est.jl`](https://github.com/gridap/GridapP4est.jl), for peta-scale handling of meshes which can be decomposed as forest of quadtrees/octrees of the computational domain. Examples of distributed solvers that combine all these building blocks can be found in the following [tutorial](https://gridap.github.io/Tutorials/dev/pages/t016_poisson_distributed/).

## A more complex example  (MPI-parallel execution mode)

In the following example, we combine `GridapDistributed` (for the parallel implementation of the PDE discretisation), `GridapGmsh` (for the distributed unstructured mesh), and `GridapPETSc` (for the linear solver step). The mesh file can be found [here](https://github.com/gridap/Tutorials/blob/master/models/demo.msh).

```julia
using Gridap
using GridapGmsh
using GridapPETSc
using GridapDistributed
using PartitionedArrays
function main(ranks)
  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    model = GmshDiscreteModel(ranks,"demo.msh")
    order = 1
    dirichlet_tags = ["boundary1","boundary2"]
    u_boundary1(x) = 0.0
    u_boundary2(x) = 1.0
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags=dirichlet_tags)
    U = TrialFESpace(V,[u_boundary1,u_boundary2])
    Ω = Interior(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(u)⋅∇(v) )dΩ
    l(v) = 0
    op = AffineFEOperator(a,l,U,V)
    solver = PETScLinearSolver()
    uh = solve(solver,op)
    writevtk(Ω,"demo",cellfields=["uh"=>uh])
  end
end
with_mpi() do distribute 
  ranks = distribute_with_mpi(LinearIndices((6,)))
  main(ranks)
end
```

## Build 

Before using `GridapDistributed.jl` package, one needs to build the [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) package. We refer to the main documentation of this package for configuration instructions.

## MPI-parallel Julia script execution instructions

In order to execute a MPI-parallel `GridapDistributed.jl` driver, we can leverage the `mpiexecjl` script provided by `MPI.jl`. (Click [here](https://juliaparallel.org/MPI.jl/stable/usage/#Julia-wrapper-for-mpiexec) for installation instructions). As an example, assuming that we are located on the root directory of `GridapDistributed.jl`,
an hypothetical MPI-parallel `GridapDistributed.jl` driver named `driver.jl` can be executed on 4 MPI tasks as:

```
mpiexecjl --project=. -n 4 julia -J sys-image.so driver.jl
```

where `-J sys-image.so` is optional, but highly recommended in order to reduce JIT compilation times. Here, `sys-image.so` is assumed to be a Julia system image pre-generated for the driver at hand using the [`PackageCompiler.jl`](https://julialang.github.io/PackageCompiler.jl/dev/index.html) package. See the `test/TestApp/compile` folder for example scripts with system image generation along with a test application with source available at `test/TestApp/`. These scripts are triggered from `.github/workflows/ci.yml` file on Github CI actions.

## Known issues

A warning when executing MPI-parallel drivers: Data race conditions in the generation of precompiled modules in cache. See [here](https://juliaparallel.github.io/MPI.jl/stable/knownissues/).

## How to cite GridapDistributed

In order to give credit to the `Gridap` and `GridapDistributed` contributors, we simply ask you to cite the `Gridap` main project as indicated [here](https://github.com/gridap/Gridap.jl#how-to-cite-gridap) and the sub-packages you use as indicated in the corresponding repositories. Please, use the reference below in any publication in which you have made use of `GridapDistributed`:

```
@article{Badia2022,
  doi = {10.21105/joss.04157},
  url = {https://doi.org/10.21105/joss.04157},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {74},
  pages = {4157},
  author = {Santiago Badia and Alberto F. Martín and Francesc Verdugo},
  title = {GridapDistributed: a massively parallel finite element toolbox in Julia},
  journal = {Journal of Open Source Software}
}
```

## Contributing to GridapDistributed

GridapDistributed is a collaborative project open to contributions. If you want to contribute, please take into account:

  - Before opening a PR with a significant contribution, contact the project administrators by [opening an issue](https://github.com/gridap/GridapDistributed.jl/issues/new) describing what you are willing to implement. Wait for feedback from other community members.
  - We adhere to the contribution and code-of-conduct instructions of the Gridap.jl project, available [here](https://github.com/gridap/Gridap.jl/blob/master/CONTRIBUTING.md) and [here](https://github.com/gridap/Gridap.jl/blob/master/CODE_OF_CONDUCT.md), resp.  Please, carefully read and follow the instructions in these files.
  - Open a PR with your contribution.

Want to help? We have [issues waiting for help](https://github.com/gridap/GridapDistributed.jl/labels/help%20wanted). You can start contributing to the GridapDistributed project by solving some of those issues.
