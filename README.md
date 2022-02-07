# GridapDistributed

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapDistributed.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapDistributed.jl/dev)
![CI](https://github.com/Gridap/GridapDistributed.jl/workflows/CI/badge.svg)
[![DOI](https://zenodo.org/badge/258832236.svg)](https://zenodo.org/badge/latestdoi/258832236)

Parallel distributed-memory version of `Gridap.jl`.

## Purpose

`GridapDistributed.jl` provides a fully-parallel distributed memory extension of the [`Gridap.jl`](https://github.com/gridap/Gridap.jl) library. It allows users to approximate PDEs on parallel computers, from multi-core CPU desktop computers to HPC clusters and supercomputers. The sub-package is designed to be as non-intrusive as possible. As a result, sequential Julia scripts written in the high level API of `Gridap.jl` can be used almost verbatim up to minor adjustments in a parallel context using `GridapDistributed.jl`. 

At present, `GridapDistributed.jl` provides scalable parallel data structures for grid handling,  finite element spaces setup, and distributed linear system assembly. For the latter part, i.e., global distributed sparse matrices and vectors, `GridapDistributed.jl` relies on [`PartitionedArrays.jl`](https://github.com/fverdugo/PartitionedArrays.jl) as distributed linear algebra backend. 

## Documentation

`GridapDistributed.jl` and `Gridap.jl` share almost the same high-level API. We refer to the documentation of `Gridap.jl` for more details about the API. In the example below, we show the minor differences among the APIs of `Gridap.jl` and `GridapDistributed.jl`. We refer to the `test/` folder for additional `GridapDistributed.jl` examples.

`GridapDistributed.jl` driver programs can be either run in sequential execution mode --very useful for developing/debugging parallel programs--, see `test/sequential/` folder for examples, or in message-passing (MPI) execution mode--when you want to deploy the code in the actual parallel computer and perform a fast simulation., see `test/mpi/` folder for examples.

## Example

The following Julia code snippet solves a 2D Poisson problem in parallel on the unit square

```julia
using Gridap
using GridapDistributed
using PartitionedArrays
partition = (2,2)
prun(mpi,partition) do parts
  domain = (0,1,0,1)
  mesh_partition = (4,4)
  model = CartesianDiscreteModel(parts,domain,mesh_partition)
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
```
The domain is discretized using the parallel Cartesian-like mesh generator built-in in 'GridapDistributed`. The only minimal difference with respect to the sequential `Gridap` script is a call to the `prun` function of [`PartitionedArrays.jl`](https://github.com/fverdugo/PartitionedArrays.jl) right at the beginning of the program. With this function, the programer sets up the `PartitionedArrays.jl` communication backend (i.e., MPI in the example), specifies the number of parts and their layout (i.e., 2x2 partition in the example), and provides a function (using Julia do-block syntax for function arguments in the example) to be run on each part. The function body is equivalent to a sequential `Gridap` script, except for the `CartesianDiscreteModel` call, which in `GridapDistributed` also requires the `parts` argument passed back by the `prun` function.


## Remarks 

1. `GridapDistributed.jl` is not a parallel unstructured mesh generator. Grid handling currently available within `GridapDistributed.jl` is restricted to Cartesian-like meshes of arbitrary-dimensional, topologically n-cube domains. See [`GridapP4est.jl`](https://github.com/gridap/GridapP4est.jl), for peta-scale handling of meshes which can be decomposed as forest of quadtrees/octrees of the computational domain, and [`GridapGmsh.jl`](https://github.com/gridap/GridapGmsh.jl) for unstrucuted mesh generation.
2. `GridapDistributed.jl` is not a library of parallel linear solvers at this moment. The linear solver kernel within `GridapDistributed.jl`, leveraged, e.g., via the backslash operator `\`, is just a sparse LU solver applied to the global system gathered on a master task (and thus obviously not scalable, but very useful for testing and debug purposes). It is in our future plans to provide highly scalable linear and nonlinear solvers tailored for the FE discretization of PDEs. For the moment, see [`GridapPETSc.jl`](https://github.com/gridap/GridapPETSc.jl) to use the full set of scalable linear and non-linear solvers in the [PETSc](https://petsc.org/release/) numerical software package. 

## Build 

Before using `GridapDistributed.jl` package, one needs to build the [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) package. We refer to the main documentation of this package for configuration instructions.

## MPI-parallel Julia script execution instructions

In order to execute a MPI-parallel `GridapDistributed.jl` driver, we can leverage the `mpiexecjl` script provided by `MPI.jl`. (Click [here](https://juliaparallel.github.io/MPI.jl/stable/configuration/#Julia-wrapper-for-mpiexec) for installation instructions). As an example, assuming that we are located on the root directory of `GridapDistributed.jl`,
an hypothetic MPI-parallel `GridapDistributed.jl` driver named `driver.jl` can be executed on 4 MPI tasks as:

```
mpiexecjl --project=. -n 4 julia -J sys-image.so driver.jl
```

where `-J sys-image.so` is optional, but highly recommended in order to reduce JIT compilation times. Here, `sys-image.so` is assumed to be a Julia system image pre-generated for the driver at hand using the [`PackageCompiler.jl`](https://julialang.github.io/PackageCompiler.jl/dev/index.html) package. See the `test/TestApp/compile` folder for example scripts with system image generation along with a test application with source available at `test/TestApp/`. These scripts are triggered from `.github/workflows/ci.yml` file on Github CI actions.

## Known issues

A warning when executing MPI-parallel drivers: Data race conditions in the generation of precompiled modules in cache. See [here](https://juliaparallel.github.io/MPI.jl/stable/knownissues/).

## How to cite GridapDistributed

In order to give credit to the `Gridap` and `GridapDistributed` contributors, we simply ask you to cite the `Gridap` main project as indicated [here](https://github.com/gridap/Gridap.jl#how-to-cite-gridap) and the sub-packages you use as indicated in the corresponding repositories. Please, use the reference below in any publication in which you have made use of `GridapDistributed`:

```
@misc{Martin_GridapDistributed_2021,
author = {Martin, Alberto F. and Verdugo, Francesc and Badia, Santiago},
doi = {10.5281/zenodo.5772591},
month = {12},
title = {{GridapDistributed}},
url = {https://github.com/gridap/GridapDistributed.jl},
year = {2021}
}
```

## Contributing to GridapDistributed

GridapDistributed is a collaborative project open to contributions. If you want to contribute, please take into account:

  - Before opening a PR with a significant contribution, contact the project administrators by [opening an issue](https://github.com/gridap/GridapDistributed.jl/issues/new) describing what you are willing to implement. Wait for feedback from other community members.
  - We adhere to the contribution and code-of-conduct instructions of the Gridap.jl project, available [here](https://github.com/gridap/Gridap.jl/blob/master/CONTRIBUTING.md) and [here](https://github.com/gridap/Gridap.jl/blob/master/CODE_OF_CONDUCT.md), resp.  Please, carefully read and follow the instructions in these files.
  - Open a PR with your contribution.

Want to help? We have [issues waiting for help](https://github.com/gridap/GridapDistributed.jl/labels/help%20wanted). You can start contributing to the GridapDistributed project by solving some of those issues.
