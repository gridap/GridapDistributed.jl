# GridapDistributed

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapDistributed.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapDistributed.jl/dev)
![CI](https://github.com/Gridap/GridapDistributed.jl/workflows/CI/badge.svg)

Parallel distributed-memory version of `Gridap.jl`.  🚧 work in progress 🚧

| ![](https://user-images.githubusercontent.com/38347633/134634010-2be9b499-201b-4166-80ac-e161f6adceb0.png)   |  ![](https://user-images.githubusercontent.com/38347633/134634023-83f37646-f6b9-435c-9f9f-291dea9f86c2.png) 
|:-------------:|:-------------:|


## Purpose

This package is currently **experimental, under development**. 

`GridapDistributed.jl` provides fully-parallel distributed memory data structures for the Finite Element (FE) numerical solution of Partial Differential Equations (PDEs) on parallel computers, from multi-core CPU desktop computers, to HPC clusters and supercomputers. These distributed data structures are designed to mirror as far as possible their counterparts in the [`Gridap.jl`](https://github.com/gridap/Gridap.jl) software package, while implementing/leveraging most of their abstract interfaces. As a result, sequential Julia scripts written in the high level API of `Gridap.jl` can be used almost verbatim up to minor adjustments in a parallel context using `GridapDistributed.jl`. This is indeed one of the main advantages of `GridapDistributed.jl` and a major design goal that we pursue. 

At present, `GridapDistributed.jl` provides scalable parallel data structures for grid handling,  finite element spaces setup, and distributed linear system assembly. For the latter part, i.e., global distributed sparse matrices and vectors, `GridapDistributed.jl` relies on [`PartitionedArrays.jl`](https://github.com/fverdugo/PartitionedArrays.jl) as distributed linear algebra backend. This implies, among others, that all `GridapDistributed.jl` driver programs can be either run in sequential execution mode--very useful for developing/debugging parallel programs--, see `test/sequential/` folder for examples, or in message-passing (MPI) execution mode--when you want to deploy the code in the actual parallel computer and perform a fast simulation., see `test/mpi/` folder for examples.

## Caveats 

At present, we have the following caveats:
1. Grid handling currently available within `GridapDistributed.jl` is restricted to Cartesian-like meshes of arbitrary-dimensional, topologically n-cube domains. 
2. The linear solver kernel within `GridapDistributed.jl`, leveraged, e.g., via the backslash operator `\`, is just a sparse LU solver applied to the global system gathered on a master task (and thus obviously not scalable). It is in our future plans to provide highly scalable linear and nonlinear solvers tailored for the FE discretization of PDEs.

 Complementarily, one may leverage two satellite packages of `GridapDistributed.jl` to address 1. and 2., namely [`GridapP4est.jl`](https://github.com/gridap/GridapP4est.jl), for peta-scale handling of meshes which can be decomposed as forest of quadtrees/octrees of the computational domain., and  [`GridapPETSc.jl`](https://github.com/gridap/GridapPETSc.jl), which offers to the full set of scalable linear and non-linear solvers in the [PETSc](https://petsc.org/release/) numerical software package. We refer to the readme of these two packages for further details.

## Build 

Before using `GridapDistributed.jl` package, one needs to build the [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) package. We refer to the main documentation of this package for configuration instructions.

## MPI-parallel Julia script execution instructions

In order to execute a MPI-parallel `GridapDistributed.jl` driver, we have first to figure out the path of the `mpirun` script corresponding to the MPI library with which `MPI.jl` was built. In order to do so, we can run the following commands from the root directory of  `GridapDistributed.jl` git repo:

```
$ julia --project=. -e "using MPI;println(MPI.mpiexec_path)" 
/home/amartin/.julia/artifacts/2fcd463fb9498f362be9d1c4ef70a63c920b0e96/bin/mpiexec
```

Alternatively, for convenience, one can assign the path of `mpirun` to an environment variable, i.e.

```
$ export MPIRUN=$(julia --project=. -e "using MPI;println(MPI.mpiexec_path)")
```

As an example, assuming that we are located on the root directory of `GridapDistributed.jl`,
an hypothetic MPI-parallel `GridapDistributed.jl` driver named `driver.jl` can be executed on 4 MPI tasks as:

```
$MPIRUN -np 4 julia --project=. -J sys-image.so driver.jl
```

where `-J sys-image.so` is optional, but highly recommended in order to reduce JIT compilation times. Here, `sys-image.so` is assumed to be a Julia system image pre-generated for the driver at hand using the [`PackageCompiler.jl`](https://julialang.github.io/PackageCompiler.jl/dev/index.html) package. See the `test/TestApp/compile` folder for example scripts with system image generation along with a test application with source available at `test/TestApp/`. These scripts are triggered from `.github/workflows/ci.yml` file on Github CI actions.


Two big warnings when executing MPI-parallel drivers:

 * Data race conditions associated to the generation of precompiled modules in cache. See [here](https://juliaparallel.github.io/MPI.jl/stable/knownissues/).

 * Each time that `GridapDistributed.jl` is modified, the first time that a parallel driver is executed, the program fails during MPI initialization. But the second, and subsequent times, it works ok. I still do not know the cause of the problem, but it is related to module precompilation as well.
