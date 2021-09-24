# GridapDistributed

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapDistributed.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapDistributed.jl/dev)
![CI](https://github.com/Gridap/GridapDistributed.jl/workflows/CI/badge.svg)

Parallel distributed-memory version of `Gridap.jl`.  🚧 work in progress 🚧

| ![](https://user-images.githubusercontent.com/38347633/134634010-2be9b499-201b-4166-80ac-e161f6adceb0.png)   |  ![](https://user-images.githubusercontent.com/38347633/134634023-83f37646-f6b9-435c-9f9f-291dea9f86c2.png) 
|:-------------:|:-------------:|


## Purpose

This package is currently **experimental, under development**. In any case, the final purpose is to provide programming paradigm-neutral, parallel finite element data structures for distributed computing environments. This feature implies that communication among tasks are not tailored for a particular programming model, and thus can be leveraged with, e.g., MPI or the master-worker programming model built-in in Julia. Whenever one sticks to MPI as the underlying communication layer,  `GridapDistributed.jl` leverages the suite of tools available in the PETSc software package for the assembly and solution of distributed discrete systems of equations.

## Build 

Before using `GridapDistributed.jl` package, we have to build `MPI.jl` and `GridapDistributedPETScWrappers.jl`. We refer to the main [`README.md`](https://github.com/gridap/GridapDistributedPETScWrappers.jl) of the latter for configuration instructions.

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

As an example, the MPI-parallel `GridapDistributed.jl` driver `MPIPETScCommunicatorsTests.jl`, located in the `test` directory, can be executed as:

```
$MPIRUN -np 2 julia --project=. -J ../Gridap.jl/compile/Gridapv0.14.1.so test/MPIPETScTests/MPIPETScCommunicatorsTests.jl
```

where `-J ../Gridap.jl/compile/Gridapv0.14.1.so` is optional, but highly recommended in order to reduce JIT compilation times. More details about how to generate this file can be found [here](https://github.com/gridap/GridapDistributed.jl/blob/master/compile/README.md).


Two big warnings when executing MPI-parallel drivers:

 * Data race conditions associated to the generation of precompiled modules in cache. See [here](https://juliaparallel.github.io/MPI.jl/stable/knownissues/).

 * Each time that `GridapDistributed.jl` is modified, the first time that a parallel driver is executed, the program fails during MPI initialization. But the second, and subsequent times, it works ok. I still do not know the cause of the problem, but it is related to module precompilation as well.
