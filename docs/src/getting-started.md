# Getting Started

## Installation requirements

GridapDistributed is tested on Linux, but it should be also possible to use it on Mac OS and Windows since it is written exclusively in Julia and it only depends on registered Julia packages.

## Installation

GridapDistributed is a registered package. Thus, the installation should be straight forward using the Julia's package manager [Pkg](https://julialang.github.io/Pkg.jl/v1/). To this end, open the Julia REPL (i.e., execute the `julia` binary), type `]` to enter package mode, and install GridapDistributed as follows

```julia
pkg> add GridapDistributed
```

You will also need the `PartitionedArrays` package. 

```julia
pkg> add PartitionedArrays
```

If you want to leverage the satellite packages of `GridapDistributed.jl`, i.e., `GridapPETSc.jl`, `GridapP4est.jl`, and/or `GridapGmsh.jl`, you have to install them separately as 

```julia
pkg> add GridapPETSc
```

```julia
pkg> add GridapGmsh
```

```julia
pkg> add GridapP4est
```

Please note that these three packages depend on binary builds of the corresponding libraries they wrap (i.e., PETSc, Gmsh, and P4est). By default, they will leverage binary builds available at the Julia package registry. However, one may also use custom installations of these libraries. See the documentation of the corresponding package for more details


For further information about how to install and manage Julia packages, see the
[Pkg documentation](https://julialang.github.io/Pkg.jl/v1/).

## Further steps

If you are new to the `Gridap` ecosystem of packages, we recommend that you first follow the [Gridap Tutorials](https://gridap.github.io/Tutorials/dev/) step by step in order to get familiar with the `Gridap.jl` library. `GridapDistributed.jl` and `Gridap.jl` share almost the same high-level API. Therefore, some familiarity with `Gridap.jl` is highly recommended (if not essential) before starting with `GridapDistributed.jl`. 

If you are already familiarized with `Gridap.jl`, we recommend you to start straight away with the following [tutorial](https://gridap.github.io/Tutorials/dev/pages/t016_poisson_distributed).