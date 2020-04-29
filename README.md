# GridapDistributed

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapDistributed.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapDistributed.jl/dev)
[![Build Status](https://travis-ci.com/gridap/GridapDistributed.jl.svg?branch=master)](https://travis-ci.com/gridap/GridapDistributed.jl)
[![Codecov](https://codecov.io/gh/gridap/GridapDistributed.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/GridapDistributed.jl)


# Usage issues

`GridapDistributed` uses, among others, the [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) Julia package. As of today, the [build step](https://julialang.github.io/Pkg.jl/v1/creating-packages/index.html#Adding-a-build-step-to-the-package-1) of the current release of `MPI.jl` (`v0.13.1`), tries to find a dynamic library with the name `libmpi.so` on your system. The directories where it searches for such a file can be specified in a number of ways, e.g., via [`Unix` shell environment variables](https://github.com/JuliaParallel/MPI.jl/blob/v0.13.1/src/paths.jl) or `Julia` global variables (i.e., `Base.DL_LOAD_PATH`). If you don't set up any of these variables **explicitly**, `MPI.jl` falls backs to searching for `libmpi.so` in the list of directories that [`Libl.jl`](https://github.com/JuliaLang/julia/blob/v1.4.1/stdlib/Libdl/src/Libdl.jl) considers "system library paths" (at this point, I was not able to determine which directories are considered as such by `Libdl.jl`, if you know, please complete).

It turns out that, in Ubuntu v16.04 + OpenMPI (I did not check with other Ubuntu versions nor other Linux distros), we have the following:

1. None of the following Ubuntu packages: `libopenmpi2:amd64`, `openmpi-bin` nor `openmpi-common` create a `libmpi.so` file. You **NEED to install** `libopenmpi-dev` as well.
2. The packages mentioned in 1. deploy the OpenMPI library files under `/usr/lib/x86_64-linux-gnu/openmpi/lib`. (You can check that, e.g., with `dpkg-query -L libopenmpi-dev` on the Unix shell). It turns out that this directory is NOT in the in the list of directories that `Libl.jl` considers "system library paths", thus the build step of `MPI.jl` fails in finding OpenMPI.

The solution for 2. that I found is to add the following environment variable definition to the `.bashrc` file: `export JULIA_MPI_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib/`, and then, with this variable already set up on the shell, run `julia` and re-install `MPI.jl` again (see note below).

**NOTE**: If you have already a "broken" installation of `MPI.jl` within your julia Pkg environment, i.e., a previous installation failure that did not succeed, I recommend that you perform in sequence the following steps before trying to add `MPI.jl` again: (1) remove the `MPI.jl` package explicitly from the current environment (e.g., using `rm MPI` in the Pkg REPL); (2) remove the package installation directory from the file system, in my case under `~/.julia/packages/MPI/ZfFyE`. Using (1)+(2) you force that all build steps of `MPI.jl` are triggered when you add it again, and that the new files generated do not clash with the ones previously generated. 

**[@santiagobadia]** _I didn't need to remove the package installation directory or touch the `.bashrc`. Summarizing (?)_

```shell
$ sudo apt-get install openmpi-bin libopenmpi-dev
$ julia
```
```julia
(@v1.4) pkg> add MPI
```


