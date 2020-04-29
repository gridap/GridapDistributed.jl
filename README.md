# GridapDistributed

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapDistributed.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapDistributed.jl/dev)
[![Build Status](https://travis-ci.com/gridap/GridapDistributed.jl.svg?branch=master)](https://travis-ci.com/gridap/GridapDistributed.jl)
[![Codecov](https://codecov.io/gh/gridap/GridapDistributed.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/GridapDistributed.jl)


# Usage issues

`GridapDistributed.jl` uses, among others, the [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) Julia package. A pre-requisite of `MPI.jl` is a working MPI library installation on your system.
Thus, before adding `GridapDistributed.jl` to your Julia package environment, you need to ensure that this requirement is fulfilled. In Ubuntu 18.04 (this may work for other Ubuntu versions, but not tested) , this can be achieved by installing the following packages:
```shell
$ sudo apt-get install openmpi-bin libopenmpi-dev
```
