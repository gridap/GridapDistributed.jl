#!/bin/bash

# This script is to be executed from this folder (compile/)
julia --project=../ --color=yes -e 'using Pkg; Pkg.instantiate()'
# See https://juliaparallel.github.io/MPI.jl/latest/knownissues/#Julia-module-precompilation-1
# for a justification of this line
julia --project=../ --color=yes -e 'using Pkg; pkg"precompile"'
julia --project=../ -O3 --check-bounds=no --color=yes compile.jl $1
