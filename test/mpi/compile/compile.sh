#!/bin/bash

if [ "$#" -ne 1 ]; then
    test_paths=$(ls ../../*Tests.jl)
    test_names=""
    for i in $test_paths 
    do	    
      name=$(basename $i)	    
      test_names="$test_names $name" 
    done
    echo "Illegal number of parameters"
    echo "Usage: $0 TESTNAME.jl, where TESTNAME.jl=$test_names"
    exit 1
fi

# This script is to be executed from this folder (compile/)
julia --project=../../.. --color=yes -e 'using Pkg; Pkg.instantiate()'
# See https://juliaparallel.github.io/MPI.jl/latest/knownissues/#Julia-module-precompilation-1
# for a justification of this line
julia --project=../../.. --color=yes -e 'using Pkg; pkg"precompile"'
julia --project=../../.. -O3 --check-bounds=no --color=yes compile.jl $1
