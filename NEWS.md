# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased 

A changelog is not maintained for this version. 

This version introduces fully-parallel distributed memory data structures for all the steps required in a finite element simulation (geometry handling, fe space setup, linear system assembly) except for the linear solver kernel, which is just a sparse LU solver applied to the global system gathered on a master task (and thus obviously not scalable). These distributed data structures mirror their counterparts in the Gridap.jl software architecture and implement most of their abstract interfaces. This version of GridapDistributed.jl relies on PartitionedArrays.jl (https://github.com/fverdugo/PartitionedArrays.jl) as distributed linear algebra backend (global distributed sparse matrices and vectors).

See also 

## [0.1.0] - 2021-09-29

A changelog is not maintained for this version.

This version although functional, is fully deprecated.
