# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.7] - 2023-02-26

### Added

- Added `DistributedCellDof`, a distributed wrapper for `Gridap.CellDof`. This new wrapper acts on `DistributedCellField` in the same way `Gridap.CellDof` acts on `CellField`. Added `get_fe_dof_basis` function, which extracts a `DistributedCellDof` from a `DistributedFESpace`. Since PR [#97](https://github.com/gridap/GridapDistributed.jl/pull/97).
- Added `gather_free_and_dirichlet_values!` and `gather_free_values!` wrapper functions. Since PR [#97](https://github.com/gridap/GridapDistributed.jl/pull/97).
- Added compatibility with MPI v0.20 and PartitionedArrays v0.2.13. Since PR [#104](https://github.com/gridap/GridapDistributed.jl/pull/104).
- `DistributedDiscreteModel` is now an abstract class. The concrete implementation is now given by `GenericDistributedDiscreteModel`. Since PR [#98](https://github.com/gridap/GridapDistributed.jl/pull/98).
- New abstract type `DistributedGridapType`. All distributed structures now inherit from it. Implements two new API methods `local_views` and `get_parts`. Since PR [#98](https://github.com/gridap/GridapDistributed.jl/pull/98).
- Added support for adaptivity. Created `DistributedAdaptedDiscreteModel`. Since PR [#98](https://github.com/gridap/GridapDistributed.jl/pull/98).
- Added `RedistributeGlue`, which allows to redistribute model data between different communicators. Since PR [#98](https://github.com/gridap/GridapDistributed.jl/pull/98).

## [0.2.6] - 2022-06-07

### Added

- Support for parallel ODE solvers (GridapDistributed+GridapODEs). Since PR [#81](https://github.com/gridap/GridapDistributed.jl/pull/81)
- Support for parallel interface (surface) coupled problems. Since PR [#84](https://github.com/gridap/GridapDistributed.jl/pull/84)
- Added the missing zero_dirichlet_values used in Multifield.jl. Since PR [#87](https://github.com/gridap/GridapDistributed.jl/pull/87)
- Model now handles gids of all faces (not only cells) and support for FESpaces on lower-dim trians. Since PR [#86](https://github.com/gridap/GridapDistributed.jl/pull/86)

### Fixed

- Minor bug in the definition of the jacobian of the PLaplacian problem. Since PR [#88](https://github.com/gridap/GridapDistributed.jl/pull/88)

## [0.2.5] - 2022-02-14

### Added

- Support for periodic boundary conditions for `CartesianDiscreteModel`. Since PR [#79](https://github.com/gridap/GridapDistributed.jl/pull/79)
- Skeleton documentation and some content. Since PR [#77](https://github.com/gridap/GridapDistributed.jl/pull/77)
- Added `interpolate_everywhere` and `interpolate_dirichlet` functions. Since PR [#74](https://github.com/gridap/GridapDistributed.jl/pull/74)
- Added `createpvd` and `savepvd` functions to save collections of VTK files. Since PR [#71](https://github.com/gridap/GridapDistributed.jl/pull/71)

### Fixed

- Visualization of functions and numbers. Since PR [#78](https://github.com/gridap/GridapDistributed.jl/pull/78)

## [0.2.4] - 2021-12-09

### Fixed

- Bug-fix in global dof numbering. Since PR [#66](https://github.com/gridap/GridapDistributed.jl/pull/66)

## [0.2.3] - 2021-12-06

### Fixed

- RT FEs in parallel. Since PR [#64](https://github.com/gridap/GridapDistributed.jl/pull/64)

## [0.2.2] - 2021-11-27

### Added

- Added new overload for `SparseMatrixAssembler` to let one select the local matrix and vector types. Since PR [#63](https://github.com/gridap/GridapDistributed.jl/pull/63)

## [0.2.1] - 2021-11-25

### Added

- Added `num_cells` method to `DistributedDiscreteModel`. Since PR [#62](https://github.com/gridap/GridapDistributed.jl/pull/62)

## [0.2.0] - 2021-11-12

This version introduces fully-parallel distributed memory data structures for all the steps required in a finite element simulation (geometry handling, fe space setup, linear system assembly) except for the linear solver kernel, which is just a sparse LU solver applied to the global system gathered on a master task (and thus obviously not scalable, but very useful for debug and testing purposes). Parallel solvers are available in the [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl) package. The distributed data structures in GridapDistributed.jl mirror their counterparts in the Gridap.jl software architecture and implement most of their abstract interfaces. This version of GridapDistributed.jl relies on PartitionedArrays.jl (https://github.com/fverdugo/PartitionedArrays.jl) as distributed linear algebra backend (global distributed sparse matrices and vectors).

More details can also be found in https://github.com/gridap/GridapDistributed.jl/issues/39

## [0.1.0] - 2021-09-29

A changelog is not maintained for this version.

This version although functional, is fully deprecated.
