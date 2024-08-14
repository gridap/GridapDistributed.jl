# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.4] 2024-08-14

### Added

- Added kwargs for VTK encoding options. Since PR[#156](https://github.com/gridap/GridapDistributed.jl/pull/156).
- Reimplemented distributed ZeroMeanFESpaces. Since PR[#155](https://github.com/gridap/GridapDistributed.jl/pull/155).

### Fixed

- Fixed distributed interpolators for Vector-Valued FESpaces. Since PR[#152](https://github.com/gridap/GridapDistributed.jl/pull/152).

## [0.4.3] 2024-07-18

### Added

- Added distributed refinement of unstructured meshes. Since PR[#149](https://github.com/gridap/GridapDistributed.jl/pull/149).

- Added keyword arguments in the signature of the constructor of `DistributedMeasure`. Since PR[#150](https://github.com/gridap/GridapDistributed.jl/pull/150).

- Added DiracDelta in distributed setting. Since PR[#133](https://github.com/gridap/GridapDistributed.jl/pull/133).

## [0.4.2] 2024-07-4

### Added

- Added uniform anisotropic refinement of distributed cartesian meshes. Since PR[#148](https://github.com/gridap/GridapDistributed.jl/pull/148).

## [0.4.1] 2024-06-25

### Fixed

- Fixed bug in block-assembly whenever owners of touched dofs were not present in the local portion of the FESpace. Since PR[#147](https://github.com/gridap/GridapDistributed.jl/pull/147).

## [0.4.0] 2024-04-12

### Changed

- `DistributedCellField` now inherits from `CellField`. To accomodate the necessary API, we now save a pointer to the `DistributedTriangulation` where it is defined. This also requires `DistributedSingleFieldFESpace` to save the triangulation. Since PR[#141](https://github.com/gridap/GridapDistributed.jl/pull/141).
- All the distributed `Multifield` cellfield types are now represented by a `DistributedMultiFieldCellField`. Both `DistributedMultiFieldFEFunction` and `DistributedMultiFieldFEBasis` structs have been removed and replaced with constant aliases, which makes it more consistent with single-field types. Since PR[#141](https://github.com/gridap/GridapDistributed.jl/pull/141).
- Major refactor of ODE module. Implementation has been significantly simplified, while increasing the capability of the API. All `TransientDistributedObjects` structs have been removed, and replaced by `DistributedTransientObjects = DistributedObjects{TransientObject}`. Full support for EX/IM/IMEX methods. See Gridap's release for details. Since PR[#141](https://github.com/gridap/GridapDistributed.jl/pull/141).

## [0.3.6] 2024-01-28

### Added

- Added redistribution for MultiFieldFESpaces. Since PR [#140](https://github.com/gridap/GridapDistributed.jl/pull/140).

### Fixed

- Fixed issue [#142](https://github.com/gridap/GridapDistributed.jl/issues/142). Since PR [#142](https://github.com/gridap/GridapDistributed.jl/pull/142).

## [0.3.5] - 2023-12-04

### Added

- Added missing methods `allocate_in_range` and `allocate_in_domain` for distributed types. Since PR [#139](https://github.com/gridap/GridapDistributed.jl/pull/139).

## [0.3.4] - 2023-11-24

### Added

- Exporting `redistribute` function. Since [PR 136](https://github.com/gridap/GridapDistributed.jl/pull/136).

## [0.3.3] - 2023-11-22

### Added

- Added missing methods for `DistributedTransientFESpace`s. Since [PR 135](https://github.com/gridap/GridapDistributed.jl/pull/135).

## [0.3.2] - 2023-11-01

### Added

- Added support for distributed block-assembly. Since PR [124](https://github.com/gridap/GridapDistributed.jl/pull/124).
- Add possibility to use `OwnAndGhostVector` as vector partition for `FESpace` dofs. Since PR [124](https://github.com/gridap/GridapDistributed.jl/pull/124).
- Implement `BlockPArray <: AbstractBlockArray`, a new type that behaves as a `BlockArray{PArray}` and which fulfills the APIs of both `PArray` and `AbstractBlockArray`. This new type will be used to implement distributed block-assembly. Since PR [124](https://github.com/gridap/GridapDistributed.jl/pull/124).
- `DistributedMultiFieldFESpace{<:BlockMultiFieldStyle}` now has a `BlockPRange` as gids and `BlockPVector` as vector type. This is necessary to create consistency between fespace and system vectors, which in turn avoids memory allocations/copies when transferring between FESpace and linear system layouts. Since PR [124](https://github.com/gridap/GridapDistributed.jl/pull/124).

### Changed

- Merged functionalities of `consistent_local_views` and `change_ghost`. `consistent_local_views` has been removed. `change_ghost` now has two keywargs `is_consistent` and `make_consistent` that take into consideration all possible use cases. `change_ghost` has also been optimized to avoid unnecessary allocations if possible. Since PR [124](https://github.com/gridap/GridapDistributed.jl/pull/124).

## [0.3.1] - 2023-10-01

### Added

- Added missing _get_cell_dof_ids_inner_space() method overload. Since PR[130](https://github.com/gridap/GridapDistributed.jl/pull/130).
- Added missing remove_ghost_cells() overload for AdaptiveTriangulation. Since PR[131](https://github.com/gridap/GridapDistributed.jl/pull/131).

### Changed

- Updated compat for FillArrays to v1. Since PR[127](https://github.com/gridap/GridapDistributed.jl/pull/127).

## [0.3.0] - 2023-08-16

### Changed

- Porting the whole library to PartitionedArrays v0.3.x. Since PR [114](https://github.com/gridap/GridapDistributed.jl/pull/114)

### Added

- Tools for redistributing FE functions among meshes; added mock tests for `RedistributeGlue`. Since PR [114](https://github.com/gridap/GridapDistributed.jl/pull/114). This functionality was already somewhere else in the Gridap ecosystem of packages (in GridapSolvers.jl in particular).
- A variant of the PArrays `assemble_coo!` function named `assemble_coo_with_column_owner!` which also exchanges processor column owners of the entries. This variant is required to circumvent the current limitation of GridapDistributed.jl assembly for the case in which the following is not fullfilled: "each processor can determine locally with a single layer of ghost cells the global indices and associated processor owners of the rows that it touches after assembly of integration terms posed on locally-owned entities." Since PR [115](https://github.com/gridap/GridapDistributed.jl/pull/115).

### Fixed

- Added missing parameter to `allocate_jacobian`, needed after Gridap v0.17.18. Since PR [126](https://github.com/gridap/GridapDistributed.jl/pull/126). 

## [0.2.8] - 2023-07-31

### Added

- Reverted some changes introduced in PR [98](https://github.com/gridap/GridapDistributed.jl/pull/98). Eliminated `DistributedGridapType`. Functions `local_views` and `get_parts` now take argument of type `Any`. Since PR [117](https://github.com/gridap/GridapDistributed.jl/pull/117).

### Fixed

- Fixed bug where operating three or more `DistributedCellFields` would fail. Since PR [110](https://github.com/gridap/GridapDistributed.jl/pull/110)

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
