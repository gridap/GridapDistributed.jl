module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.Algebra
using Gridap.MultiField

using SparseArrays
using MPI
using GridapDistributedPETScWrappers
using p4est_wrapper
using FillArrays
using SparseMatricesCSR

export Communicator
export num_parts
export num_workers
export get_part
export do_on_parts
export i_am_master
export SequentialCommunicator
export MPIPETScCommunicator

export DistributedData
export get_comm
export get_part_type
export gather
export gather!
export scatter
export scatter_value

export DistributedIndexSet
export IndexSet
export num_gids

export DistributedVector
export exchange!

export OwnedAndGhostCellsAssemblyStrategy
export OwnedCellsAssemblyStrategy
export MapDoFsTypeGlobal
export MapDoFsTypeProcLocal
export OwnedCells
export OwnedAndGhostCells

export remove_ghost_cells
export include_ghost_cells

export PETScLinearSolver
export UniformlyRefinedForestOfOctreesDiscreteModel


include("Communicators.jl")

include("SequentialCommunicators.jl")

include("MPIPETScCommunicators.jl")

include("MPITimers.jl")

include("DistributedData.jl")

include("SequentialDistributedData.jl")

include("MPIPETScDistributedData.jl")

include("DistributedIndexSets.jl")

include("SequentialDistributedIndexSets.jl")

include("MPIPETScDistributedIndexSets.jl")

include("DistributedVectors.jl")

include("SequentialDistributedVectors.jl")

include("MPIPETScDistributedVectors.jl")

include("DistributedDiscreteModels.jl")

include("CartesianDiscreteModels.jl")

include("UniformlyRefinedForestOfOctreesDiscreteModels.jl")

include("DistributedFESpaces.jl")

include("ZeroMeanDistributedFESpaces.jl")

include("MultiFieldDistributedFESpaces.jl")

include("DistributedFESpaceFactories.jl")

include("DistributedTriangulations.jl")

include("DistributedAssemblers.jl")

include("SequentialDistributedAssemblersInterfaces.jl")

include("MPIPETScAlgebraInterfaces.jl")

include("MPIPETScDistributedAssemblersInterfaces.jl")

include("DistributedFEOperators.jl")

include("MPIPETScLinearSolvers.jl")

import Gridap.TensorValues: inner, outer, double_contraction, symmetric_part
import LinearAlgebra: det, tr, cross, dot, ⋅
import Base: inv, abs, abs2, *, +, -, /, adjoint, transpose, real, imag, conj
include("GridapHighLevelAPI.jl")


end # module
