module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.Algebra

#using MPI

export Communicator
export num_parts
export num_workers
export do_on_parts
export i_am_master
export SequentialCommunicator

export DistributedData
export get_comm
export get_part_type
export gather
export gather!
export scatter

export DistributedIndexSet
export IndexSet
export DistributedVector
export exchange!

#export SequentialCommunicator
#export MPICommunicator
#export do_on_parts
#export GloballyAddressableVector
#export GloballyAddressableMatrix
#export ScatteredVector
#export scatter
#export gather
#export i_am_master
#export remove_ghost_cells
#export include_ghost_cells

include("Communicators.jl")

include("DistributedData.jl")

include("DistributedIndexSets.jl")

include("DistributedVectors.jl")

include("DistributedDiscreteModels.jl")

include("CartesianDiscreteModels.jl")

include("DistributedFESpaces.jl")

#
#include("ScatteredVectors.jl")
#
#include("GhostedVectors.jl")
#
#include("GloballyAddressableArrays.jl")
#
#include("DistributedTriangulations.jl")
#
#include("DistributedCellFields.jl")
#
#include("DistributedCellQuadratures.jl")
#
#include("DistributedDiscreteModels.jl")
#
#include("CartesianDiscreteModels.jl")
#
#include("DistributedFESpaces.jl")
#
#include("SparseMatrixAssemblers.jl")
#
#include("DistributedFETerms.jl")
#
#include("DistributedAssemblers.jl")

end # module
