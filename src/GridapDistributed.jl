module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.Algebra

using MPI
using PETSc

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

export RowsComputedLocally
export OwnedCellsStrategy

export remove_ghost_cells
export include_ghost_cells

include("Communicators.jl")

include("MPIPETScCommunicators.jl")

include("DistributedData.jl")

include("MPIPETScDistributedData.jl")

include("DistributedIndexSets.jl")

include("MPIPETScDistributedIndexSets.jl")

include("DistributedVectors.jl")

include("DistributedDiscreteModels.jl")

include("CartesianDiscreteModels.jl")

include("DistributedFESpaces.jl")

include("DistributedAssemblers.jl")

include("DistributedFETerms.jl")

include("DistributedFEOperators.jl")

include("DistributedTriangulations.jl")

end # module
