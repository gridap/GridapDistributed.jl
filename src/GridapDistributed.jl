module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.Algebra

using MPI

export SequentialCommunicator
export MPICommunicator
export do_on_parts
export GloballyAddressableVector
export GloballyAddressableMatrix
export get_models_and_gids
export get_spaces_and_gids

include("Communicators.jl")

include("ScatteredVectors.jl")

include("GhostedVectors.jl")

include("DistributedDiscreteModels.jl")

include("CartesianDiscreteModels.jl")

include("DistributedFESpaces.jl")

include("GloballyAddressableArrays.jl")

include("SparseMatrixAssemblers.jl")

end # module
