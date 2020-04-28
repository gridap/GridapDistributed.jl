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

include("Communicators.jl")

include("DistributedData.jl")

include("ScatteredVectors.jl")

include("GhostedVectors.jl")

include("GloballyAddressableArrays.jl")

include("DistributedDiscreteModels.jl")

include("DistributedTriangulations.jl")

include("CartesianDiscreteModels.jl")

include("DistributedFESpaces.jl")

include("SparseMatrixAssemblers.jl")

end # module
