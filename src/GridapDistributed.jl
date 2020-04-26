module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Arrays

export SequentialCommunicator
export do_on_parts

include("Communicators.jl")

include("ScatteredVectors.jl")

include("GhostedVectors.jl")

include("DistributedDiscreteModels.jl")

include("CartesianDiscreteModels.jl")

include("DistributedFESpaces.jl")

end # module
