module GridapDistributed

using Gridap
using Gridap.Geometry
using Gridap.Visualization

using PartitionedArrays
const PArrays = PartitionedArrays

include("DiscreteModels.jl")

include("Visualization.jl")

end # module
