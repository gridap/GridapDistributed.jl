module GridapDistributed

using Gridap
using Gridap.Geometry
using Gridap.Visualization

using PartitionedArrays
const PArrays = PartitionedArrays

include("DistributedDiscreteModels.jl")

include("Visualization.jl")

end # module
