module GridapDistributed

using Gridap
using Gridap.Geometry
using Gridap.Visualization

using PartitionedArrays
const PArrays = PartitionedArrays

include("Geometry.jl")

include("Triangulations.jl")

include("Visualization.jl")

include("FESpaces.jl")

end # module
