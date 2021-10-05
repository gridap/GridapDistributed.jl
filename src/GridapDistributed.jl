module GridapDistributed

using Gridap
using Gridap.Helpers
using Gridap.Arrays
using Gridap.Geometry
using Gridap.CellData
using Gridap.Visualization

using PartitionedArrays
const PArrays = PartitionedArrays

include("Geometry.jl")

include("CellData.jl")

include("Visualization.jl")

#include("FESpaces.jl")

end # module
