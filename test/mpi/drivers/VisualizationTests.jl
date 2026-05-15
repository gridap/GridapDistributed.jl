module VisualizationMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "VisualizationTests.jl"))

with_mpi() do distribute
  VisualizationTests.main(distribute, (2, 2))
end

end # module
