module AdaptivityCartesianMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "AdaptivityCartesianTests.jl"))

with_mpi() do distribute
  AdaptivityCartesianTests.main(distribute)
end

end # module
