module AdaptivityMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "AdaptivityTests.jl"))

with_mpi() do distribute
  AdaptivityTests.main(distribute)
end

end # module
