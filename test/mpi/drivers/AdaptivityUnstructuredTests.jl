module AdaptivityUnstructuredMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "AdaptivityUnstructuredTests.jl"))

with_mpi() do distribute
  AdaptivityUnstructuredTests.main(distribute)
end

end # module
