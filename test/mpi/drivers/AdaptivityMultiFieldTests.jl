module AdaptivityMultiFieldMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "AdaptivityMultiFieldTests.jl"))

with_mpi() do distribute
  AdaptivityMultiFieldTests.main(distribute)
end

end # module
