module PolytopalCoarseningMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "PolytopalCoarseningTests.jl"))

with_mpi() do distribute
  PolytopalCoarseningTests.main(distribute, (2, 2))
end

end # module
