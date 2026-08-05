module ConstantFESpacesMPIDriver

using GridapDistributed
using PartitionedArrays
const PArrays = PartitionedArrays
using MPI

include(joinpath(@__DIR__, "..", "..", "ConstantFESpacesTests.jl"))

with_mpi() do distribute
  ConstantFESpacesTests.main(distribute, (2, 2))
end

end # module
